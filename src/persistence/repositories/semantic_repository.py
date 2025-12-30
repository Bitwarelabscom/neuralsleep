"""
Semantic Memory Repository for NeuralSleep

Manages persistent storage of semantic memory states (256-dim tensors).
Uses cache-through pattern: Redis cache + PostgreSQL persistence.
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime

import torch

from .base import BaseRepository
from .user_repository import UserRepository
from persistence.database import DatabaseManager
from persistence.redis_client import RedisClient
from persistence.tensor_serialization import tensor_to_bytes, bytes_to_tensor

logger = logging.getLogger(__name__)


class SemanticRepository(BaseRepository[torch.Tensor]):
    """
    Repository for semantic memory states.

    Stores 256-dimensional tensors representing user semantic models.
    Implements cache-through pattern for low-latency reads.
    """

    _cache_ttl = 3600  # 1 hour
    _tensor_size = 256

    def __init__(
        self,
        db: DatabaseManager = None,
        redis: RedisClient = None,
        user_repo: UserRepository = None
    ):
        super().__init__(db, redis)
        self._user_repo = user_repo or UserRepository(db, redis)

    def _cache_key(self, user_id: str) -> str:
        """Generate cache key for semantic state."""
        return f"semantic:{user_id}"

    def _serialize(self, value: torch.Tensor) -> bytes:
        """Serialize tensor for storage."""
        return tensor_to_bytes(value, auto_compress=True)

    def _deserialize(self, data: bytes) -> torch.Tensor:
        """Deserialize tensor from storage."""
        return bytes_to_tensor(data, device='cpu')

    def _get_from_db(self, external_user_id: str) -> Optional[torch.Tensor]:
        """Get semantic state from database by external user ID."""
        try:
            # Get internal user ID
            user = self._user_repo.get_by_external_id(external_user_id)
            if not user:
                return None

            result = self.db.fetch_one(
                """
                SELECT state_tensor FROM semantic_states
                WHERE user_id = %s
                """,
                (user.id,)
            )

            if result and result[0]:
                tensor_bytes = bytes(result[0])
                return self._deserialize(tensor_bytes)

        except Exception as e:
            logger.error(f"Failed to get semantic state for {external_user_id}: {e}")

        return None

    def _save_to_db(self, external_user_id: str, value: torch.Tensor) -> bool:
        """Save semantic state to database."""
        try:
            # Get or create user
            user = self._user_repo.get_or_create(external_user_id)

            # Serialize tensor
            tensor_bytes = self._serialize(value)

            # Upsert semantic state
            self.db.execute(
                """
                SELECT upsert_semantic_state(%s, %s, %s)
                """,
                (user.id, tensor_bytes, self._tensor_size)
            )

            logger.debug(f"Saved semantic state for {external_user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save semantic state for {external_user_id}: {e}")
            return False

    def _delete_from_db(self, external_user_id: str) -> bool:
        """Delete semantic state from database."""
        try:
            user = self._user_repo.get_by_external_id(external_user_id)
            if not user:
                return True  # Nothing to delete

            self.db.execute(
                "DELETE FROM semantic_states WHERE user_id = %s",
                (user.id,)
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete semantic state for {external_user_id}: {e}")
            return False

    # ==================== Semantic-Specific Methods ====================

    def get_user_state(self, user_id: str, device: str = 'cpu') -> Optional[torch.Tensor]:
        """
        Get user's semantic memory state.

        Args:
            user_id: External user ID (from MemoryCore)
            device: Target device for tensor

        Returns:
            256-dim tensor or None
        """
        tensor = self.get(user_id)
        if tensor is not None and device != 'cpu':
            tensor = tensor.to(device)
        return tensor

    def save_user_state(self, user_id: str, state: torch.Tensor) -> bool:
        """
        Save user's semantic memory state.

        Args:
            user_id: External user ID
            state: 256-dim tensor to save

        Returns:
            True if successful
        """
        # Validate tensor
        if state.dim() != 1 or state.shape[0] != self._tensor_size:
            logger.error(
                f"Invalid semantic state shape: expected ({self._tensor_size},), "
                f"got {state.shape}"
            )
            return False

        # Move to CPU for storage
        if state.is_cuda:
            state = state.cpu()

        return self.save(user_id, state)

    def get_or_create_state(
        self,
        user_id: str,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Get user state or create new zero-initialized state.

        Args:
            user_id: External user ID
            device: Target device

        Returns:
            Semantic state tensor (existing or new)
        """
        state = self.get_user_state(user_id, device)
        if state is None:
            state = torch.zeros(self._tensor_size, device=device)
            # Don't save zero state - wait for actual data
        return state

    def batch_load_users(
        self,
        user_ids: List[str],
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Batch load semantic states for multiple users.

        Args:
            user_ids: List of external user IDs
            device: Target device

        Returns:
            Dict of user_id -> tensor
        """
        result = {}
        for user_id in user_ids:
            state = self.get_user_state(user_id, device)
            if state is not None:
                result[user_id] = state
        return result

    def get_state_version(self, user_id: str) -> int:
        """
        Get current version of user's semantic state.

        Args:
            user_id: External user ID

        Returns:
            Version number (0 if no state)
        """
        try:
            user = self._user_repo.get_by_external_id(user_id)
            if not user:
                return 0

            result = self.db.fetch_one(
                "SELECT version FROM semantic_states WHERE user_id = %s",
                (user.id,)
            )
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Failed to get state version for {user_id}: {e}")
            return 0

    def get_all_user_ids_with_states(self, limit: int = 1000) -> List[str]:
        """
        Get external IDs of all users with semantic states.

        Args:
            limit: Maximum number of users

        Returns:
            List of external user IDs
        """
        try:
            result = self.db.fetch_all(
                """
                SELECT u.external_id
                FROM semantic_states ss
                JOIN users u ON u.id = ss.user_id
                ORDER BY ss.updated_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            return [row[0] for row in result]

        except Exception as e:
            logger.error(f"Failed to get user IDs with states: {e}")
            return []

    def count_states(self) -> int:
        """
        Get total count of semantic states.

        Returns:
            Number of stored states
        """
        try:
            result = self.db.fetch_one("SELECT COUNT(*) FROM semantic_states")
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to count semantic states: {e}")
            return 0

    def get_stats(self) -> Dict:
        """
        Get repository statistics.

        Returns:
            Statistics dictionary
        """
        try:
            result = self.db.fetch_dict(
                """
                SELECT
                    COUNT(*) as total_states,
                    AVG(tensor_size) as avg_tensor_size,
                    MAX(updated_at) as last_updated,
                    MIN(created_at) as oldest_state
                FROM semantic_states
                """
            )
            if result:
                stats = result[0]
                return {
                    'total_states': stats['total_states'] or 0,
                    'avg_tensor_size': float(stats['avg_tensor_size'] or 0),
                    'last_updated': stats['last_updated'].isoformat() if stats['last_updated'] else None,
                    'oldest_state': stats['oldest_state'].isoformat() if stats['oldest_state'] else None,
                    'cache_ttl': self._cache_ttl
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")

        return {'total_states': 0, 'cache_ttl': self._cache_ttl}
