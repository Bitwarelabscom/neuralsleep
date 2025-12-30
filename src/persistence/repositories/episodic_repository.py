"""
Episodic Memory Repository for NeuralSleep

Manages persistent storage of episodic experiences (128-dim tensors).
Supports time-windowed queries and consolidation tracking.
"""

import json
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import torch

from .base import BaseRepository
from .user_repository import UserRepository
from persistence.database import DatabaseManager
from persistence.redis_client import RedisClient
from persistence.tensor_serialization import tensor_to_bytes, bytes_to_tensor

logger = logging.getLogger(__name__)


@dataclass
class EpisodicExperience:
    """Episodic experience entity."""
    id: str
    user_id: str
    tensor: torch.Tensor
    importance: float
    event_timestamp: datetime
    event_type: str = None
    character_id: str = None
    correct: bool = None
    consolidated: bool = False
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without tensor)."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'importance': self.importance,
            'event_timestamp': self.event_timestamp.isoformat() if self.event_timestamp else None,
            'event_type': self.event_type,
            'character_id': self.character_id,
            'correct': self.correct,
            'consolidated': self.consolidated,
            'metadata': self.metadata or {}
        }


class EpisodicRepository:
    """
    Repository for episodic memory experiences.

    Stores 128-dimensional tensors with metadata for each learning event.
    Supports time-windowed queries and batch operations.
    """

    _cache_ttl = 900  # 15 minutes
    _tensor_size = 128

    def __init__(
        self,
        db: DatabaseManager = None,
        redis: RedisClient = None,
        user_repo: UserRepository = None
    ):
        from persistence.database import get_database
        from persistence.redis_client import get_redis

        self._db = db or get_database()
        self._redis = redis or get_redis()
        self._user_repo = user_repo or UserRepository(self._db, self._redis)

    @property
    def db(self) -> DatabaseManager:
        return self._db

    @property
    def redis(self) -> RedisClient:
        return self._redis

    def _cache_key(self, user_id: str) -> str:
        """Generate cache key for user's recent experiences."""
        return f"episodic:recent:{user_id}"

    # ==================== Store Operations ====================

    def store_experience(
        self,
        user_id: str,
        tensor: torch.Tensor,
        importance: float,
        event_timestamp: datetime = None,
        event_type: str = None,
        character_id: str = None,
        correct: bool = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        """
        Store a single episodic experience.

        Args:
            user_id: External user ID
            tensor: 128-dim experience tensor
            importance: Importance score (0-1)
            event_timestamp: When event occurred
            event_type: Type of event (practice, hint, etc.)
            character_id: Character involved
            correct: Whether response was correct
            metadata: Additional metadata

        Returns:
            Experience ID or None on failure
        """
        try:
            # Get or create user
            user = self._user_repo.get_or_create(user_id)

            # Validate tensor
            if tensor.dim() != 1 or tensor.shape[0] != self._tensor_size:
                logger.error(f"Invalid episodic tensor shape: {tensor.shape}")
                return None

            # Serialize tensor
            if tensor.is_cuda:
                tensor = tensor.cpu()
            tensor_bytes = tensor_to_bytes(tensor)

            # Insert into database
            result = self.db.fetch_one(
                """
                INSERT INTO episodic_experiences
                (user_id, experience_tensor, tensor_size, importance,
                 event_timestamp, event_type, character_id, correct, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    user.id,
                    tensor_bytes,
                    self._tensor_size,
                    importance,
                    event_timestamp or datetime.now(),
                    event_type,
                    character_id,
                    correct,
                    json.dumps(metadata or {})
                )
            )

            if result:
                exp_id = str(result[0])
                # Invalidate recent cache
                self._invalidate_recent_cache(user_id)
                logger.debug(f"Stored experience {exp_id} for {user_id}")
                return exp_id

        except Exception as e:
            logger.error(f"Failed to store experience for {user_id}: {e}")

        return None

    def batch_store_experiences(
        self,
        user_id: str,
        experiences: List[Dict[str, Any]]
    ) -> int:
        """
        Batch store multiple experiences.

        Args:
            user_id: External user ID
            experiences: List of experience dicts with keys:
                - tensor: torch.Tensor
                - importance: float
                - event_timestamp: datetime (optional)
                - event_type: str (optional)
                - character_id: str (optional)
                - correct: bool (optional)

        Returns:
            Number of experiences stored
        """
        if not experiences:
            return 0

        try:
            # Get or create user
            user = self._user_repo.get_or_create(user_id)

            # Prepare batch
            params_list = []
            for exp in experiences:
                tensor = exp.get('tensor')
                if tensor is None:
                    continue

                if tensor.is_cuda:
                    tensor = tensor.cpu()

                params_list.append((
                    user.id,
                    tensor_to_bytes(tensor),
                    self._tensor_size,
                    exp.get('importance', 0.5),
                    exp.get('event_timestamp', datetime.now()),
                    exp.get('event_type'),
                    exp.get('character_id'),
                    exp.get('correct'),
                    json.dumps(exp.get('metadata', {}))
                ))

            if not params_list:
                return 0

            # Batch insert
            count = self.db.execute_many(
                """
                INSERT INTO episodic_experiences
                (user_id, experience_tensor, tensor_size, importance,
                 event_timestamp, event_type, character_id, correct, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                params_list
            )

            # Invalidate cache
            self._invalidate_recent_cache(user_id)

            logger.debug(f"Batch stored {count} experiences for {user_id}")
            return count

        except Exception as e:
            logger.error(f"Failed to batch store experiences for {user_id}: {e}")
            return 0

    # ==================== Query Operations ====================

    def get_user_experiences(
        self,
        user_id: str,
        time_window: str = 'all',
        limit: int = 1000,
        include_consolidated: bool = True,
        device: str = 'cpu'
    ) -> List[EpisodicExperience]:
        """
        Get user's episodic experiences.

        Args:
            user_id: External user ID
            time_window: 'hour', 'day', 'week', 'month', 'all'
            limit: Maximum experiences to return
            include_consolidated: Include already consolidated experiences
            device: Target device for tensors

        Returns:
            List of EpisodicExperience objects
        """
        try:
            user = self._user_repo.get_by_external_id(user_id)
            if not user:
                return []

            # Build time filter
            time_filter = ""
            time_param = None
            if time_window == 'hour':
                time_filter = "AND event_timestamp > NOW() - INTERVAL '1 hour'"
            elif time_window == 'day':
                time_filter = "AND event_timestamp > NOW() - INTERVAL '1 day'"
            elif time_window == 'week':
                time_filter = "AND event_timestamp > NOW() - INTERVAL '1 week'"
            elif time_window == 'month':
                time_filter = "AND event_timestamp > NOW() - INTERVAL '1 month'"

            # Build consolidated filter
            consolidated_filter = "" if include_consolidated else "AND NOT consolidated"

            query = f"""
                SELECT id, experience_tensor, importance, event_timestamp,
                       event_type, character_id, correct, consolidated, metadata
                FROM episodic_experiences
                WHERE user_id = %s {time_filter} {consolidated_filter}
                ORDER BY event_timestamp DESC
                LIMIT %s
            """

            result = self.db.fetch_dict(query, (user.id, limit))

            experiences = []
            for row in result:
                tensor = bytes_to_tensor(bytes(row['experience_tensor']), device=device)
                experiences.append(EpisodicExperience(
                    id=str(row['id']),
                    user_id=user_id,
                    tensor=tensor,
                    importance=row['importance'],
                    event_timestamp=row['event_timestamp'],
                    event_type=row.get('event_type'),
                    character_id=row.get('character_id'),
                    correct=row.get('correct'),
                    consolidated=row['consolidated'],
                    metadata=row.get('metadata', {})
                ))

            return experiences

        except Exception as e:
            logger.error(f"Failed to get experiences for {user_id}: {e}")
            return []

    def get_unconsolidated_experiences(
        self,
        user_id: str,
        limit: int = 100,
        device: str = 'cpu'
    ) -> List[EpisodicExperience]:
        """
        Get experiences not yet consolidated.

        Args:
            user_id: External user ID
            limit: Maximum experiences
            device: Target device

        Returns:
            List of unconsolidated experiences
        """
        return self.get_user_experiences(
            user_id,
            time_window='all',
            limit=limit,
            include_consolidated=False,
            device=device
        )

    def get_high_importance_experiences(
        self,
        user_id: str,
        min_importance: float = 0.7,
        limit: int = 100,
        device: str = 'cpu'
    ) -> List[EpisodicExperience]:
        """
        Get high-importance experiences.

        Args:
            user_id: External user ID
            min_importance: Minimum importance threshold
            limit: Maximum experiences
            device: Target device

        Returns:
            List of high-importance experiences
        """
        try:
            user = self._user_repo.get_by_external_id(user_id)
            if not user:
                return []

            result = self.db.fetch_dict(
                """
                SELECT id, experience_tensor, importance, event_timestamp,
                       event_type, character_id, correct, consolidated, metadata
                FROM episodic_experiences
                WHERE user_id = %s AND importance >= %s
                ORDER BY importance DESC, event_timestamp DESC
                LIMIT %s
                """,
                (user.id, min_importance, limit)
            )

            experiences = []
            for row in result:
                tensor = bytes_to_tensor(bytes(row['experience_tensor']), device=device)
                experiences.append(EpisodicExperience(
                    id=str(row['id']),
                    user_id=user_id,
                    tensor=tensor,
                    importance=row['importance'],
                    event_timestamp=row['event_timestamp'],
                    event_type=row.get('event_type'),
                    character_id=row.get('character_id'),
                    correct=row.get('correct'),
                    consolidated=row['consolidated'],
                    metadata=row.get('metadata', {})
                ))

            return experiences

        except Exception as e:
            logger.error(f"Failed to get high-importance experiences for {user_id}: {e}")
            return []

    # ==================== Consolidation Operations ====================

    def mark_consolidated(self, experience_ids: List[str]) -> int:
        """
        Mark experiences as consolidated.

        Args:
            experience_ids: List of experience IDs

        Returns:
            Number of experiences marked
        """
        if not experience_ids:
            return 0

        try:
            result = self.db.fetch_one(
                "SELECT mark_experiences_consolidated(%s::uuid[])",
                (experience_ids,)
            )
            return result[0] if result else 0

        except Exception as e:
            logger.error(f"Failed to mark experiences consolidated: {e}")
            return 0

    def get_consolidation_candidates(
        self,
        user_id: str,
        min_count: int = 10,
        max_age_hours: int = 24
    ) -> List[EpisodicExperience]:
        """
        Get experiences ready for consolidation.

        Args:
            user_id: External user ID
            min_count: Minimum experiences needed
            max_age_hours: Maximum age for consolidation

        Returns:
            List of consolidation candidates
        """
        experiences = self.get_unconsolidated_experiences(
            user_id,
            limit=min_count * 2
        )

        if len(experiences) < min_count:
            return []

        # Filter by age
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        return [e for e in experiences if e.event_timestamp > cutoff]

    # ==================== Cleanup Operations ====================

    def delete_old_experiences(
        self,
        user_id: str,
        older_than_days: int = 90
    ) -> int:
        """
        Delete old consolidated experiences.

        Args:
            user_id: External user ID
            older_than_days: Delete experiences older than this

        Returns:
            Number deleted
        """
        try:
            user = self._user_repo.get_by_external_id(user_id)
            if not user:
                return 0

            result = self.db.fetch_one(
                """
                DELETE FROM episodic_experiences
                WHERE user_id = %s
                  AND consolidated = TRUE
                  AND event_timestamp < NOW() - INTERVAL '%s days'
                RETURNING COUNT(*)
                """,
                (user.id, older_than_days)
            )

            count = result[0] if result else 0
            if count > 0:
                self._invalidate_recent_cache(user_id)
            return count

        except Exception as e:
            logger.error(f"Failed to delete old experiences for {user_id}: {e}")
            return 0

    # ==================== Cache Operations ====================

    def _invalidate_recent_cache(self, user_id: str):
        """Invalidate user's recent experiences cache."""
        try:
            self.redis.delete(self._cache_key(user_id))
        except Exception as e:
            logger.warning(f"Failed to invalidate cache for {user_id}: {e}")

    # ==================== Statistics ====================

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics for user's episodic memory.

        Args:
            user_id: External user ID

        Returns:
            Statistics dictionary
        """
        try:
            user = self._user_repo.get_by_external_id(user_id)
            if not user:
                return {}

            result = self.db.fetch_dict(
                """
                SELECT
                    COUNT(*) as total_experiences,
                    COUNT(*) FILTER (WHERE consolidated) as consolidated_count,
                    COUNT(*) FILTER (WHERE NOT consolidated) as pending_count,
                    AVG(importance) as avg_importance,
                    MAX(event_timestamp) as latest_event,
                    MIN(event_timestamp) as earliest_event
                FROM episodic_experiences
                WHERE user_id = %s
                """,
                (user.id,)
            )

            if result:
                stats = result[0]
                return {
                    'total_experiences': stats['total_experiences'] or 0,
                    'consolidated_count': stats['consolidated_count'] or 0,
                    'pending_count': stats['pending_count'] or 0,
                    'avg_importance': float(stats['avg_importance'] or 0),
                    'latest_event': stats['latest_event'].isoformat() if stats['latest_event'] else None,
                    'earliest_event': stats['earliest_event'].isoformat() if stats['earliest_event'] else None
                }

        except Exception as e:
            logger.error(f"Failed to get stats for {user_id}: {e}")

        return {}

    def count_experiences(self) -> int:
        """Get total experience count."""
        try:
            result = self.db.fetch_one("SELECT COUNT(*) FROM episodic_experiences")
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to count experiences: {e}")
            return 0
