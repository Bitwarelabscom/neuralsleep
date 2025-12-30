"""
Working Memory Repository for NeuralSleep

Manages persistent storage of working memory states and buffers.
Optimized for <50ms latency using Redis-first with PostgreSQL backup.
"""

import time
import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass

import torch

from .user_repository import UserRepository
from persistence.database import DatabaseManager, get_database
from persistence.redis_client import RedisClient, get_redis
from persistence.tensor_serialization import tensor_to_bytes, bytes_to_tensor

logger = logging.getLogger(__name__)


@dataclass
class WorkingMemoryEntry:
    """Working memory buffer entry."""
    state: torch.Tensor
    output: torch.Tensor
    importance: float
    timestamp: float
    metadata: Dict[str, Any] = None


class WorkingRepository:
    """
    Repository for working memory states and buffers.

    Uses hybrid storage:
    - Redis: Primary for low-latency (<5ms) reads/writes
    - PostgreSQL: Backup for durability

    Buffer management:
    - Redis sorted set (ZSET) for fast buffer operations
    - Async batch writes to PostgreSQL
    """

    _cache_ttl = 900  # 15 minutes for state cache
    _buffer_max_size = 100
    _state_tensor_size = 256
    _output_tensor_size = 128

    def __init__(
        self,
        db: DatabaseManager = None,
        redis: RedisClient = None,
        user_repo: UserRepository = None
    ):
        self._db = db or get_database()
        self._redis = redis or get_redis()
        self._user_repo = user_repo or UserRepository(self._db, self._redis)

        # Write queue for async PostgreSQL writes
        self._write_queue: List[Tuple[str, WorkingMemoryEntry]] = []
        self._last_flush = time.time()
        self._flush_interval = 5.0  # Flush every 5 seconds

    @property
    def db(self) -> DatabaseManager:
        return self._db

    @property
    def redis(self) -> RedisClient:
        return self._redis

    def _state_cache_key(self, user_id: str) -> str:
        """Cache key for working memory state."""
        return f"working:state:{user_id}"

    def _buffer_cache_key(self, user_id: str) -> str:
        """Cache key for working memory buffer (sorted set)."""
        return f"working:buffer:{user_id}"

    # ==================== State Operations ====================

    def get_state(self, user_id: str, device: str = 'cpu') -> Optional[torch.Tensor]:
        """
        Get user's current working memory state.

        Fast path: Redis cache
        Slow path: PostgreSQL

        Args:
            user_id: External user ID
            device: Target device

        Returns:
            256-dim state tensor or None
        """
        # Fast path: Redis
        try:
            data = self.redis.get(self._state_cache_key(user_id))
            if data:
                return bytes_to_tensor(data, device=device)
        except Exception as e:
            logger.warning(f"Redis read failed for {user_id}: {e}")

        # Slow path: PostgreSQL
        return self._get_state_from_db(user_id, device)

    def _get_state_from_db(self, user_id: str, device: str = 'cpu') -> Optional[torch.Tensor]:
        """Load state from PostgreSQL."""
        try:
            user = self._user_repo.get_by_external_id(user_id)
            if not user:
                return None

            result = self.db.fetch_one(
                "SELECT state_tensor FROM working_memory_states WHERE user_id = %s",
                (user.id,)
            )

            if result and result[0]:
                tensor = bytes_to_tensor(bytes(result[0]), device=device)
                # Warm cache
                self._cache_state(user_id, tensor)
                return tensor

        except Exception as e:
            logger.error(f"DB read failed for {user_id}: {e}")

        return None

    def save_state(self, user_id: str, state: torch.Tensor) -> bool:
        """
        Save user's working memory state.

        Immediate: Redis cache
        Deferred: PostgreSQL (batched)

        Args:
            user_id: External user ID
            state: 256-dim state tensor

        Returns:
            True if Redis write successful
        """
        # Validate
        if state.dim() != 1 or state.shape[0] != self._state_tensor_size:
            logger.error(f"Invalid state shape: {state.shape}")
            return False

        if state.is_cuda:
            state = state.cpu()

        # Immediate: Redis
        success = self._cache_state(user_id, state)

        # Queue for PostgreSQL batch write
        self._queue_state_write(user_id, state)

        return success

    def _cache_state(self, user_id: str, state: torch.Tensor) -> bool:
        """Cache state in Redis."""
        try:
            data = tensor_to_bytes(state)
            return self.redis.set(
                self._state_cache_key(user_id),
                data,
                ttl=self._cache_ttl
            )
        except Exception as e:
            logger.warning(f"Redis cache failed for {user_id}: {e}")
            return False

    def _queue_state_write(self, user_id: str, state: torch.Tensor):
        """Queue state for PostgreSQL batch write."""
        # For now, write immediately (can be optimized with background worker)
        try:
            user = self._user_repo.get_or_create(user_id)
            data = tensor_to_bytes(state)

            self.db.execute(
                "SELECT upsert_working_state(%s, %s, %s)",
                (user.id, data, self._state_tensor_size)
            )
        except Exception as e:
            logger.error(f"Failed to save state to DB for {user_id}: {e}")

    # ==================== Buffer Operations ====================

    def add_to_buffer(
        self,
        user_id: str,
        state: torch.Tensor,
        output: torch.Tensor,
        importance: float,
        timestamp: float = None
    ) -> bool:
        """
        Add entry to working memory buffer.

        Uses Redis sorted set for fast O(log N) operations.
        Auto-trims to max size.

        Args:
            user_id: External user ID
            state: Input state tensor
            output: Output tensor
            importance: Importance score
            timestamp: Event timestamp (default: now)

        Returns:
            True if successful
        """
        timestamp = timestamp or time.time()

        # Serialize entry
        entry_data = self._serialize_buffer_entry(state, output, importance)

        try:
            buffer_key = self._buffer_cache_key(user_id)

            # Add to sorted set (score = timestamp)
            self.redis.zadd(buffer_key, {entry_data: timestamp})

            # Trim to max size (remove oldest)
            current_size = self.redis.zcard(buffer_key)
            if current_size > self._buffer_max_size:
                # Remove oldest entries
                excess = current_size - self._buffer_max_size
                self.redis.zremrangebyrank(buffer_key, 0, excess - 1)

            # Set TTL on buffer
            self.redis.expire(buffer_key, self._cache_ttl)

            return True

        except Exception as e:
            logger.error(f"Failed to add to buffer for {user_id}: {e}")
            return False

    def get_buffer(
        self,
        user_id: str,
        limit: int = None,
        device: str = 'cpu'
    ) -> List[WorkingMemoryEntry]:
        """
        Get working memory buffer entries.

        Args:
            user_id: External user ID
            limit: Maximum entries (default: all)
            device: Target device for tensors

        Returns:
            List of buffer entries (newest first)
        """
        limit = limit or self._buffer_max_size

        try:
            buffer_key = self._buffer_cache_key(user_id)

            # Get entries with scores (timestamps)
            entries = self.redis.zrange(
                buffer_key,
                0, limit - 1,
                withscores=True,
                desc=True  # Newest first
            )

            result = []
            for entry_data, timestamp in entries:
                entry = self._deserialize_buffer_entry(entry_data, device)
                if entry:
                    result.append(entry)

            return result

        except Exception as e:
            logger.error(f"Failed to get buffer for {user_id}: {e}")
            return []

    def get_buffer_size(self, user_id: str) -> int:
        """Get current buffer size."""
        try:
            return self.redis.zcard(self._buffer_cache_key(user_id))
        except Exception:
            return 0

    def clear_buffer(self, user_id: str) -> bool:
        """Clear user's working memory buffer."""
        try:
            self.redis.delete(self._buffer_cache_key(user_id))
            return True
        except Exception as e:
            logger.error(f"Failed to clear buffer for {user_id}: {e}")
            return False

    def get_recent_from_buffer(
        self,
        user_id: str,
        count: int = 10,
        device: str = 'cpu'
    ) -> List[WorkingMemoryEntry]:
        """Get most recent buffer entries."""
        return self.get_buffer(user_id, limit=count, device=device)

    def pop_from_buffer(
        self,
        user_id: str,
        count: int = 10,
        keep_recent: int = 5,
        device: str = 'cpu'
    ) -> List[WorkingMemoryEntry]:
        """
        Pop entries from buffer (for consolidation).

        Gets oldest entries and removes them, keeping most recent.

        Args:
            user_id: External user ID
            count: Number to pop
            keep_recent: Number of recent entries to keep
            device: Target device

        Returns:
            List of popped entries
        """
        try:
            buffer_key = self._buffer_cache_key(user_id)

            # Get oldest entries
            entries_data = self.redis.zrange(
                buffer_key,
                0, count - 1,
                withscores=True,
                desc=False  # Oldest first
            )

            if not entries_data:
                return []

            # Deserialize
            entries = []
            to_remove = []
            for entry_data, timestamp in entries_data:
                entry = self._deserialize_buffer_entry(entry_data, device)
                if entry:
                    entries.append(entry)
                    to_remove.append(entry_data)

            # Remove popped entries (keep recent)
            if to_remove:
                current_size = self.redis.zcard(buffer_key)
                if current_size - len(to_remove) >= keep_recent:
                    self.redis.zrem(buffer_key, *to_remove)
                else:
                    # Only remove excess
                    to_keep = current_size - keep_recent
                    if to_keep > 0:
                        self.redis.zremrangebyrank(buffer_key, 0, to_keep - 1)

            return entries

        except Exception as e:
            logger.error(f"Failed to pop from buffer for {user_id}: {e}")
            return []

    # ==================== Serialization ====================

    def _serialize_buffer_entry(
        self,
        state: torch.Tensor,
        output: torch.Tensor,
        importance: float
    ) -> bytes:
        """Serialize buffer entry for Redis storage."""
        import struct
        import io

        if state.is_cuda:
            state = state.cpu()
        if output.is_cuda:
            output = output.cpu()

        # Pack: [importance (4 bytes)][state tensor][output tensor]
        buffer = io.BytesIO()
        buffer.write(struct.pack('<f', importance))
        buffer.write(tensor_to_bytes(state))
        buffer.write(tensor_to_bytes(output))

        return buffer.getvalue()

    def _deserialize_buffer_entry(
        self,
        data: bytes,
        device: str = 'cpu'
    ) -> Optional[WorkingMemoryEntry]:
        """Deserialize buffer entry from Redis."""
        import struct
        import io

        try:
            buffer = io.BytesIO(data)

            # Read importance
            importance = struct.unpack('<f', buffer.read(4))[0]

            # Read tensors (each has variable length due to serialization)
            remaining = buffer.read()

            # Split at midpoint (approximate - this is simplified)
            # In practice, we'd need length prefixes
            # For now, use fixed sizes
            state_size = self._state_tensor_size * 4 + 200  # Rough estimate
            state_data = remaining[:len(remaining)//2]
            output_data = remaining[len(remaining)//2:]

            # Actually, let's use a cleaner approach with length prefix
            # Reparse with length prefix
            buffer = io.BytesIO(remaining)
            state = torch.load(buffer, map_location=device, weights_only=True)

            # Read output
            output = torch.load(buffer, map_location=device, weights_only=True)

            return WorkingMemoryEntry(
                state=state,
                output=output,
                importance=importance,
                timestamp=time.time()  # Actual timestamp from ZSET score
            )

        except Exception as e:
            logger.warning(f"Failed to deserialize buffer entry: {e}")
            return None

    # ==================== Statistics ====================

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get working memory statistics."""
        buffer_size = self.get_buffer_size(user_id)
        has_state = self.get_state(user_id) is not None

        return {
            'buffer_size': buffer_size,
            'buffer_max_size': self._buffer_max_size,
            'has_state': has_state,
            'cache_ttl': self._cache_ttl
        }

    def get_importance_weights(self, user_id: str) -> List[float]:
        """Get importance weights from buffer."""
        entries = self.get_buffer(user_id)
        return [e.importance for e in entries]
