"""
Base Repository for NeuralSleep

Abstract base class implementing cache-through pattern
with Redis caching and PostgreSQL persistence.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, TypeVar, Generic

from persistence.database import DatabaseManager, get_database
from persistence.redis_client import RedisClient, get_redis
from persistence.tensor_serialization import tensor_to_bytes, bytes_to_tensor

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository with cache-through pattern.

    Implements:
    - Read: Cache first, then DB, populate cache on miss
    - Write: Write to DB, then update cache
    - Delete: Delete from DB, then invalidate cache

    Subclasses must implement:
    - _cache_key(): Generate Redis cache key
    - _cache_ttl: Cache TTL in seconds
    - _serialize() / _deserialize(): Data conversion
    """

    # Default cache TTL (override in subclass)
    _cache_ttl: int = 3600  # 1 hour

    def __init__(
        self,
        db: DatabaseManager = None,
        redis: RedisClient = None
    ):
        """
        Initialize repository.

        Args:
            db: Database manager (uses global if not provided)
            redis: Redis client (uses global if not provided)
        """
        self._db = db
        self._redis = redis
        self._fallback_cache: Dict[str, Any] = {}  # In-memory fallback

    @property
    def db(self) -> DatabaseManager:
        """Get database manager (lazy initialization)."""
        if self._db is None:
            self._db = get_database()
        return self._db

    @property
    def redis(self) -> RedisClient:
        """Get Redis client (lazy initialization)."""
        if self._redis is None:
            self._redis = get_redis()
        return self._redis

    @abstractmethod
    def _cache_key(self, identifier: str) -> str:
        """
        Generate cache key for given identifier.

        Args:
            identifier: Entity identifier (e.g., user_id)

        Returns:
            Redis cache key
        """
        pass

    def _serialize(self, value: T) -> bytes:
        """
        Serialize value for storage.

        Override for custom serialization.
        Default: Use tensor_to_bytes for torch tensors.
        """
        import torch
        if isinstance(value, torch.Tensor):
            return tensor_to_bytes(value)
        raise NotImplementedError(
            f"Serialization not implemented for {type(value)}"
        )

    def _deserialize(self, data: bytes) -> T:
        """
        Deserialize value from storage.

        Override for custom deserialization.
        Default: Use bytes_to_tensor.
        """
        return bytes_to_tensor(data)

    # ==================== Cache Operations ====================

    def _get_from_cache(self, identifier: str) -> Optional[T]:
        """
        Get value from Redis cache.

        Args:
            identifier: Entity identifier

        Returns:
            Cached value or None
        """
        try:
            key = self._cache_key(identifier)
            data = self.redis.get(key)
            if data is not None:
                return self._deserialize(data)
        except Exception as e:
            logger.warning(f"Cache read error for {identifier}: {e}")
            # Try fallback cache
            key = self._cache_key(identifier)
            if key in self._fallback_cache:
                return self._fallback_cache[key]
        return None

    def _set_in_cache(
        self,
        identifier: str,
        value: T,
        ttl: int = None
    ) -> bool:
        """
        Set value in Redis cache.

        Args:
            identifier: Entity identifier
            value: Value to cache
            ttl: TTL override

        Returns:
            True if successful
        """
        try:
            key = self._cache_key(identifier)
            data = self._serialize(value)
            self.redis.set(key, data, ttl=ttl or self._cache_ttl)
            # Also update fallback
            self._fallback_cache[key] = value
            return True
        except Exception as e:
            logger.warning(f"Cache write error for {identifier}: {e}")
            # Still update fallback
            key = self._cache_key(identifier)
            self._fallback_cache[key] = value
            return False

    def _delete_from_cache(self, identifier: str) -> bool:
        """
        Delete value from cache.

        Args:
            identifier: Entity identifier

        Returns:
            True if successful
        """
        try:
            key = self._cache_key(identifier)
            self.redis.delete(key)
            self._fallback_cache.pop(key, None)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for {identifier}: {e}")
            return False

    # ==================== Abstract DB Operations ====================

    @abstractmethod
    def _get_from_db(self, identifier: str) -> Optional[T]:
        """
        Get value from database.

        Args:
            identifier: Entity identifier

        Returns:
            Value from DB or None
        """
        pass

    @abstractmethod
    def _save_to_db(self, identifier: str, value: T) -> bool:
        """
        Save value to database.

        Args:
            identifier: Entity identifier
            value: Value to save

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def _delete_from_db(self, identifier: str) -> bool:
        """
        Delete value from database.

        Args:
            identifier: Entity identifier

        Returns:
            True if successful
        """
        pass

    # ==================== Public Interface ====================

    def get(self, identifier: str) -> Optional[T]:
        """
        Get value with cache-through pattern.

        1. Check cache
        2. If miss, load from DB
        3. Populate cache on hit

        Args:
            identifier: Entity identifier

        Returns:
            Value or None
        """
        # Try cache first
        value = self._get_from_cache(identifier)
        if value is not None:
            logger.debug(f"Cache hit: {identifier}")
            return value

        # Cache miss - load from DB
        logger.debug(f"Cache miss: {identifier}")
        value = self._get_from_db(identifier)

        # Populate cache on hit
        if value is not None:
            self._set_in_cache(identifier, value)

        return value

    def save(self, identifier: str, value: T) -> bool:
        """
        Save value with write-through pattern.

        1. Write to DB
        2. Update cache

        Args:
            identifier: Entity identifier
            value: Value to save

        Returns:
            True if DB write successful
        """
        # Write to DB first
        success = self._save_to_db(identifier, value)

        if success:
            # Update cache
            self._set_in_cache(identifier, value)
        else:
            # Invalidate cache on failure
            self._delete_from_cache(identifier)

        return success

    def delete(self, identifier: str) -> bool:
        """
        Delete value from both stores.

        1. Delete from DB
        2. Invalidate cache

        Args:
            identifier: Entity identifier

        Returns:
            True if successful
        """
        # Delete from DB first
        success = self._delete_from_db(identifier)

        # Always invalidate cache
        self._delete_from_cache(identifier)

        return success

    def exists(self, identifier: str) -> bool:
        """
        Check if value exists.

        Args:
            identifier: Entity identifier

        Returns:
            True if exists
        """
        return self.get(identifier) is not None

    def invalidate_cache(self, identifier: str) -> bool:
        """
        Force cache invalidation.

        Args:
            identifier: Entity identifier

        Returns:
            True if successful
        """
        return self._delete_from_cache(identifier)

    def refresh(self, identifier: str) -> Optional[T]:
        """
        Force refresh from DB to cache.

        Args:
            identifier: Entity identifier

        Returns:
            Fresh value or None
        """
        # Invalidate cache
        self._delete_from_cache(identifier)

        # Reload from DB
        value = self._get_from_db(identifier)

        # Populate cache
        if value is not None:
            self._set_in_cache(identifier, value)

        return value

    # ==================== Batch Operations ====================

    def batch_get(self, identifiers: List[str]) -> Dict[str, Optional[T]]:
        """
        Get multiple values efficiently.

        Args:
            identifiers: List of identifiers

        Returns:
            Dict of identifier -> value (None for missing)
        """
        result = {}
        cache_misses = []

        # Check cache for all
        for identifier in identifiers:
            value = self._get_from_cache(identifier)
            if value is not None:
                result[identifier] = value
            else:
                cache_misses.append(identifier)

        # Load misses from DB
        if cache_misses:
            for identifier in cache_misses:
                value = self._get_from_db(identifier)
                result[identifier] = value
                if value is not None:
                    self._set_in_cache(identifier, value)

        return result

    # ==================== Health Check ====================

    def health_check(self) -> Dict[str, bool]:
        """
        Check repository health.

        Returns:
            Dict with db_healthy and cache_healthy
        """
        db_healthy = self.db.health_check()
        cache_healthy = self.redis.health_check()

        return {
            'db_healthy': db_healthy,
            'cache_healthy': cache_healthy,
            'fallback_active': not cache_healthy
        }
