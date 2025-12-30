"""
Redis Client for NeuralSleep

Provides caching, distributed locking, and pub/sub capabilities
for high-performance state management.
"""

import os
import time
import uuid
import logging
from typing import Optional, Any, Dict, List, Tuple, Union
from contextlib import contextmanager

import redis
from redis import ConnectionPool, Redis

logger = logging.getLogger(__name__)


class RedisError(Exception):
    """Base exception for Redis operations."""
    pass


class LockAcquisitionError(RedisError):
    """Failed to acquire distributed lock."""
    pass


class RedisClient:
    """
    Redis client with connection pooling and high-level operations.

    Usage:
        client = RedisClient()
        client.set('key', b'value', ttl=3600)
        value = client.get('key')
    """

    # Key prefix for all NeuralSleep keys
    KEY_PREFIX = 'neuralsleep:'

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = 0,
        password: str = None,
        max_connections: int = 50,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 2.0,
        retry_on_timeout: bool = True
    ):
        """
        Initialize Redis client.

        Args:
            host: Redis host (default: from env)
            port: Redis port (default: from env)
            db: Redis database number
            password: Redis password
            max_connections: Maximum pool connections
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Retry on timeout errors
        """
        self.host = host or os.getenv('REDIS_HOST', 'localhost')
        self.port = port or int(os.getenv('REDIS_PORT', 6381))
        self.db = db
        self.password = password or os.getenv('REDIS_PASSWORD', None)

        self.max_connections = max_connections
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout

        self._pool: Optional[ConnectionPool] = None
        self._client: Optional[Redis] = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the connection pool.

        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            return True

        try:
            self._pool = ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                socket_timeout=self.socket_timeout,
                socket_connect_timeout=self.socket_connect_timeout,
                retry_on_timeout=self.retry_on_timeout,
                decode_responses=False  # We handle binary data
            )
            self._client = Redis(connection_pool=self._pool)

            # Test connection
            self._client.ping()

            self._initialized = True
            logger.info(
                f"Redis client initialized: {self.host}:{self.port} "
                f"(max_connections={self.max_connections})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            return False

    def _ensure_initialized(self):
        """Ensure client is initialized."""
        if not self._initialized:
            if not self.initialize():
                raise RedisError("Redis client not initialized")

    def _prefixed_key(self, key: str) -> str:
        """Add prefix to key."""
        if key.startswith(self.KEY_PREFIX):
            return key
        return f"{self.KEY_PREFIX}{key}"

    # ==================== Basic Operations ====================

    def get(self, key: str) -> Optional[bytes]:
        """
        Get a value by key.

        Args:
            key: Cache key

        Returns:
            Value bytes or None if not found
        """
        self._ensure_initialized()
        return self._client.get(self._prefixed_key(key))

    def set(
        self,
        key: str,
        value: bytes,
        ttl: int = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set a value with optional TTL.

        Args:
            key: Cache key
            value: Value bytes
            ttl: Time-to-live in seconds
            nx: Only set if key doesn't exist
            xx: Only set if key exists

        Returns:
            True if set successfully
        """
        self._ensure_initialized()
        return bool(self._client.set(
            self._prefixed_key(key),
            value,
            ex=ttl,
            nx=nx,
            xx=xx
        ))

    def delete(self, *keys: str) -> int:
        """
        Delete one or more keys.

        Args:
            keys: Keys to delete

        Returns:
            Number of keys deleted
        """
        self._ensure_initialized()
        prefixed = [self._prefixed_key(k) for k in keys]
        return self._client.delete(*prefixed)

    def exists(self, *keys: str) -> int:
        """
        Check if keys exist.

        Args:
            keys: Keys to check

        Returns:
            Number of keys that exist
        """
        self._ensure_initialized()
        prefixed = [self._prefixed_key(k) for k in keys]
        return self._client.exists(*prefixed)

    def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL on existing key.

        Args:
            key: Key to expire
            ttl: Time-to-live in seconds

        Returns:
            True if TTL was set
        """
        self._ensure_initialized()
        return bool(self._client.expire(self._prefixed_key(key), ttl))

    def ttl(self, key: str) -> int:
        """
        Get TTL of a key.

        Args:
            key: Key to check

        Returns:
            TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        self._ensure_initialized()
        return self._client.ttl(self._prefixed_key(key))

    # ==================== Sorted Sets (for buffers) ====================

    def zadd(
        self,
        key: str,
        mapping: Dict[bytes, float],
        nx: bool = False,
        xx: bool = False
    ) -> int:
        """
        Add members to sorted set.

        Args:
            key: Sorted set key
            mapping: {member: score} mapping
            nx: Only add new elements
            xx: Only update existing elements

        Returns:
            Number of elements added
        """
        self._ensure_initialized()
        return self._client.zadd(
            self._prefixed_key(key),
            {m: s for m, s in mapping.items()},
            nx=nx,
            xx=xx
        )

    def zrange(
        self,
        key: str,
        start: int,
        end: int,
        withscores: bool = False,
        desc: bool = False
    ) -> Union[List[bytes], List[Tuple[bytes, float]]]:
        """
        Get range from sorted set.

        Args:
            key: Sorted set key
            start: Start index (0-based)
            end: End index (-1 for all)
            withscores: Include scores in result
            desc: Reverse order (highest first)

        Returns:
            List of members (optionally with scores)
        """
        self._ensure_initialized()
        if desc:
            return self._client.zrevrange(
                self._prefixed_key(key),
                start, end,
                withscores=withscores
            )
        return self._client.zrange(
            self._prefixed_key(key),
            start, end,
            withscores=withscores
        )

    def zrangebyscore(
        self,
        key: str,
        min_score: float,
        max_score: float,
        start: int = None,
        num: int = None,
        withscores: bool = False
    ) -> Union[List[bytes], List[Tuple[bytes, float]]]:
        """
        Get members by score range.

        Args:
            key: Sorted set key
            min_score: Minimum score
            max_score: Maximum score
            start: Offset for pagination
            num: Limit for pagination
            withscores: Include scores

        Returns:
            List of members
        """
        self._ensure_initialized()
        return self._client.zrangebyscore(
            self._prefixed_key(key),
            min_score, max_score,
            start=start,
            num=num,
            withscores=withscores
        )

    def zrem(self, key: str, *members: bytes) -> int:
        """
        Remove members from sorted set.

        Args:
            key: Sorted set key
            members: Members to remove

        Returns:
            Number removed
        """
        self._ensure_initialized()
        return self._client.zrem(self._prefixed_key(key), *members)

    def zremrangebyrank(self, key: str, start: int, end: int) -> int:
        """
        Remove members by rank range (for buffer trimming).

        Args:
            key: Sorted set key
            start: Start rank
            end: End rank

        Returns:
            Number removed
        """
        self._ensure_initialized()
        return self._client.zremrangebyrank(self._prefixed_key(key), start, end)

    def zcard(self, key: str) -> int:
        """
        Get sorted set cardinality.

        Args:
            key: Sorted set key

        Returns:
            Number of members
        """
        self._ensure_initialized()
        return self._client.zcard(self._prefixed_key(key))

    # ==================== Distributed Locking ====================

    def acquire_lock(
        self,
        key: str,
        ttl: int = 5,
        blocking: bool = True,
        timeout: float = 10.0
    ) -> Optional[str]:
        """
        Acquire a distributed lock.

        Args:
            key: Lock key
            ttl: Lock TTL in seconds
            blocking: Wait for lock if not available
            timeout: Maximum wait time in seconds

        Returns:
            Lock token if acquired, None otherwise
        """
        self._ensure_initialized()
        lock_key = f"lock:{key}"
        token = str(uuid.uuid4())

        end_time = time.time() + timeout if blocking else time.time()

        while time.time() <= end_time:
            if self.set(lock_key, token.encode(), ttl=ttl, nx=True):
                return token
            if not blocking:
                return None
            time.sleep(0.1)

        return None

    def release_lock(self, key: str, token: str) -> bool:
        """
        Release a distributed lock.

        Args:
            key: Lock key
            token: Lock token from acquire_lock

        Returns:
            True if lock was released
        """
        self._ensure_initialized()
        lock_key = self._prefixed_key(f"lock:{key}")

        # Lua script for atomic check-and-delete
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        return bool(self._client.eval(script, 1, lock_key, token.encode()))

    @contextmanager
    def lock(self, key: str, ttl: int = 5, timeout: float = 10.0):
        """
        Context manager for distributed locking.

        Args:
            key: Lock key
            ttl: Lock TTL
            timeout: Maximum wait time

        Yields:
            Lock token

        Raises:
            LockAcquisitionError: If lock cannot be acquired
        """
        token = self.acquire_lock(key, ttl=ttl, timeout=timeout)
        if token is None:
            raise LockAcquisitionError(f"Failed to acquire lock: {key}")

        try:
            yield token
        finally:
            self.release_lock(key, token)

    # ==================== Counter Operations (for rate limiting) ====================

    def incr(self, key: str, amount: int = 1) -> int:
        """
        Increment a counter.

        Args:
            key: Counter key
            amount: Increment amount

        Returns:
            New counter value
        """
        self._ensure_initialized()
        return self._client.incrby(self._prefixed_key(key), amount)

    def get_counter(self, key: str) -> int:
        """
        Get counter value.

        Args:
            key: Counter key

        Returns:
            Counter value (0 if not exists)
        """
        self._ensure_initialized()
        value = self._client.get(self._prefixed_key(key))
        return int(value) if value else 0

    # ==================== Pipeline Operations ====================

    def pipeline(self, transaction: bool = True):
        """
        Get a pipeline for batch operations.

        Args:
            transaction: Use MULTI/EXEC

        Returns:
            Redis pipeline
        """
        self._ensure_initialized()
        return self._client.pipeline(transaction=transaction)

    def batch_get(self, keys: List[str]) -> Dict[str, Optional[bytes]]:
        """
        Get multiple keys in one round-trip.

        Args:
            keys: List of keys

        Returns:
            Dict of key -> value
        """
        self._ensure_initialized()
        prefixed = [self._prefixed_key(k) for k in keys]
        values = self._client.mget(prefixed)
        return {k: v for k, v in zip(keys, values)}

    def batch_set(
        self,
        mapping: Dict[str, bytes],
        ttl: int = None
    ) -> bool:
        """
        Set multiple keys in one operation.

        Args:
            mapping: Key -> value mapping
            ttl: Optional TTL for all keys

        Returns:
            True if successful
        """
        self._ensure_initialized()
        prefixed = {self._prefixed_key(k): v for k, v in mapping.items()}

        if ttl is None:
            return bool(self._client.mset(prefixed))

        # Use pipeline for TTL
        pipe = self._client.pipeline()
        for key, value in prefixed.items():
            pipe.setex(key, ttl, value)
        pipe.execute()
        return True

    # ==================== Health & Status ====================

    def health_check(self) -> bool:
        """
        Check Redis connectivity.

        Returns:
            True if healthy
        """
        try:
            self._ensure_initialized()
            return self._client.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server info.

        Returns:
            Server info dictionary
        """
        self._ensure_initialized()
        info = self._client.info()
        return {
            'redis_version': info.get('redis_version'),
            'connected_clients': info.get('connected_clients'),
            'used_memory_human': info.get('used_memory_human'),
            'total_connections_received': info.get('total_connections_received'),
            'keyspace_hits': info.get('keyspace_hits'),
            'keyspace_misses': info.get('keyspace_misses')
        }

    def close(self):
        """Close all connections."""
        if self._pool:
            self._pool.disconnect()
            self._pool = None
            self._client = None
            self._initialized = False
            logger.info("Redis client closed")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Global Redis instance (lazy initialization)
_redis_instance: Optional[RedisClient] = None


def get_redis() -> RedisClient:
    """
    Get the global Redis client instance.

    Returns:
        RedisClient singleton
    """
    global _redis_instance
    if _redis_instance is None:
        _redis_instance = RedisClient()
        _redis_instance.initialize()
    return _redis_instance


def close_redis():
    """Close the global Redis instance."""
    global _redis_instance
    if _redis_instance:
        _redis_instance.close()
        _redis_instance = None
