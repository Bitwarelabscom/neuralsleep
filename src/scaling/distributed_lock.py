"""
Distributed Locking for NeuralSleep

Redis-based distributed locks to prevent race conditions
in multi-instance deployments.
"""

import time
import uuid
import logging
from typing import Optional, Callable, Any
from contextlib import contextmanager
from functools import wraps

from persistence.redis_client import RedisClient, get_redis

logger = logging.getLogger(__name__)


class LockAcquisitionError(Exception):
    """Failed to acquire distributed lock."""
    pass


class DistributedLock:
    """
    Redis-based distributed locking with automatic renewal and deadlock prevention.

    Usage:
        lock = DistributedLock(redis_client)

        # Context manager
        with lock.user_lock('semantic', 'user123'):
            # Critical section

        # Decorator
        @lock.with_lock('semantic', lambda args: args[0])
        def update_user(user_id, data):
            pass
    """

    # Default lock TTL in seconds
    DEFAULT_TTL = 5

    # Lock acquisition timeout
    DEFAULT_TIMEOUT = 10.0

    # Retry interval for lock acquisition
    RETRY_INTERVAL = 0.1

    def __init__(self, redis_client: RedisClient = None):
        """
        Initialize distributed lock manager.

        Args:
            redis_client: Redis client instance (uses global if not provided)
        """
        self._redis = redis_client

    @property
    def redis(self) -> RedisClient:
        """Get Redis client (lazy initialization)."""
        if self._redis is None:
            self._redis = get_redis()
        return self._redis

    def _lock_key(self, service: str, user_id: str) -> str:
        """Generate lock key."""
        return f"lock:{service}:{user_id}"

    def acquire(
        self,
        service: str,
        user_id: str,
        ttl: int = DEFAULT_TTL,
        blocking: bool = True,
        timeout: float = DEFAULT_TIMEOUT
    ) -> Optional[str]:
        """
        Acquire a distributed lock.

        Args:
            service: Service name (semantic, episodic, working, consciousness)
            user_id: User identifier
            ttl: Lock TTL in seconds
            blocking: Wait for lock if not available
            timeout: Maximum wait time in seconds

        Returns:
            Lock token if acquired, None otherwise
        """
        lock_key = self._lock_key(service, user_id)
        token = str(uuid.uuid4())
        end_time = time.time() + timeout if blocking else time.time()

        while time.time() <= end_time:
            # Try to acquire lock with NX (only if not exists)
            if self.redis.set(lock_key, token.encode(), ttl=ttl, nx=True):
                logger.debug(f"Lock acquired: {lock_key} (token={token[:8]}...)")
                return token

            if not blocking:
                return None

            # Wait before retry
            time.sleep(self.RETRY_INTERVAL)

        logger.warning(f"Lock acquisition timeout: {lock_key}")
        return None

    def release(self, service: str, user_id: str, token: str) -> bool:
        """
        Release a distributed lock.

        Args:
            service: Service name
            user_id: User identifier
            token: Lock token from acquire()

        Returns:
            True if lock was released, False if token mismatch
        """
        lock_key = self._lock_key(service, user_id)

        # Atomic check-and-delete using Lua script
        # Only delete if the token matches (we own the lock)
        current = self.redis.get(lock_key)
        if current is None:
            logger.debug(f"Lock already released: {lock_key}")
            return True

        if current == token.encode():
            self.redis.delete(lock_key)
            logger.debug(f"Lock released: {lock_key}")
            return True

        logger.warning(f"Lock token mismatch: {lock_key} (expected={token[:8]}...)")
        return False

    def extend(
        self,
        service: str,
        user_id: str,
        token: str,
        ttl: int = DEFAULT_TTL
    ) -> bool:
        """
        Extend lock TTL (refresh the lock).

        Args:
            service: Service name
            user_id: User identifier
            token: Lock token from acquire()
            ttl: New TTL in seconds

        Returns:
            True if extended, False if token mismatch
        """
        lock_key = self._lock_key(service, user_id)

        # Verify ownership before extending
        current = self.redis.get(lock_key)
        if current != token.encode():
            return False

        return self.redis.expire(lock_key, ttl)

    def is_locked(self, service: str, user_id: str) -> bool:
        """
        Check if a lock is held.

        Args:
            service: Service name
            user_id: User identifier

        Returns:
            True if locked
        """
        lock_key = self._lock_key(service, user_id)
        return self.redis.exists(lock_key) > 0

    @contextmanager
    def user_lock(
        self,
        service: str,
        user_id: str,
        ttl: int = DEFAULT_TTL,
        timeout: float = DEFAULT_TIMEOUT,
        auto_extend: bool = False
    ):
        """
        Context manager for distributed locking.

        Args:
            service: Service name
            user_id: User identifier
            ttl: Lock TTL
            timeout: Acquisition timeout
            auto_extend: Not implemented (for future use)

        Yields:
            Lock token

        Raises:
            LockAcquisitionError: If lock cannot be acquired

        Usage:
            with lock.user_lock('semantic', 'user123'):
                # Critical section - lock auto-released on exit
        """
        token = self.acquire(service, user_id, ttl=ttl, timeout=timeout)
        if token is None:
            raise LockAcquisitionError(
                f"Failed to acquire lock for {service}:{user_id} "
                f"after {timeout}s timeout"
            )

        try:
            yield token
        finally:
            self.release(service, user_id, token)

    def with_lock(
        self,
        service: str,
        user_id_extractor: Callable[..., str],
        ttl: int = DEFAULT_TTL,
        timeout: float = DEFAULT_TIMEOUT
    ):
        """
        Decorator for functions that need locking.

        Args:
            service: Service name
            user_id_extractor: Function to extract user_id from args
            ttl: Lock TTL
            timeout: Acquisition timeout

        Returns:
            Decorated function

        Usage:
            @lock.with_lock('semantic', lambda user_id, **kw: user_id)
            def update_user_state(user_id, state):
                pass
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                user_id = user_id_extractor(*args, **kwargs)
                with self.user_lock(service, user_id, ttl=ttl, timeout=timeout):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def multi_lock(
        self,
        locks: list,
        ttl: int = DEFAULT_TTL,
        timeout: float = DEFAULT_TIMEOUT
    ):
        """
        Acquire multiple locks atomically (all or none).

        Args:
            locks: List of (service, user_id) tuples
            ttl: Lock TTL
            timeout: Total timeout for all locks

        Yields:
            List of lock tokens

        Raises:
            LockAcquisitionError: If any lock cannot be acquired
        """
        # Sort locks to prevent deadlock (always acquire in same order)
        sorted_locks = sorted(locks)
        acquired = []

        try:
            end_time = time.time() + timeout
            for service, user_id in sorted_locks:
                remaining = max(0, end_time - time.time())
                token = self.acquire(
                    service, user_id,
                    ttl=ttl,
                    timeout=remaining,
                    blocking=True
                )
                if token is None:
                    raise LockAcquisitionError(
                        f"Failed to acquire lock for {service}:{user_id}"
                    )
                acquired.append((service, user_id, token))

            yield [t for _, _, t in acquired]

        finally:
            # Release in reverse order
            for service, user_id, token in reversed(acquired):
                self.release(service, user_id, token)


class LockGuard:
    """
    RAII-style lock guard for manual lock management.

    Usage:
        guard = LockGuard(lock_manager, 'semantic', 'user123')
        try:
            guard.acquire()
            # Do work
        finally:
            guard.release()
    """

    def __init__(
        self,
        lock_manager: DistributedLock,
        service: str,
        user_id: str,
        ttl: int = DistributedLock.DEFAULT_TTL,
        timeout: float = DistributedLock.DEFAULT_TIMEOUT
    ):
        self.lock_manager = lock_manager
        self.service = service
        self.user_id = user_id
        self.ttl = ttl
        self.timeout = timeout
        self.token: Optional[str] = None

    def acquire(self) -> bool:
        """Acquire the lock."""
        self.token = self.lock_manager.acquire(
            self.service, self.user_id,
            ttl=self.ttl, timeout=self.timeout
        )
        return self.token is not None

    def release(self) -> bool:
        """Release the lock."""
        if self.token:
            result = self.lock_manager.release(
                self.service, self.user_id, self.token
            )
            self.token = None
            return result
        return True

    def is_held(self) -> bool:
        """Check if lock is held."""
        return self.token is not None

    def extend(self) -> bool:
        """Extend lock TTL."""
        if self.token:
            return self.lock_manager.extend(
                self.service, self.user_id,
                self.token, self.ttl
            )
        return False


# Global lock manager (lazy initialization)
_lock_instance: Optional[DistributedLock] = None


def get_lock_manager() -> DistributedLock:
    """
    Get the global distributed lock manager.

    Returns:
        DistributedLock singleton
    """
    global _lock_instance
    if _lock_instance is None:
        _lock_instance = DistributedLock()
    return _lock_instance
