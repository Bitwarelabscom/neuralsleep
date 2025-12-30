"""
Rate Limiter for NeuralSleep

Redis-based sliding window rate limiting for API protection.
"""

import time
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

from persistence.redis_client import RedisClient, get_redis

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Rate limit exceeded exception."""
    def __init__(self, endpoint: str, limit: int, reset_in: int):
        self.endpoint = endpoint
        self.limit = limit
        self.reset_in = reset_in
        super().__init__(f"Rate limit exceeded for {endpoint}: {limit}/min, reset in {reset_in}s")


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int
    burst_size: int = None  # Allow burst up to this (default: 2x rate)

    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = self.requests_per_minute * 2


# Default rate limits by endpoint pattern
DEFAULT_LIMITS = {
    '/working/process': RateLimitConfig(200),      # High-frequency real-time
    '/semantic/consolidate': RateLimitConfig(10),  # Expensive operation
    '/episodic/store': RateLimitConfig(100),       # Batch stores
    '/consciousness/compute': RateLimitConfig(5),  # Very expensive
    'default': RateLimitConfig(60)                 # Default for other endpoints
}


class RateLimiter:
    """
    Sliding window rate limiter using Redis.

    Implements sliding window log algorithm for accurate rate limiting.
    Supports per-user, per-endpoint limits with configurable rates.
    """

    # Window size in seconds
    WINDOW_SIZE = 60

    def __init__(
        self,
        redis: RedisClient = None,
        limits: Dict[str, RateLimitConfig] = None
    ):
        """
        Initialize rate limiter.

        Args:
            redis: Redis client (uses global if not provided)
            limits: Custom rate limits by endpoint
        """
        self._redis = redis
        self._limits = limits or DEFAULT_LIMITS

    @property
    def redis(self) -> RedisClient:
        """Get Redis client (lazy initialization)."""
        if self._redis is None:
            self._redis = get_redis()
        return self._redis

    def _get_limit(self, endpoint: str) -> RateLimitConfig:
        """Get rate limit config for endpoint."""
        # Check for exact match
        if endpoint in self._limits:
            return self._limits[endpoint]

        # Check for prefix match
        for pattern, config in self._limits.items():
            if pattern != 'default' and endpoint.startswith(pattern):
                return config

        # Return default
        return self._limits.get('default', RateLimitConfig(60))

    def _key(self, user_id: str, endpoint: str) -> str:
        """Generate Redis key for rate limit."""
        # Normalize endpoint (remove trailing slashes)
        endpoint = endpoint.rstrip('/')
        return f"ratelimit:{user_id}:{endpoint}"

    def check(
        self,
        user_id: str,
        endpoint: str,
        increment: bool = True
    ) -> Tuple[bool, int, int]:
        """
        Check if request is allowed and optionally increment counter.

        Uses sliding window algorithm:
        1. Remove expired entries (older than window)
        2. Count entries in current window
        3. If under limit, add new entry

        Args:
            user_id: User identifier
            endpoint: API endpoint
            increment: Whether to increment counter if allowed

        Returns:
            Tuple of (allowed, remaining, reset_in_seconds)
        """
        config = self._get_limit(endpoint)
        key = self._key(user_id, endpoint)
        now = time.time()
        window_start = now - self.WINDOW_SIZE

        try:
            # Use pipeline for atomic operations
            pipe = self.redis.pipeline()

            # Remove old entries
            pipe.zremrangebyscore(key, '-inf', window_start)

            # Count current entries
            pipe.zcard(key)

            # Execute
            results = pipe.execute()
            current_count = results[1]

            # Check limit
            allowed = current_count < config.requests_per_minute
            remaining = max(0, config.requests_per_minute - current_count - 1)

            # Calculate reset time
            if current_count > 0:
                # Get oldest entry timestamp
                oldest = self.redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_time = oldest[0][1]
                    reset_in = max(0, int(oldest_time + self.WINDOW_SIZE - now))
                else:
                    reset_in = self.WINDOW_SIZE
            else:
                reset_in = self.WINDOW_SIZE

            if allowed and increment:
                # Add new entry
                self.redis.zadd(key, {str(now).encode(): now})
                # Set TTL on key
                self.redis.expire(key, self.WINDOW_SIZE + 10)

            return allowed, remaining, reset_in

        except Exception as e:
            logger.error(f"Rate limit check failed for {user_id}/{endpoint}: {e}")
            # Fail open - allow request if Redis is down
            return True, config.requests_per_minute, self.WINDOW_SIZE

    def is_allowed(self, user_id: str, endpoint: str) -> bool:
        """
        Simple check if request is allowed.

        Args:
            user_id: User identifier
            endpoint: API endpoint

        Returns:
            True if allowed
        """
        allowed, _, _ = self.check(user_id, endpoint)
        return allowed

    def get_limit_info(
        self,
        user_id: str,
        endpoint: str
    ) -> Dict[str, int]:
        """
        Get rate limit info without incrementing.

        Args:
            user_id: User identifier
            endpoint: API endpoint

        Returns:
            Dict with limit, remaining, reset_in
        """
        config = self._get_limit(endpoint)
        allowed, remaining, reset_in = self.check(user_id, endpoint, increment=False)

        return {
            'limit': config.requests_per_minute,
            'remaining': remaining,
            'reset_in': reset_in,
            'burst_size': config.burst_size
        }

    def require(self, user_id: str, endpoint: str):
        """
        Check rate limit and raise exception if exceeded.

        Args:
            user_id: User identifier
            endpoint: API endpoint

        Raises:
            RateLimitExceeded: If limit is exceeded
        """
        config = self._get_limit(endpoint)
        allowed, remaining, reset_in = self.check(user_id, endpoint)

        if not allowed:
            raise RateLimitExceeded(
                endpoint=endpoint,
                limit=config.requests_per_minute,
                reset_in=reset_in
            )

    def reset(self, user_id: str, endpoint: str = None) -> bool:
        """
        Reset rate limit for user.

        Args:
            user_id: User identifier
            endpoint: Specific endpoint (or all if None)

        Returns:
            True if successful
        """
        try:
            if endpoint:
                self.redis.delete(self._key(user_id, endpoint))
            else:
                # Delete all rate limit keys for user
                pattern = f"ratelimit:{user_id}:*"
                # Note: SCAN would be better for production
                # For now, we'll reset known endpoints
                for ep in self._limits.keys():
                    if ep != 'default':
                        self.redis.delete(self._key(user_id, ep))
            return True
        except Exception as e:
            logger.error(f"Failed to reset rate limit for {user_id}: {e}")
            return False

    def set_limit(self, endpoint: str, config: RateLimitConfig):
        """
        Set custom rate limit for endpoint.

        Args:
            endpoint: API endpoint pattern
            config: Rate limit configuration
        """
        self._limits[endpoint] = config

    def get_user_usage(self, user_id: str) -> Dict[str, Dict[str, int]]:
        """
        Get current usage for all endpoints.

        Args:
            user_id: User identifier

        Returns:
            Dict of endpoint -> usage info
        """
        usage = {}
        for endpoint in self._limits.keys():
            if endpoint != 'default':
                info = self.get_limit_info(user_id, endpoint)
                usage[endpoint] = info
        return usage


# Flask middleware helper
def rate_limit_middleware(limiter: RateLimiter):
    """
    Flask middleware for rate limiting.

    Usage:
        from flask import Flask, request, jsonify

        app = Flask(__name__)
        limiter = RateLimiter()

        @app.before_request
        def check_rate_limit():
            return rate_limit_middleware(limiter)()
    """
    from flask import request, jsonify

    def check():
        # Get user ID from request
        user_id = None

        # Try to get from JSON body
        if request.is_json:
            data = request.get_json(silent=True)
            if data:
                user_id = data.get('userId')

        # Fallback to query param or header
        if not user_id:
            user_id = request.args.get('userId') or request.headers.get('X-User-ID')

        # Skip if no user ID (let endpoint handle auth)
        if not user_id:
            return None

        endpoint = request.path
        try:
            limiter.require(user_id, endpoint)
            return None
        except RateLimitExceeded as e:
            return jsonify({
                'error': 'rate_limit_exceeded',
                'message': str(e),
                'limit': e.limit,
                'reset_in': e.reset_in
            }), 429

    return check


# Global rate limiter instance
_limiter_instance: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _limiter_instance
    if _limiter_instance is None:
        _limiter_instance = RateLimiter()
    return _limiter_instance
