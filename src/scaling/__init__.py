"""
NeuralSleep Scaling Module

Provides distributed locking, rate limiting, and user partitioning
for multi-user scaling.
"""

from .distributed_lock import DistributedLock, LockAcquisitionError
from .rate_limiter import RateLimiter, RateLimitExceeded

__all__ = [
    'DistributedLock',
    'LockAcquisitionError',
    'RateLimiter',
    'RateLimitExceeded'
]
