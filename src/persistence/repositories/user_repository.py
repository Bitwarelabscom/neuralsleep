"""
User Repository for NeuralSleep

Manages user registration, lookup, and metadata.
"""

import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, asdict

from .base import BaseRepository
from persistence.database import DatabaseManager
from persistence.redis_client import RedisClient

logger = logging.getLogger(__name__)


@dataclass
class User:
    """User entity."""
    id: str
    external_id: str
    created_at: datetime
    updated_at: datetime
    last_active_at: Optional[datetime] = None
    settings: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'external_id': self.external_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_active_at': self.last_active_at.isoformat() if self.last_active_at else None,
            'settings': self.settings or {},
            'metadata': self.metadata or {}
        }


class UserRepository(BaseRepository[User]):
    """
    Repository for user management.

    Handles user registration, lookup by external ID,
    and user metadata storage.
    """

    _cache_ttl = 3600  # 1 hour

    def __init__(
        self,
        db: DatabaseManager = None,
        redis: RedisClient = None
    ):
        super().__init__(db, redis)
        self._id_mapping_cache: Dict[str, str] = {}  # external_id -> id

    def _cache_key(self, identifier: str) -> str:
        """Generate cache key (by internal UUID)."""
        return f"user:{identifier}"

    def _external_id_cache_key(self, external_id: str) -> str:
        """Generate cache key for external ID mapping."""
        return f"user:ext:{external_id}"

    def _serialize(self, value: User) -> bytes:
        """Serialize user to bytes."""
        return json.dumps(value.to_dict()).encode('utf-8')

    def _deserialize(self, data: bytes) -> User:
        """Deserialize user from bytes."""
        d = json.loads(data.decode('utf-8'))
        return User(
            id=d['id'],
            external_id=d['external_id'],
            created_at=datetime.fromisoformat(d['created_at']) if d.get('created_at') else None,
            updated_at=datetime.fromisoformat(d['updated_at']) if d.get('updated_at') else None,
            last_active_at=datetime.fromisoformat(d['last_active_at']) if d.get('last_active_at') else None,
            settings=d.get('settings', {}),
            metadata=d.get('metadata', {})
        )

    def _get_from_db(self, identifier: str) -> Optional[User]:
        """Get user by internal ID from database."""
        try:
            result = self.db.fetch_dict(
                """
                SELECT id, external_id, created_at, updated_at,
                       last_active_at, settings, metadata
                FROM users WHERE id = %s
                """,
                (identifier,)
            )
            if result:
                row = result[0]
                return User(
                    id=str(row['id']),
                    external_id=row['external_id'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    last_active_at=row.get('last_active_at'),
                    settings=row.get('settings', {}),
                    metadata=row.get('metadata', {})
                )
        except Exception as e:
            logger.error(f"Failed to get user {identifier}: {e}")
        return None

    def _save_to_db(self, identifier: str, value: User) -> bool:
        """Save user to database (upsert)."""
        try:
            self.db.execute(
                """
                INSERT INTO users (id, external_id, settings, metadata, last_active_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (id) DO UPDATE SET
                    settings = EXCLUDED.settings,
                    metadata = EXCLUDED.metadata,
                    last_active_at = NOW()
                """,
                (
                    value.id,
                    value.external_id,
                    json.dumps(value.settings or {}),
                    json.dumps(value.metadata or {})
                )
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save user {identifier}: {e}")
            return False

    def _delete_from_db(self, identifier: str) -> bool:
        """Delete user from database."""
        try:
            self.db.execute(
                "DELETE FROM users WHERE id = %s",
                (identifier,)
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete user {identifier}: {e}")
            return False

    # ==================== User-Specific Methods ====================

    def get_by_external_id(self, external_id: str) -> Optional[User]:
        """
        Get user by external (MemoryCore) ID.

        Args:
            external_id: External user identifier

        Returns:
            User or None
        """
        # Check cache for ID mapping
        cache_key = self._external_id_cache_key(external_id)
        try:
            user_id_bytes = self.redis.get(cache_key)
            if user_id_bytes:
                user_id = user_id_bytes.decode('utf-8')
                return self.get(user_id)
        except Exception:
            pass

        # Also check local mapping cache
        if external_id in self._id_mapping_cache:
            return self.get(self._id_mapping_cache[external_id])

        # Query database
        try:
            result = self.db.fetch_dict(
                """
                SELECT id, external_id, created_at, updated_at,
                       last_active_at, settings, metadata
                FROM users WHERE external_id = %s
                """,
                (external_id,)
            )
            if result:
                row = result[0]
                user = User(
                    id=str(row['id']),
                    external_id=row['external_id'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    last_active_at=row.get('last_active_at'),
                    settings=row.get('settings', {}),
                    metadata=row.get('metadata', {})
                )
                # Cache the mapping and user
                self._cache_id_mapping(external_id, user.id)
                self._set_in_cache(user.id, user)
                return user
        except Exception as e:
            logger.error(f"Failed to get user by external_id {external_id}: {e}")

        return None

    def get_or_create(self, external_id: str) -> User:
        """
        Get existing user or create new one.

        Args:
            external_id: External user identifier

        Returns:
            User (existing or newly created)
        """
        # Try to get existing
        user = self.get_by_external_id(external_id)
        if user:
            return user

        # Create new user using database function
        try:
            result = self.db.fetch_one(
                "SELECT get_or_create_user(%s)",
                (external_id,)
            )
            if result:
                user_id = str(result[0])
                # Fetch the created user
                user = self._get_from_db(user_id)
                if user:
                    self._cache_id_mapping(external_id, user.id)
                    self._set_in_cache(user.id, user)
                    return user
        except Exception as e:
            logger.error(f"Failed to get_or_create user {external_id}: {e}")

        # Fallback: create manually
        import uuid
        user_id = str(uuid.uuid4())
        user = User(
            id=user_id,
            external_id=external_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        try:
            self.db.execute(
                """
                INSERT INTO users (id, external_id)
                VALUES (%s, %s)
                ON CONFLICT (external_id) DO UPDATE SET
                    last_active_at = NOW()
                RETURNING id
                """,
                (user_id, external_id)
            )
            self._cache_id_mapping(external_id, user_id)
            self._set_in_cache(user_id, user)
        except Exception as e:
            logger.error(f"Failed to create user {external_id}: {e}")

        return user

    def _cache_id_mapping(self, external_id: str, user_id: str):
        """Cache the external_id -> user_id mapping."""
        try:
            cache_key = self._external_id_cache_key(external_id)
            self.redis.set(cache_key, user_id.encode('utf-8'), ttl=self._cache_ttl)
            self._id_mapping_cache[external_id] = user_id
        except Exception as e:
            logger.warning(f"Failed to cache ID mapping: {e}")
            self._id_mapping_cache[external_id] = user_id

    def update_last_active(self, user_id: str) -> bool:
        """
        Update user's last active timestamp.

        Args:
            user_id: Internal user ID

        Returns:
            True if successful
        """
        try:
            self.db.execute(
                "UPDATE users SET last_active_at = NOW() WHERE id = %s",
                (user_id,)
            )
            # Invalidate cache to refresh
            self._delete_from_cache(user_id)
            return True
        except Exception as e:
            logger.error(f"Failed to update last_active for {user_id}: {e}")
            return False

    def get_all_user_ids(self, limit: int = 1000) -> List[str]:
        """
        Get all user external IDs.

        Args:
            limit: Maximum number of users

        Returns:
            List of external IDs
        """
        try:
            result = self.db.fetch_all(
                "SELECT external_id FROM users ORDER BY last_active_at DESC NULLS LAST LIMIT %s",
                (limit,)
            )
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"Failed to get all user IDs: {e}")
            return []

    def get_active_users(
        self,
        since_hours: int = 24,
        limit: int = 100
    ) -> List[User]:
        """
        Get recently active users.

        Args:
            since_hours: Hours since last activity
            limit: Maximum number of users

        Returns:
            List of active users
        """
        try:
            result = self.db.fetch_dict(
                """
                SELECT id, external_id, created_at, updated_at,
                       last_active_at, settings, metadata
                FROM users
                WHERE last_active_at > NOW() - INTERVAL '%s hours'
                ORDER BY last_active_at DESC
                LIMIT %s
                """,
                (since_hours, limit)
            )
            return [
                User(
                    id=str(row['id']),
                    external_id=row['external_id'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    last_active_at=row.get('last_active_at'),
                    settings=row.get('settings', {}),
                    metadata=row.get('metadata', {})
                )
                for row in result
            ]
        except Exception as e:
            logger.error(f"Failed to get active users: {e}")
            return []

    def count_users(self) -> int:
        """
        Get total user count.

        Returns:
            Number of users
        """
        try:
            result = self.db.fetch_one("SELECT COUNT(*) FROM users")
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to count users: {e}")
            return 0
