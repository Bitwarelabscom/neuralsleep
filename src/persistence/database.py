"""
PostgreSQL Database Manager for NeuralSleep

Provides connection pooling, transaction management, and retry logic
for reliable database operations.
"""

import os
import time
import logging
from typing import Optional, Any, Dict, List, Generator
from contextlib import contextmanager
from functools import wraps

import psycopg2
from psycopg2 import pool, sql, extras
from psycopg2.extensions import connection as PgConnection

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class ConnectionError(DatabaseError):
    """Failed to connect to database."""
    pass


class QueryError(DatabaseError):
    """Query execution failed."""
    pass


def retry_on_error(max_retries: int = 3, delay: float = 0.5):
    """
    Decorator for retrying database operations on transient errors.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Database operation failed (attempt {attempt + 1}), "
                            f"retrying in {sleep_time}s: {e}"
                        )
                        time.sleep(sleep_time)
            raise ConnectionError(f"Database operation failed after {max_retries} retries: {last_error}")
        return wrapper
    return decorator


class DatabaseManager:
    """
    PostgreSQL connection pool manager with transaction support.

    Usage:
        db = DatabaseManager(config)
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users")
                rows = cur.fetchall()
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None,
        min_connections: int = 5,
        max_connections: int = 20,
        connect_timeout: int = 5,
        query_timeout: int = 30
    ):
        """
        Initialize the database manager.

        Args:
            host: PostgreSQL host (default: from env)
            port: PostgreSQL port (default: from env)
            database: Database name (default: from env)
            user: Database user (default: from env)
            password: Database password (default: from env)
            min_connections: Minimum pool size
            max_connections: Maximum pool size
            connect_timeout: Connection timeout in seconds
            query_timeout: Query timeout in seconds
        """
        self.host = host or os.getenv('POSTGRES_HOST', 'localhost')
        self.port = port or int(os.getenv('POSTGRES_PORT', 5435))
        self.database = database or os.getenv('POSTGRES_DB', 'neuralsleep')
        self.user = user or os.getenv('POSTGRES_USER', 'neuralsleep_user')
        self.password = password or os.getenv('POSTGRES_PASSWORD', '')

        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connect_timeout = connect_timeout
        self.query_timeout = query_timeout

        self._pool: Optional[pool.ThreadedConnectionPool] = None
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
            dsn = self._build_dsn()
            self._pool = pool.ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                dsn=dsn
            )
            self._initialized = True
            logger.info(
                f"Database pool initialized: {self.host}:{self.port}/{self.database} "
                f"(min={self.min_connections}, max={self.max_connections})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            return False

    def _build_dsn(self) -> str:
        """Build PostgreSQL connection string."""
        return (
            f"host={self.host} "
            f"port={self.port} "
            f"dbname={self.database} "
            f"user={self.user} "
            f"password={self.password} "
            f"connect_timeout={self.connect_timeout} "
            f"options='-c statement_timeout={self.query_timeout * 1000}'"
        )

    @contextmanager
    def get_connection(self, autocommit: bool = False) -> Generator[PgConnection, None, None]:
        """
        Get a connection from the pool.

        Args:
            autocommit: Enable autocommit mode

        Yields:
            PostgreSQL connection

        Usage:
            with db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
        """
        if not self._initialized:
            self.initialize()

        if self._pool is None:
            raise ConnectionError("Database pool not initialized")

        conn = None
        try:
            conn = self._pool.getconn()
            conn.autocommit = autocommit
            yield conn
            if not autocommit:
                conn.commit()
        except Exception as e:
            if conn and not autocommit:
                conn.rollback()
            raise QueryError(f"Database operation failed: {e}") from e
        finally:
            if conn:
                self._pool.putconn(conn)

    @contextmanager
    def transaction(self) -> Generator[PgConnection, None, None]:
        """
        Context manager for explicit transactions.

        Usage:
            with db.transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO ...")
                    cur.execute("UPDATE ...")
                # Auto-commit on success, rollback on exception
        """
        with self.get_connection(autocommit=False) as conn:
            yield conn

    @retry_on_error(max_retries=3, delay=0.5)
    def execute(
        self,
        query: str,
        params: tuple = None,
        fetch: bool = False
    ) -> Optional[List[tuple]]:
        """
        Execute a single query.

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results

        Returns:
            Query results if fetch=True, None otherwise
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if fetch:
                    return cur.fetchall()
                return None

    @retry_on_error(max_retries=3, delay=0.5)
    def execute_many(
        self,
        query: str,
        params_list: List[tuple]
    ) -> int:
        """
        Execute a query with multiple parameter sets.

        Args:
            query: SQL query string
            params_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                extras.execute_batch(cur, query, params_list, page_size=100)
                return cur.rowcount

    @retry_on_error(max_retries=3, delay=0.5)
    def fetch_one(
        self,
        query: str,
        params: tuple = None
    ) -> Optional[tuple]:
        """
        Execute a query and fetch one row.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Single row or None
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()

    @retry_on_error(max_retries=3, delay=0.5)
    def fetch_all(
        self,
        query: str,
        params: tuple = None
    ) -> List[tuple]:
        """
        Execute a query and fetch all rows.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of rows
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    def fetch_dict(
        self,
        query: str,
        params: tuple = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a query and fetch results as dictionaries.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of row dictionaries
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=extras.RealDictCursor) as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]

    def health_check(self) -> bool:
        """
        Check database connectivity.

        Returns:
            True if healthy, False otherwise
        """
        try:
            result = self.fetch_one("SELECT 1")
            return result is not None and result[0] == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_pool_status(self) -> Dict[str, Any]:
        """
        Get connection pool status.

        Returns:
            Pool statistics dictionary
        """
        if not self._pool:
            return {'status': 'not_initialized'}

        return {
            'status': 'initialized',
            'min_connections': self.min_connections,
            'max_connections': self.max_connections,
            'host': self.host,
            'port': self.port,
            'database': self.database
        }

    def close(self):
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            self._initialized = False
            logger.info("Database pool closed")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Global database instance (lazy initialization)
_db_instance: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """
    Get the global database manager instance.

    Returns:
        DatabaseManager singleton
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
        _db_instance.initialize()
    return _db_instance


def close_database():
    """Close the global database instance."""
    global _db_instance
    if _db_instance:
        _db_instance.close()
        _db_instance = None
