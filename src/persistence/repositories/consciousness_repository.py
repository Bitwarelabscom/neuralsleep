"""
Consciousness Repository for NeuralSleep

Manages persistent storage of consciousness metrics, state history,
and self-reference data.
"""

import json
import hashlib
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass

import numpy as np
import torch

from persistence.database import DatabaseManager, get_database
from persistence.redis_client import RedisClient, get_redis
from persistence.tensor_serialization import tensor_to_bytes, bytes_to_tensor

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessMetricsRecord:
    """Consciousness metrics record."""
    id: str
    integrated_information: float
    self_reference_depth: int
    temporal_integration: float
    causal_density: float
    dynamical_complexity: float
    information_flow: Dict[str, float]
    consciousness_level: str
    computed_at: datetime
    computation_time_ms: int = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'integrated_information': self.integrated_information,
            'self_reference_depth': self.self_reference_depth,
            'temporal_integration': self.temporal_integration,
            'causal_density': self.causal_density,
            'dynamical_complexity': self.dynamical_complexity,
            'information_flow': self.information_flow,
            'consciousness_level': self.consciousness_level,
            'computed_at': self.computed_at.isoformat() if self.computed_at else None,
            'computation_time_ms': self.computation_time_ms,
            'metadata': self.metadata or {}
        }


class ConsciousnessRepository:
    """
    Repository for consciousness metrics and state history.

    Stores:
    - Consciousness metrics snapshots
    - Combined state history for Phi computation
    - Current memory states for real-time tracking
    """

    _metrics_cache_ttl = 300  # 5 minutes
    _state_cache_ttl = 60  # 1 minute for current states
    _max_history_size = 100

    def __init__(
        self,
        db: DatabaseManager = None,
        redis: RedisClient = None
    ):
        self._db = db or get_database()
        self._redis = redis or get_redis()

    @property
    def db(self) -> DatabaseManager:
        return self._db

    @property
    def redis(self) -> RedisClient:
        return self._redis

    # ==================== Current States ====================

    def set_current_state(
        self,
        state_type: str,
        state: np.ndarray
    ) -> bool:
        """
        Set current memory state for consciousness monitoring.

        Args:
            state_type: 'working', 'episodic', or 'semantic'
            state: State array

        Returns:
            True if successful
        """
        try:
            cache_key = f"consciousness:{state_type}"
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            data = state.tobytes()
            return self.redis.set(cache_key, data, ttl=self._state_cache_ttl)
        except Exception as e:
            logger.warning(f"Failed to set {state_type} state: {e}")
            return False

    def get_current_state(self, state_type: str) -> Optional[np.ndarray]:
        """
        Get current memory state.

        Args:
            state_type: 'working', 'episodic', or 'semantic'

        Returns:
            State array or None
        """
        try:
            cache_key = f"consciousness:{state_type}"
            data = self.redis.get(cache_key)
            if data:
                return np.frombuffer(data, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get {state_type} state: {e}")
        return None

    def get_all_current_states(self) -> Dict[str, Optional[np.ndarray]]:
        """Get all current memory states."""
        return {
            'working': self.get_current_state('working'),
            'episodic': self.get_current_state('episodic'),
            'semantic': self.get_current_state('semantic')
        }

    # ==================== Metrics Storage ====================

    def save_metrics(
        self,
        metrics: Dict[str, Any],
        computation_time_ms: int = None
    ) -> Optional[str]:
        """
        Save consciousness metrics snapshot.

        Args:
            metrics: Metrics dictionary with:
                - integrated_information (phi)
                - self_reference_depth
                - temporal_integration
                - causal_density
                - dynamical_complexity
                - information_flow
                - consciousness_level
            computation_time_ms: Time to compute

        Returns:
            Record ID or None
        """
        try:
            # Compute state hashes for correlation
            states = self.get_all_current_states()
            hashes = {
                'working': self._hash_state(states.get('working')),
                'episodic': self._hash_state(states.get('episodic')),
                'semantic': self._hash_state(states.get('semantic'))
            }

            result = self.db.fetch_one(
                """
                INSERT INTO consciousness_metrics
                (integrated_information, self_reference_depth, temporal_integration,
                 causal_density, dynamical_complexity, information_flow,
                 consciousness_level, working_state_hash, episodic_state_hash,
                 semantic_state_hash, computation_time_ms, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    metrics.get('integrated_information', 0),
                    metrics.get('self_reference_depth', 0),
                    metrics.get('temporal_integration', 0),
                    metrics.get('causal_density', 0),
                    metrics.get('dynamical_complexity', 0),
                    json.dumps(metrics.get('information_flow', {})),
                    metrics.get('consciousness_level', 'unknown'),
                    hashes['working'],
                    hashes['episodic'],
                    hashes['semantic'],
                    computation_time_ms,
                    json.dumps(metrics.get('metadata', {}))
                )
            )

            if result:
                record_id = str(result[0])
                # Cache latest metrics
                self._cache_latest_metrics(metrics)
                return record_id

        except Exception as e:
            logger.error(f"Failed to save consciousness metrics: {e}")

        return None

    def _hash_state(self, state: Optional[np.ndarray]) -> Optional[str]:
        """Compute hash of state for correlation."""
        if state is None:
            return None
        return hashlib.sha256(state.tobytes()).hexdigest()[:16]

    def _cache_latest_metrics(self, metrics: Dict[str, Any]):
        """Cache latest metrics in Redis."""
        try:
            cache_key = "consciousness:latest"
            data = json.dumps(metrics).encode('utf-8')
            self.redis.set(cache_key, data, ttl=self._metrics_cache_ttl)
        except Exception as e:
            logger.warning(f"Failed to cache metrics: {e}")

    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get most recent metrics (from cache or DB).

        Returns:
            Latest metrics dictionary
        """
        # Try cache first
        try:
            cache_key = "consciousness:latest"
            data = self.redis.get(cache_key)
            if data:
                return json.loads(data.decode('utf-8'))
        except Exception:
            pass

        # Fallback to DB
        try:
            result = self.db.fetch_dict(
                """
                SELECT integrated_information, self_reference_depth,
                       temporal_integration, causal_density, dynamical_complexity,
                       information_flow, consciousness_level, computed_at,
                       computation_time_ms
                FROM consciousness_metrics
                ORDER BY computed_at DESC
                LIMIT 1
                """
            )

            if result:
                row = result[0]
                metrics = {
                    'integrated_information': row['integrated_information'],
                    'self_reference_depth': row['self_reference_depth'],
                    'temporal_integration': row['temporal_integration'],
                    'causal_density': row['causal_density'],
                    'dynamical_complexity': row['dynamical_complexity'],
                    'information_flow': row['information_flow'] or {},
                    'consciousness_level': row['consciousness_level'],
                    'computed_at': row['computed_at'].isoformat() if row['computed_at'] else None,
                    'computation_time_ms': row['computation_time_ms']
                }
                self._cache_latest_metrics(metrics)
                return metrics

        except Exception as e:
            logger.error(f"Failed to get latest metrics: {e}")

        return None

    def get_metrics_history(
        self,
        limit: int = 100,
        since_hours: int = None
    ) -> List[ConsciousnessMetricsRecord]:
        """
        Get historical metrics.

        Args:
            limit: Maximum records
            since_hours: Only return records from last N hours

        Returns:
            List of metrics records
        """
        try:
            time_filter = ""
            params = [limit]

            if since_hours:
                time_filter = "WHERE computed_at > NOW() - INTERVAL '%s hours'"
                params = [since_hours, limit]

            query = f"""
                SELECT id, integrated_information, self_reference_depth,
                       temporal_integration, causal_density, dynamical_complexity,
                       information_flow, consciousness_level, computed_at,
                       computation_time_ms, metadata
                FROM consciousness_metrics
                {time_filter}
                ORDER BY computed_at DESC
                LIMIT %s
            """

            result = self.db.fetch_dict(query, tuple(params))

            return [
                ConsciousnessMetricsRecord(
                    id=str(row['id']),
                    integrated_information=row['integrated_information'],
                    self_reference_depth=row['self_reference_depth'],
                    temporal_integration=row['temporal_integration'],
                    causal_density=row['causal_density'],
                    dynamical_complexity=row['dynamical_complexity'],
                    information_flow=row['information_flow'] or {},
                    consciousness_level=row['consciousness_level'],
                    computed_at=row['computed_at'],
                    computation_time_ms=row['computation_time_ms'],
                    metadata=row['metadata'] or {}
                )
                for row in result
            ]

        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []

    # ==================== State History ====================

    def save_state_history(
        self,
        combined_state: np.ndarray,
        working_contribution: float = None,
        episodic_contribution: float = None,
        semantic_contribution: float = None
    ) -> bool:
        """
        Save combined state snapshot for Phi computation.

        Args:
            combined_state: Combined state array
            working_contribution: Working memory contribution ratio
            episodic_contribution: Episodic memory contribution ratio
            semantic_contribution: Semantic memory contribution ratio

        Returns:
            True if successful
        """
        try:
            state_bytes = combined_state.tobytes()

            self.db.execute(
                """
                INSERT INTO consciousness_state_history
                (combined_state, state_dimensions, working_contribution,
                 episodic_contribution, semantic_contribution)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    state_bytes,
                    len(combined_state),
                    working_contribution,
                    episodic_contribution,
                    semantic_contribution
                )
            )

            # Cleanup old history
            self._cleanup_old_history()

            return True

        except Exception as e:
            logger.error(f"Failed to save state history: {e}")
            return False

    def get_state_history(
        self,
        limit: int = 100
    ) -> List[np.ndarray]:
        """
        Get recent state history for Phi computation.

        Args:
            limit: Maximum states to return

        Returns:
            List of state arrays (newest first)
        """
        try:
            result = self.db.fetch_all(
                """
                SELECT combined_state, state_dimensions
                FROM consciousness_state_history
                ORDER BY recorded_at DESC
                LIMIT %s
                """,
                (limit,)
            )

            return [
                np.frombuffer(bytes(row[0]), dtype=np.float32)
                for row in result
            ]

        except Exception as e:
            logger.error(f"Failed to get state history: {e}")
            return []

    def _cleanup_old_history(self):
        """Remove old state history entries."""
        try:
            # Keep only last N entries
            self.db.execute(
                """
                DELETE FROM consciousness_state_history
                WHERE id NOT IN (
                    SELECT id FROM consciousness_state_history
                    ORDER BY recorded_at DESC
                    LIMIT %s
                )
                """,
                (self._max_history_size,)
            )
        except Exception as e:
            logger.warning(f"Failed to cleanup old history: {e}")

    # ==================== Statistics ====================

    def get_phi_statistics(self) -> Dict[str, Any]:
        """
        Get Phi statistics over time.

        Returns:
            Statistics dictionary
        """
        try:
            result = self.db.fetch_dict(
                """
                SELECT
                    COUNT(*) as total_measurements,
                    AVG(integrated_information) as avg_phi,
                    MAX(integrated_information) as max_phi,
                    MIN(integrated_information) as min_phi,
                    STDDEV(integrated_information) as stddev_phi,
                    AVG(computation_time_ms) as avg_computation_time,
                    MAX(computed_at) as last_computed
                FROM consciousness_metrics
                WHERE computed_at > NOW() - INTERVAL '24 hours'
                """
            )

            if result:
                row = result[0]
                return {
                    'total_measurements_24h': row['total_measurements'] or 0,
                    'avg_phi': float(row['avg_phi'] or 0),
                    'max_phi': float(row['max_phi'] or 0),
                    'min_phi': float(row['min_phi'] or 0),
                    'stddev_phi': float(row['stddev_phi'] or 0),
                    'avg_computation_time_ms': float(row['avg_computation_time'] or 0),
                    'last_computed': row['last_computed'].isoformat() if row['last_computed'] else None
                }

        except Exception as e:
            logger.error(f"Failed to get phi statistics: {e}")

        return {}

    def get_consciousness_level_distribution(
        self,
        hours: int = 24
    ) -> Dict[str, int]:
        """
        Get distribution of consciousness levels.

        Args:
            hours: Time window

        Returns:
            Dict of level -> count
        """
        try:
            result = self.db.fetch_all(
                """
                SELECT consciousness_level, COUNT(*) as count
                FROM consciousness_metrics
                WHERE computed_at > NOW() - INTERVAL '%s hours'
                GROUP BY consciousness_level
                ORDER BY count DESC
                """,
                (hours,)
            )

            return {row[0]: row[1] for row in result}

        except Exception as e:
            logger.error(f"Failed to get level distribution: {e}")
            return {}
