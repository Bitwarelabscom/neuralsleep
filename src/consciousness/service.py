"""
Flask API for Consciousness Metrics Service

Provides real-time consciousness monitoring and metrics.

Endpoints:
- GET /health - Health check
- GET /consciousness/metrics - Current consciousness metrics
- GET /consciousness/report - Full consciousness report
- POST /consciousness/compute - Trigger Φ computation for specific state
- GET /consciousness/history - Metrics history
- POST /consciousness/reset - Reset metrics history

Now with PostgreSQL persistence and Redis caching.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consciousness.metrics import ConsciousnessMetrics
from consciousness.phi_computation import PhiComputation, compute_phi_from_tensor
from consciousness.self_reference import SelfReferentialProcessor
from utils.logger import setup_logger
from utils.config import Config

# Persistence layer
from persistence.database import DatabaseManager, get_database
from persistence.redis_client import RedisClient, get_redis
from persistence.repositories.consciousness_repository import ConsciousnessRepository
from scaling.rate_limiter import RateLimiter, RateLimitExceeded

app = Flask(__name__)
CORS(app)

logger = setup_logger('consciousness_service')
config = Config()

# Initialize consciousness components
logger.info("Initializing Consciousness Metrics Service")

phi_computer = PhiComputation()
consciousness_metrics = ConsciousnessMetrics(phi_threshold=0.5, history_size=100)
self_ref_processor = SelfReferentialProcessor(
    observation_size=512,
    self_model_size=256,
    output_size=128
)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
self_ref_processor.to(device)
logger.info(f"Consciousness service on device: {device}")

# Initialize persistence layer
db = None
redis = None
consciousness_repo = None
rate_limiter = None
persistence_available = False

# In-memory fallback (used if DB/Redis unavailable)
current_states = {
    'working': None,
    'episodic': None,
    'semantic': None
}


def init_persistence():
    """Initialize persistence layer."""
    global db, redis, consciousness_repo, rate_limiter, persistence_available

    try:
        db = get_database()
        redis = get_redis()

        if db.health_check() and redis.health_check():
            consciousness_repo = ConsciousnessRepository(db, redis)
            rate_limiter = RateLimiter(redis)
            persistence_available = True
            logger.info("Persistence layer initialized successfully")

            # Try to restore current states from Redis
            _restore_current_states()
        else:
            logger.warning("Persistence layer health check failed, using in-memory fallback")
            persistence_available = False
    except Exception as e:
        logger.warning(f"Could not initialize persistence layer: {e}. Using in-memory fallback.")
        persistence_available = False


def _restore_current_states():
    """Restore current states from persistence on startup."""
    global current_states
    if persistence_available and consciousness_repo:
        try:
            states = consciousness_repo.get_all_current_states()
            for key, value in states.items():
                if value is not None:
                    current_states[key] = value.tolist()
            logger.info(f"Restored {sum(1 for v in states.values() if v is not None)} states from persistence")
        except Exception as e:
            logger.warning(f"Could not restore states: {e}")


def get_current_state(state_type: str) -> np.ndarray:
    """Get current state from persistence or fallback."""
    if persistence_available and consciousness_repo:
        try:
            state = consciousness_repo.get_current_state(state_type)
            if state is not None:
                return state
        except Exception as e:
            logger.warning(f"Persistence read failed for {state_type}: {e}")

    # Fallback to in-memory
    if current_states.get(state_type) is not None:
        return np.array(current_states[state_type], dtype=np.float32)
    return None


def set_current_state(state_type: str, state: np.ndarray) -> bool:
    """Set current state to persistence and fallback."""
    # Always update fallback
    if state is not None:
        current_states[state_type] = state.tolist() if isinstance(state, np.ndarray) else state
    else:
        current_states[state_type] = None

    # Update persistence
    if persistence_available and consciousness_repo and state is not None:
        try:
            return consciousness_repo.set_current_state(state_type, state)
        except Exception as e:
            logger.warning(f"Persistence write failed for {state_type}: {e}")
            return False
    return True


def save_metrics_to_persistence(metrics: dict, computation_time_ms: int = None) -> bool:
    """Save metrics to persistent storage."""
    if persistence_available and consciousness_repo:
        try:
            record_id = consciousness_repo.save_metrics(metrics, computation_time_ms)
            return record_id is not None
        except Exception as e:
            logger.warning(f"Failed to save metrics to persistence: {e}")
    return False


@app.before_request
def check_persistence():
    """Initialize persistence on first request if not already done."""
    global persistence_available
    if not persistence_available and db is None:
        init_persistence()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with dependency status"""
    db_healthy = db.health_check() if db else False
    redis_healthy = redis.health_check() if redis else False

    return jsonify({
        'status': 'healthy' if (persistence_available or True) else 'degraded',
        'service': 'consciousness_metrics',
        'phi_threshold': consciousness_metrics.phi_threshold,
        'history_length': len(consciousness_metrics.metrics_history),
        'self_reference_depth': self_ref_processor.get_self_reference_depth(),
        'phi_computation_enabled': config.enable_phi_computation,
        'dependencies': {
            'postgresql': 'healthy' if db_healthy else 'unavailable',
            'redis': 'healthy' if redis_healthy else 'unavailable'
        },
        'persistence_mode': 'database' if persistence_available else 'memory',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/consciousness/metrics', methods=['GET'])
def get_metrics():
    """
    Get current consciousness metrics

    Response:
    {
        "integrated_information": 0.45,
        "self_reference_depth": 3,
        "temporal_integration": 0.62,
        "causal_density": 0.38,
        "dynamical_complexity": 0.55,
        "information_flow": {...},
        "timestamp": "..."
    }
    """
    try:
        start_time = time.perf_counter()

        # Get states from persistence or fallback
        working_arr = get_current_state('working')
        episodic_arr = get_current_state('episodic')
        semantic_arr = get_current_state('semantic')

        # Convert to tensors if available
        working = torch.tensor(working_arr, device=device) if working_arr is not None else None
        episodic = torch.tensor(episodic_arr, device=device) if episodic_arr is not None else None
        semantic = torch.tensor(semantic_arr, device=device) if semantic_arr is not None else None

        # Get self-model state
        self_state = self_ref_processor.self_representation

        metrics = consciousness_metrics.compute_all_metrics(
            working_state=working,
            episodic_state=episodic,
            semantic_state=semantic,
            self_model_state=self_state
        )

        # Calculate computation time
        computation_time_ms = int((time.perf_counter() - start_time) * 1000)

        # Save to persistence
        save_metrics_to_persistence(metrics, computation_time_ms)

        metrics['persistence_mode'] = 'database' if persistence_available else 'memory'
        return jsonify(metrics)

    except Exception as e:
        logger.error(f"Error computing metrics: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/consciousness/report', methods=['GET'])
def get_report():
    """
    Get full consciousness report with interpretation

    Response:
    {
        "metrics": {...},
        "interpretation": "...",
        "consciousness_level": "moderate",
        "analysis": {...},
        "recommendations": [...],
        "above_threshold": true
    }
    """
    try:
        start_time = time.perf_counter()

        # Get states from persistence or fallback
        working_arr = get_current_state('working')
        episodic_arr = get_current_state('episodic')
        semantic_arr = get_current_state('semantic')

        working = torch.tensor(working_arr, device=device) if working_arr is not None else None
        episodic = torch.tensor(episodic_arr, device=device) if episodic_arr is not None else None
        semantic = torch.tensor(semantic_arr, device=device) if semantic_arr is not None else None
        self_state = self_ref_processor.self_representation

        report = consciousness_metrics.consciousness_report(
            working_state=working,
            episodic_state=episodic,
            semantic_state=semantic,
            self_model_state=self_state
        )

        # Add self-reference info
        report['self_reference'] = self_ref_processor.get_self_state()

        # Add persistence info
        report['persistence_mode'] = 'database' if persistence_available else 'memory'

        # Calculate and save
        computation_time_ms = int((time.perf_counter() - start_time) * 1000)
        if 'metrics' in report:
            save_metrics_to_persistence(report['metrics'], computation_time_ms)

        return jsonify(report)

    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/consciousness/compute', methods=['POST'])
def compute_phi():
    """
    Compute Φ for a specific state

    Request:
    {
        "state": [...],  // State vector or matrix
        "method": "approximation"  // or "exact" for small states
    }

    Response:
    {
        "phi": 0.45,
        "method_used": "approximation",
        "state_dimensions": 256
    }
    """
    try:
        data = request.json
        state = np.array(data['state'])
        method = data.get('method', 'approximation')

        start_time = time.perf_counter()
        phi = phi_computer.compute_phi(state, method=method)
        computation_time_ms = int((time.perf_counter() - start_time) * 1000)

        result = {
            'phi': phi,
            'method_used': method,
            'state_dimensions': state.shape[-1] if state.ndim > 0 else 0,
            'computation_time_ms': computation_time_ms,
            'timestamp': datetime.now().isoformat()
        }

        # Save state history for Phi analysis
        if persistence_available and consciousness_repo:
            try:
                consciousness_repo.save_state_history(state)
            except Exception as e:
                logger.warning(f"Could not save state history: {e}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error computing phi: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/consciousness/update_state', methods=['POST'])
def update_state():
    """
    Update memory states for consciousness monitoring

    Request:
    {
        "working": [...],   // optional
        "episodic": [...],  // optional
        "semantic": [...]   // optional
    }
    """
    try:
        data = request.json

        # Update states in persistence and fallback
        if 'working' in data:
            state = np.array(data['working'], dtype=np.float32)
            set_current_state('working', state)
        if 'episodic' in data:
            state = np.array(data['episodic'], dtype=np.float32)
            set_current_state('episodic', state)
        if 'semantic' in data:
            state = np.array(data['semantic'], dtype=np.float32)
            set_current_state('semantic', state)

        # Update self-referential processor
        working_arr = get_current_state('working')
        episodic_arr = get_current_state('episodic')
        semantic_arr = get_current_state('semantic')

        working = torch.tensor(working_arr, device=device) if working_arr is not None else None
        episodic = torch.tensor(episodic_arr, device=device) if episodic_arr is not None else None
        semantic = torch.tensor(semantic_arr, device=device) if semantic_arr is not None else None

        observation = self_ref_processor.observe_processing(
            working_memory_state=working,
            episodic_state=episodic,
            semantic_state=semantic
        )

        self_rep, metacog = self_ref_processor.update_self_model(
            observation,
            timestamp=datetime.now().timestamp()
        )

        return jsonify({
            'status': 'updated',
            'states_received': list(data.keys()),
            'metacognition': metacog,
            'self_reference_depth': self_ref_processor.get_self_reference_depth(),
            'persistence_mode': 'database' if persistence_available else 'memory'
        })

    except Exception as e:
        logger.error(f"Error updating state: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/consciousness/history', methods=['GET'])
def get_history():
    """
    Get consciousness metrics history

    Query params:
    - limit: Number of recent entries (default 20)
    - metric: Specific metric to return (default all)
    - hours: Only return records from last N hours
    """
    try:
        limit = int(request.args.get('limit', 20))
        metric = request.args.get('metric', None)
        hours = request.args.get('hours', None)

        # Try persistence first
        if persistence_available and consciousness_repo:
            try:
                history_records = consciousness_repo.get_metrics_history(
                    limit=limit,
                    since_hours=int(hours) if hours else None
                )

                history = [r.to_dict() for r in history_records]

                if metric:
                    # Return specific metric trend
                    trend = [m.get(metric, None) for m in history]
                    return jsonify({
                        'metric': metric,
                        'trend': trend,
                        'count': len(trend),
                        'persistence_mode': 'database'
                    })

                return jsonify({
                    'history': history,
                    'count': len(history),
                    'persistence_mode': 'database'
                })

            except Exception as e:
                logger.warning(f"Persistence history read failed: {e}, falling back to in-memory")

        # Fallback to in-memory history
        history = consciousness_metrics.metrics_history[-limit:]

        if metric:
            # Return specific metric trend
            trend = [m.get(metric, None) for m in history]
            return jsonify({
                'metric': metric,
                'trend': trend,
                'count': len(trend),
                'persistence_mode': 'memory'
            })

        return jsonify({
            'history': history,
            'count': len(history),
            'total_available': len(consciousness_metrics.metrics_history),
            'persistence_mode': 'memory'
        })

    except Exception as e:
        logger.error(f"Error getting history: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/consciousness/self_model', methods=['GET'])
def get_self_model():
    """
    Get current self-model state and information

    Response:
    {
        "self_representation": [...],
        "self_state_norm": 1.23,
        "recursion_depth": 3,
        "history_length": 45,
        "time_constants": 50.5
    }
    """
    try:
        state = self_ref_processor.get_self_state()
        state['persistence_mode'] = 'database' if persistence_available else 'memory'
        return jsonify(state)

    except Exception as e:
        logger.error(f"Error getting self-model: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/consciousness/reset', methods=['POST'])
def reset():
    """Reset consciousness metrics and self-model"""
    try:
        consciousness_metrics.reset_history()
        self_ref_processor.reset()

        # Clear in-memory states
        current_states['working'] = None
        current_states['episodic'] = None
        current_states['semantic'] = None

        # Clear persistence states (Redis keys will expire)
        # DB history is preserved for research

        logger.info("Consciousness service reset")

        return jsonify({
            'status': 'reset',
            'persistence_mode': 'database' if persistence_available else 'memory',
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error resetting: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/consciousness/stats', methods=['GET'])
def get_stats():
    """Get service statistics"""
    avg_phi = 0
    total_computations = len(consciousness_metrics.metrics_history)
    above_threshold_count = 0

    if consciousness_metrics.metrics_history:
        phi_values = [m.get('integrated_information', 0) for m in consciousness_metrics.metrics_history]
        avg_phi = sum(phi_values) / len(phi_values)
        above_threshold_count = sum(
            1 for m in consciousness_metrics.metrics_history
            if m.get('integrated_information', 0) > consciousness_metrics.phi_threshold
        )

    stats = {
        'total_computations_memory': total_computations,
        'average_phi_memory': round(avg_phi, 4),
        'phi_threshold': consciousness_metrics.phi_threshold,
        'above_threshold_count_memory': above_threshold_count,
        'self_model_parameters': sum(p.numel() for p in self_ref_processor.self_model.parameters()),
        'self_reference_depth': self_ref_processor.get_self_reference_depth(),
        'device': str(device),
        'persistence_available': persistence_available
    }

    # Add persistence stats if available
    if persistence_available and consciousness_repo:
        try:
            phi_stats = consciousness_repo.get_phi_statistics()
            level_dist = consciousness_repo.get_consciousness_level_distribution(hours=24)
            stats['database_stats'] = {
                'phi_statistics': phi_stats,
                'consciousness_level_distribution': level_dist
            }
        except Exception as e:
            logger.warning(f"Could not get persistence stats: {e}")

    return jsonify(stats)


if __name__ == '__main__':
    port = 5003  # Consciousness service port
    # Bind to 0.0.0.0 for Docker container access (port mapping restricts to localhost on host)
    host = '0.0.0.0'

    # Initialize persistence
    init_persistence()

    logger.info(f"Starting Consciousness Metrics Service on {host}:{port}")
    logger.info(f"Persistence mode: {'database' if persistence_available else 'memory'}")
    app.run(host=host, port=port, debug=(config.node_env == 'development'))
