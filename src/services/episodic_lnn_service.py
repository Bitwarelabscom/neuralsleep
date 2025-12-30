"""
Flask API for Episodic Memory LNN

Endpoints:
- POST /episodic/store - Store new experiences
- POST /episodic/extract - Extract patterns
- POST /episodic/query - Query episodic memory
- GET /health - Health check

Now with PostgreSQL persistence and Redis caching.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.memory_networks import EpisodicMemoryLNN
from integration.memorycore_adapter import MemoryCoreAdapter
from utils.logger import setup_logger
from utils.config import Config

# Persistence layer
from persistence.database import DatabaseManager, get_database
from persistence.redis_client import RedisClient, get_redis
from persistence.repositories.episodic_repository import EpisodicRepository
from persistence.repositories.user_repository import UserRepository
from scaling.distributed_lock import DistributedLock, LockAcquisitionError
from scaling.rate_limiter import RateLimiter, RateLimitExceeded

app = Flask(__name__)
CORS(app)

logger = setup_logger('episodic_lnn_service')
config = Config()

# Initialize Episodic Memory LNN
logger.info(f"Initializing Episodic Memory LNN with hidden_size={config.episodic_hidden_size}")
episodic_lnn = EpisodicMemoryLNN(
    input_size=128,
    hidden_size=config.episodic_hidden_size,
    output_size=256
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
episodic_lnn = episodic_lnn.to(device)
logger.info(f"Model moved to device: {device}")

# Initialize persistence layer
db = None
redis = None
episodic_repo = None
lock_manager = None
rate_limiter = None
persistence_available = False

# In-memory fallback
_memory_fallback = {}  # userId -> list of episode dicts


def init_persistence():
    """Initialize persistence layer."""
    global db, redis, episodic_repo, lock_manager, rate_limiter, persistence_available

    try:
        db = get_database()
        redis = get_redis()

        if db.health_check() and redis.health_check():
            user_repo = UserRepository(db, redis)
            episodic_repo = EpisodicRepository(db, redis, user_repo)
            lock_manager = DistributedLock(redis)
            rate_limiter = RateLimiter(redis)
            persistence_available = True
            logger.info("Persistence layer initialized successfully")
        else:
            logger.warning("Persistence layer health check failed, using in-memory fallback")
            persistence_available = False
    except Exception as e:
        logger.warning(f"Could not initialize persistence layer: {e}. Using in-memory fallback.")
        persistence_available = False


# Load saved model if exists
model_path = '/opt/neuralsleep/models/episodic_lnn.pt'
if os.path.exists(model_path):
    try:
        episodic_lnn.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {e}")


def get_user_episodes(user_id: str):
    """Get user episodes from persistence or fallback."""
    if persistence_available and episodic_repo:
        try:
            experiences = episodic_repo.get_user_experiences(
                user_id, time_window='all', limit=1000, device=str(device)
            )
            return [
                {
                    'tensor': exp.tensor,
                    'timestamp': exp.event_timestamp.timestamp() if exp.event_timestamp else 0,
                    'importance': exp.importance
                }
                for exp in experiences
            ]
        except Exception as e:
            logger.warning(f"Persistence read failed for {user_id}: {e}")

    # Fallback
    return _memory_fallback.get(user_id, [])


def store_user_experiences(user_id: str, experiences: list) -> int:
    """Store experiences to persistence and fallback."""
    stored = 0

    # Prepare data
    exp_data = []
    for exp in experiences:
        exp_tensor = MemoryCoreAdapter.encode_learning_event(exp).to(device)
        exp_data.append({
            'tensor': exp_tensor,
            'importance': exp.get('importance', 0.5),
            'event_timestamp': datetime.fromtimestamp(exp.get('timestamp', datetime.now().timestamp())),
            'event_type': exp.get('eventType'),
            'character_id': exp.get('characterId'),
            'correct': exp.get('correct')
        })

    # Update fallback
    if user_id not in _memory_fallback:
        _memory_fallback[user_id] = []
    for ed in exp_data:
        _memory_fallback[user_id].append({
            'tensor': ed['tensor'],
            'timestamp': ed['event_timestamp'].timestamp(),
            'importance': ed['importance']
        })

    # Persist to database
    if persistence_available and episodic_repo:
        try:
            stored = episodic_repo.batch_store_experiences(user_id, exp_data)
        except Exception as e:
            logger.warning(f"Persistence write failed for {user_id}: {e}")
            stored = len(exp_data)
    else:
        stored = len(exp_data)

    return stored


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

    total_episodes = sum(len(eps) for eps in _memory_fallback.values())

    return jsonify({
        'status': 'healthy' if (persistence_available or total_episodes >= 0) else 'degraded',
        'service': 'episodic_lnn',
        'users_tracked': len(_memory_fallback),
        'total_episodes': total_episodes,
        'model_loaded': True,
        'dependencies': {
            'postgresql': 'healthy' if db_healthy else 'unavailable',
            'redis': 'healthy' if redis_healthy else 'unavailable'
        },
        'persistence_mode': 'database' if persistence_available else 'memory'
    })


@app.route('/episodic/store', methods=['POST'])
def store_experiences():
    """
    Store new learning experiences

    Request:
    {
        "userId": "user123",
        "experiences": [
            {
                "eventType": "practice",
                "characterId": "你",
                "correct": true,
                "timeSpent": 5000,
                "importance": 0.8,
                "timestamp": 1234567890.0
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        user_id = data['userId']
        experiences = data.get('experiences', [])

        # Rate limiting
        if rate_limiter:
            try:
                rate_limiter.require(user_id, '/episodic/store')
            except RateLimitExceeded as e:
                return jsonify({
                    'error': 'rate_limit_exceeded',
                    'message': str(e),
                    'reset_in': e.reset_in
                }), 429

        # Store experiences
        stored = store_user_experiences(user_id, experiences)

        logger.info(f"Stored {stored} experiences for user {user_id}")

        return jsonify({
            'status': 'success',
            'experiences_stored': stored,
            'total_episodes': len(_memory_fallback.get(user_id, [])),
            'persistence_mode': 'database' if persistence_available else 'memory'
        })

    except Exception as e:
        logger.error(f"Error storing experiences: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/episodic/extract', methods=['POST'])
def extract_patterns():
    """
    Extract patterns from episodic memory

    Request:
    {
        "userId": "user123",
        "timeWindow": "24h"  // or "all", "hour", "day", "week", "month"
    }

    Response:
    {
        "patterns": [ ... list of pattern tensors ... ],
        "count": 5
    }
    """
    try:
        data = request.json
        user_id = data['userId']
        time_window = data.get('timeWindow', 'all')

        # Get episodes
        if persistence_available and episodic_repo:
            experiences = episodic_repo.get_user_experiences(
                user_id, time_window=time_window, limit=1000, device=str(device)
            )
            episodes = [
                {
                    'tensor': exp.tensor,
                    'timestamp': exp.event_timestamp.timestamp() if exp.event_timestamp else 0,
                    'importance': exp.importance
                }
                for exp in experiences
            ]
        else:
            episodes = _memory_fallback.get(user_id, [])

        if len(episodes) < 2:
            return jsonify({'patterns': [], 'count': 0})

        # Stack tensors and timestamps
        experience_tensors = torch.stack([ep['tensor'] for ep in episodes])
        timestamps = torch.tensor([ep['timestamp'] for ep in episodes], dtype=torch.float32, device=device)

        # Normalize timestamps
        timestamps = timestamps - timestamps[0]

        # Extract patterns using episodic LNN
        episodic_lnn.eval()
        with torch.no_grad():
            patterns = episodic_lnn.extract_patterns(
                experience_tensors.unsqueeze(1),  # Add batch dimension
                timestamps
            )

        # Convert to list
        patterns_list = [patterns[0].tolist()]

        logger.info(f"Extracted {len(patterns_list)} patterns for user {user_id}")

        return jsonify({
            'patterns': patterns_list,
            'count': len(patterns_list)
        })

    except Exception as e:
        logger.error(f"Error extracting patterns: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/episodic/query', methods=['POST'])
def query_episodes():
    """
    Query episodic memory

    Request:
    {
        "userId": "user123",
        "filters": { ... }
    }
    """
    try:
        data = request.json
        user_id = data['userId']

        # Get stats from repository or fallback
        if persistence_available and episodic_repo:
            stats = episodic_repo.get_user_stats(user_id)
            return jsonify({
                'episodes': stats.get('total_experiences', 0),
                'count': stats.get('total_experiences', 0),
                'consolidated': stats.get('consolidated_count', 0),
                'pending': stats.get('pending_count', 0),
                'oldest': stats.get('earliest_event'),
                'newest': stats.get('latest_event'),
                'avg_importance': stats.get('avg_importance', 0)
            })
        else:
            episodes = _memory_fallback.get(user_id, [])
            return jsonify({
                'episodes': len(episodes),
                'count': len(episodes),
                'oldest': min(ep['timestamp'] for ep in episodes) if episodes else None,
                'newest': max(ep['timestamp'] for ep in episodes) if episodes else None
            })

    except Exception as e:
        logger.error(f"Error querying episodes: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/episodic/save', methods=['POST'])
def save_model():
    """Save current model weights"""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(episodic_lnn.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        return jsonify({'status': 'success', 'path': model_path})
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/episodic/stats', methods=['GET'])
def get_stats():
    """Get service statistics"""
    stats = {
        'total_users_in_memory': len(_memory_fallback),
        'total_episodes_in_memory': sum(len(eps) for eps in _memory_fallback.values()),
        'model_parameters': sum(p.numel() for p in episodic_lnn.parameters()),
        'persistence_available': persistence_available
    }

    # Add repository stats if available
    if persistence_available and episodic_repo:
        try:
            stats['total_experiences_in_db'] = episodic_repo.count_experiences()
        except Exception as e:
            logger.warning(f"Could not get repository stats: {e}")

    return jsonify(stats)


if __name__ == '__main__':
    port = config.episodic_lnn_port
    # Bind to 0.0.0.0 for Docker container access (port mapping restricts to localhost on host)
    host = '0.0.0.0'

    # Initialize persistence
    init_persistence()

    logger.info(f"Starting Episodic LNN Service on {host}:{port}")
    logger.info(f"Persistence mode: {'database' if persistence_available else 'memory'}")
    app.run(host=host, port=port, debug=(config.node_env == 'development'))
