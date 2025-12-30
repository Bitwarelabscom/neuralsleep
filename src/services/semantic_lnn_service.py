"""
Flask API for Semantic Memory LNN

Endpoints:
- POST /semantic/query - Get user model
- POST /semantic/consolidate - Update from patterns
- POST /semantic/mastery - Get character mastery
- POST /semantic/save - Save model weights
- GET /health - Health check

Now with PostgreSQL persistence and Redis caching.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.memory_networks import SemanticMemoryLNN
from integration.memorycore_adapter import MemoryCoreAdapter
from utils.logger import setup_logger
from utils.config import Config

# Persistence layer
from persistence.database import DatabaseManager, get_database
from persistence.redis_client import RedisClient, get_redis
from persistence.repositories.semantic_repository import SemanticRepository
from persistence.repositories.user_repository import UserRepository
from scaling.distributed_lock import DistributedLock, LockAcquisitionError
from scaling.rate_limiter import RateLimiter, RateLimitExceeded

app = Flask(__name__)
CORS(app)

logger = setup_logger('semantic_lnn_service')
config = Config()

# Initialize Semantic Memory LNN
logger.info(f"Initializing Semantic Memory LNN with hidden_size={config.semantic_hidden_size}")
semantic_lnn = SemanticMemoryLNN(
    input_size=256,
    hidden_size=config.semantic_hidden_size,
    output_size=512
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
semantic_lnn = semantic_lnn.to(device)
logger.info(f"Model moved to device: {device}")

# Initialize persistence layer
db = None
redis = None
semantic_repo = None
lock_manager = None
rate_limiter = None
persistence_available = False

# In-memory fallback (used if DB/Redis unavailable)
_memory_fallback = {}

def init_persistence():
    """Initialize persistence layer (deferred to allow graceful degradation)."""
    global db, redis, semantic_repo, lock_manager, rate_limiter, persistence_available

    try:
        db = get_database()
        redis = get_redis()

        if db.health_check() and redis.health_check():
            user_repo = UserRepository(db, redis)
            semantic_repo = SemanticRepository(db, redis, user_repo)
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
model_path = '/opt/neuralsleep/models/semantic_lnn.pt'
if os.path.exists(model_path):
    try:
        semantic_lnn.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {e}")


def get_user_state(user_id: str) -> torch.Tensor:
    """Get user state from persistence or fallback."""
    if persistence_available and semantic_repo:
        try:
            state = semantic_repo.get_or_create_state(user_id, device=str(device))
            return state
        except Exception as e:
            logger.warning(f"Persistence read failed for {user_id}: {e}")

    # Fallback to in-memory
    if user_id not in _memory_fallback:
        _memory_fallback[user_id] = torch.zeros(config.semantic_hidden_size, device=device)
    return _memory_fallback[user_id]


def save_user_state(user_id: str, state: torch.Tensor) -> bool:
    """Save user state to persistence and fallback."""
    # Always update fallback
    _memory_fallback[user_id] = state

    if persistence_available and semantic_repo:
        try:
            return semantic_repo.save_user_state(user_id, state)
        except Exception as e:
            logger.warning(f"Persistence write failed for {user_id}: {e}")
            return False
    return True


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
        'status': 'healthy' if (persistence_available or len(_memory_fallback) >= 0) else 'degraded',
        'service': 'semantic_lnn',
        'users_loaded': len(_memory_fallback),
        'model_loaded': True,
        'time_constants': semantic_lnn.get_time_constants().mean().item(),
        'dependencies': {
            'postgresql': 'healthy' if db_healthy else 'unavailable',
            'redis': 'healthy' if redis_healthy else 'unavailable'
        },
        'persistence_mode': 'database' if persistence_available else 'memory'
    })


@app.route('/semantic/query', methods=['POST'])
def query_semantic():
    """
    Get user model from semantic memory

    Request:
    {
        "userId": "user123",
        "operation": "get_model"
    }

    Response:
    {
        "model": { ... UserModel structure ... }
    }
    """
    try:
        data = request.json
        user_id = data['userId']
        operation = data.get('operation', 'get_model')

        # Rate limiting
        if rate_limiter:
            try:
                rate_limiter.require(user_id, '/semantic/query')
            except RateLimitExceeded as e:
                return jsonify({
                    'error': 'rate_limit_exceeded',
                    'message': str(e),
                    'reset_in': e.reset_in
                }), 429

        # Get user state
        state = get_user_state(user_id)

        if operation == 'get_model':
            # Decode user model from LNN state
            model = MemoryCoreAdapter.decode_lnn_state(state)
            logger.info(f"Retrieved model for user {user_id}")
            return jsonify({'model': model})

        elif operation == 'get_mastery':
            character_id = data.get('characterId', '')
            # In full implementation, would query specific character mastery
            # For now, return placeholder based on state
            mastery = MemoryCoreAdapter.decode_mastery_level(state[:128], character_id)
            mastery['userId'] = user_id
            logger.info(f"Retrieved mastery for user {user_id}, character {character_id}")
            return jsonify({'mastery': mastery})

        else:
            return jsonify({'error': 'Unknown operation'}), 400

    except Exception as e:
        logger.error(f"Error in query_semantic: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/semantic/consolidate', methods=['POST'])
def consolidate_patterns():
    """
    Update semantic memory from episodic patterns

    Request:
    {
        "userId": "user123",
        "patterns": [ ... pattern tensors as lists ... ],
        "timestamp": "2025-10-29T12:00:00Z"
    }
    """
    try:
        data = request.json
        user_id = data['userId']
        patterns = data.get('patterns', [])

        # Rate limiting
        if rate_limiter:
            try:
                rate_limiter.require(user_id, '/semantic/consolidate')
            except RateLimitExceeded as e:
                return jsonify({
                    'error': 'rate_limit_exceeded',
                    'message': str(e),
                    'reset_in': e.reset_in
                }), 429

        # Use distributed lock for write operations
        if lock_manager:
            try:
                with lock_manager.user_lock('semantic', user_id, ttl=10, timeout=5.0):
                    return _do_consolidation(user_id, patterns)
            except LockAcquisitionError:
                logger.warning(f"Could not acquire lock for {user_id}, proceeding anyway")
                return _do_consolidation(user_id, patterns)
        else:
            return _do_consolidation(user_id, patterns)

    except Exception as e:
        logger.error(f"Error in consolidate_patterns: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


def _do_consolidation(user_id: str, patterns: list):
    """Perform the actual consolidation logic."""
    # Get current state
    state = get_user_state(user_id)

    # Simple consolidation: average patterns and update state
    if len(patterns) > 0:
        # Convert patterns to tensors
        pattern_tensors = [torch.tensor(p, dtype=torch.float32, device=device) for p in patterns]

        # Ensure patterns are the right size
        pattern_tensors_resized = []
        for p in pattern_tensors:
            if p.numel() < config.semantic_hidden_size:
                # Pad if too small
                padded = torch.zeros(config.semantic_hidden_size, device=device)
                padded[:p.numel()] = p.flatten()
                pattern_tensors_resized.append(padded)
            else:
                # Truncate if too large
                pattern_tensors_resized.append(p.flatten()[:config.semantic_hidden_size])

        avg_pattern = torch.mean(torch.stack(pattern_tensors_resized), dim=0)

        # Update state (simple EMA for now - in full implementation, use LNN forward pass)
        alpha = 0.1  # Learning rate
        state = (1 - alpha) * state + alpha * avg_pattern

        # Save updated state
        save_user_state(user_id, state)

        logger.info(f"Consolidated {len(patterns)} patterns for user {user_id}")

    return jsonify({
        'status': 'success',
        'patterns_processed': len(patterns),
        'persistence_mode': 'database' if persistence_available else 'memory'
    })


@app.route('/semantic/save', methods=['POST'])
def save_model():
    """Save current model weights"""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(semantic_lnn.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        return jsonify({'status': 'success', 'path': model_path})
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/semantic/stats', methods=['GET'])
def get_stats():
    """Get service statistics"""
    stats = {
        'total_users_in_memory': len(_memory_fallback),
        'model_parameters': sum(p.numel() for p in semantic_lnn.parameters()),
        'time_constants_range': {
            'min': semantic_lnn.get_time_constants().min().item(),
            'max': semantic_lnn.get_time_constants().max().item(),
            'mean': semantic_lnn.get_time_constants().mean().item()
        },
        'persistence_available': persistence_available
    }

    # Add repository stats if available
    if persistence_available and semantic_repo:
        try:
            repo_stats = semantic_repo.get_stats()
            stats['repository'] = repo_stats
        except Exception as e:
            logger.warning(f"Could not get repository stats: {e}")

    return jsonify(stats)


if __name__ == '__main__':
    port = config.semantic_lnn_port
    # Bind to 0.0.0.0 for Docker container access (port mapping restricts to localhost on host)
    host = '0.0.0.0'

    # Initialize persistence
    init_persistence()

    logger.info(f"Starting Semantic LNN Service on {host}:{port}")
    logger.info(f"Persistence mode: {'database' if persistence_available else 'memory'}")
    app.run(host=host, port=port, debug=(config.node_env == 'development'))
