"""
Flask API for Working Memory LNN

Real-time processing with <50ms latency target.
Handles immediate context and recent experiences.

Now with Redis-first persistence for low latency.

Endpoints:
- POST /working/process - Process real-time input
- POST /working/buffer - Get/manage experience buffer
- POST /working/clear - Clear user buffer
- POST /working/save - Save model weights
- GET /health - Health check
- GET /working/stats - Performance statistics
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import sys
import time
from datetime import datetime
from collections import deque
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.memory_networks import WorkingMemoryLNN
from integration.memorycore_adapter import MemoryCoreAdapter
from utils.logger import setup_logger
from utils.config import Config

# Persistence layer
from persistence.database import DatabaseManager, get_database
from persistence.redis_client import RedisClient, get_redis
from persistence.repositories.working_repository import WorkingRepository
from persistence.repositories.user_repository import UserRepository
from scaling.rate_limiter import RateLimiter, RateLimitExceeded

app = Flask(__name__)
CORS(app)

logger = setup_logger('working_lnn_service')
config = Config()

# Initialize Working Memory LNN
logger.info(f"Initializing Working Memory LNN with hidden_size={config.working_hidden_size}")
working_lnn = WorkingMemoryLNN(
    input_size=512,
    hidden_size=config.working_hidden_size,
    output_size=128
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
working_lnn = working_lnn.to(device)
logger.info(f"Model moved to device: {device}")

# Initialize persistence layer
db = None
redis = None
working_repo = None
rate_limiter = None
persistence_available = False

# In-memory fallback (always maintained for latency)
BUFFER_MAX_SIZE = 100
user_buffers: Dict[str, deque] = {}
user_states: Dict[str, torch.Tensor] = {}

# Performance metrics
latency_samples: List[float] = []
MAX_LATENCY_SAMPLES = 1000


def init_persistence():
    """Initialize persistence layer."""
    global db, redis, working_repo, rate_limiter, persistence_available

    try:
        db = get_database()
        redis = get_redis()

        if redis.health_check():
            user_repo = UserRepository(db, redis)
            working_repo = WorkingRepository(db, redis, user_repo)
            rate_limiter = RateLimiter(redis)
            persistence_available = True
            logger.info("Persistence layer initialized (Redis-first mode)")
        else:
            logger.warning("Redis not available, using in-memory only")
            persistence_available = False
    except Exception as e:
        logger.warning(f"Could not initialize persistence: {e}. Using in-memory only.")
        persistence_available = False


# Load saved model if exists
model_path = '/opt/neuralsleep/models/working_lnn.pt'
if os.path.exists(model_path):
    try:
        working_lnn.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {e}")


def record_latency(latency_ms: float):
    """Record latency for performance tracking"""
    global latency_samples
    latency_samples.append(latency_ms)
    if len(latency_samples) > MAX_LATENCY_SAMPLES:
        latency_samples = latency_samples[-MAX_LATENCY_SAMPLES:]


def get_user_state(user_id: str) -> torch.Tensor:
    """Get user state (Redis first, then in-memory, then create new)."""
    # Always check in-memory first (fastest)
    if user_id in user_states:
        return user_states[user_id]

    # Try Redis if available
    if persistence_available and working_repo:
        try:
            state = working_repo.get_state(user_id, device=str(device))
            if state is not None:
                user_states[user_id] = state
                return state
        except Exception as e:
            logger.warning(f"Redis state read failed for {user_id}: {e}")

    # Create new state
    state = torch.zeros(config.working_hidden_size, device=device)
    user_states[user_id] = state
    return state


def save_user_state(user_id: str, state: torch.Tensor):
    """Save user state (in-memory + Redis async)."""
    # Always update in-memory
    user_states[user_id] = state

    # Async save to Redis if available
    if persistence_available and working_repo:
        try:
            working_repo.save_state(user_id, state)
        except Exception as e:
            logger.warning(f"Redis state save failed for {user_id}: {e}")


@app.before_request
def check_persistence():
    """Initialize persistence on first request if not already done."""
    global persistence_available
    if not persistence_available and redis is None:
        init_persistence()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint with latency metrics"""
    avg_latency = sum(latency_samples) / len(latency_samples) if latency_samples else 0
    p95_latency = sorted(latency_samples)[int(len(latency_samples) * 0.95)] if len(latency_samples) > 20 else avg_latency

    redis_healthy = redis.health_check() if redis else False

    return jsonify({
        'status': 'healthy',
        'service': 'working_lnn',
        'users_tracked': len(user_buffers),
        'total_experiences': sum(len(buf) for buf in user_buffers.values()),
        'model_loaded': True,
        'time_constants': working_lnn.get_time_constants().mean().item(),
        'dependencies': {
            'redis': 'healthy' if redis_healthy else 'unavailable'
        },
        'persistence_mode': 'redis' if persistence_available else 'memory',
        'performance': {
            'avg_latency_ms': round(avg_latency, 2),
            'p95_latency_ms': round(p95_latency, 2),
            'target_latency_ms': 50,
            'samples': len(latency_samples)
        }
    })


@app.route('/working/process', methods=['POST'])
def process_input():
    """
    Process real-time input through working memory

    Request:
    {
        "userId": "user123",
        "input": {
            "eventType": "interaction",
            "data": [...],  // or raw tensor values
            "timestamp": 1234567890.0
        }
    }

    Response:
    {
        "output": [...],
        "latency_ms": 12.5,
        "buffer_size": 15
    }
    """
    start_time = time.perf_counter()

    try:
        data = request.json
        user_id = data['userId']
        input_data = data.get('input', {})

        # Rate limiting (skip if not available to maintain latency)
        if rate_limiter:
            try:
                rate_limiter.require(user_id, '/working/process')
            except RateLimitExceeded as e:
                return jsonify({
                    'error': 'rate_limit_exceeded',
                    'message': str(e),
                    'reset_in': e.reset_in
                }), 429

        # Initialize user buffer if needed
        if user_id not in user_buffers:
            user_buffers[user_id] = deque(maxlen=BUFFER_MAX_SIZE)

        # Get current state
        current_state = get_user_state(user_id)

        # Encode input to tensor
        if 'tensor' in input_data:
            # Direct tensor input
            input_tensor = torch.tensor(input_data['tensor'], dtype=torch.float32, device=device)
        else:
            # Encode from learning event format
            input_tensor = MemoryCoreAdapter.encode_learning_event(input_data).to(device)

        # Ensure correct input size (512)
        if input_tensor.numel() < 512:
            padded = torch.zeros(512, device=device)
            padded[:input_tensor.numel()] = input_tensor.flatten()
            input_tensor = padded
        else:
            input_tensor = input_tensor.flatten()[:512]

        # Create time sequence (single step for real-time)
        t = torch.tensor([0.0, 0.1], device=device)  # 100ms step

        # Process through working memory LNN
        working_lnn.eval()
        with torch.no_grad():
            outputs, new_state = working_lnn(
                input_tensor.unsqueeze(0).unsqueeze(0),  # [1, 1, 512]
                t,
                initial_state=current_state.unsqueeze(0)
            )

        # Update user state
        save_user_state(user_id, new_state.squeeze(0))

        # Store experience in buffer
        timestamp = input_data.get('timestamp', datetime.now().timestamp())
        importance = input_data.get('importance', 0.5)

        output_tensor = outputs[-1].squeeze().cpu()
        user_buffers[user_id].append({
            'state': input_tensor.cpu(),
            'output': output_tensor,
            'timestamp': timestamp,
            'importance': importance
        })

        # Also persist to Redis buffer if available
        if persistence_available and working_repo:
            try:
                working_repo.add_to_buffer(
                    user_id,
                    input_tensor,
                    output_tensor.to(device),
                    importance,
                    timestamp
                )
            except Exception as e:
                logger.warning(f"Redis buffer write failed: {e}")

        # Calculate latency
        latency_ms = (time.perf_counter() - start_time) * 1000
        record_latency(latency_ms)

        output_list = output_tensor.tolist()

        logger.debug(f"Processed input for user {user_id} in {latency_ms:.2f}ms")

        return jsonify({
            'output': output_list,
            'latency_ms': round(latency_ms, 2),
            'buffer_size': len(user_buffers[user_id]),
            'meets_target': latency_ms < 50,
            'persistence_mode': 'redis' if persistence_available else 'memory'
        })

    except Exception as e:
        logger.error(f"Error processing input: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/working/buffer', methods=['POST'])
def get_buffer():
    """
    Get user's working memory buffer

    Request:
    {
        "userId": "user123",
        "limit": 10  // optional, default all
    }

    Response:
    {
        "experiences": [...],
        "count": 10,
        "importance_weights": [...]
    }
    """
    try:
        data = request.json
        user_id = data['userId']
        limit = data.get('limit', None)

        if user_id not in user_buffers:
            return jsonify({
                'experiences': [],
                'count': 0,
                'importance_weights': []
            })

        buffer = list(user_buffers[user_id])
        if limit:
            buffer = buffer[-limit:]

        # Extract importance weights for consolidation
        importance_weights = [exp['importance'] for exp in buffer]

        return jsonify({
            'count': len(buffer),
            'oldest': buffer[0]['timestamp'] if buffer else None,
            'newest': buffer[-1]['timestamp'] if buffer else None,
            'importance_weights': importance_weights,
            'experiences': [
                {
                    'timestamp': exp['timestamp'],
                    'importance': exp['importance'],
                    'state': exp['state'].tolist(),
                    'output': exp['output'].tolist()
                }
                for exp in buffer
            ]
        })

    except Exception as e:
        logger.error(f"Error getting buffer: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/working/clear', methods=['POST'])
def clear_buffer():
    """
    Clear user's working memory buffer (after consolidation)

    Request:
    {
        "userId": "user123",
        "keep_recent": 10  // optional, keep last N experiences
    }
    """
    try:
        data = request.json
        user_id = data['userId']
        keep_recent = data.get('keep_recent', 0)

        if user_id not in user_buffers:
            return jsonify({'status': 'success', 'cleared': 0})

        old_count = len(user_buffers[user_id])

        if keep_recent > 0:
            # Keep only recent experiences
            recent = list(user_buffers[user_id])[-keep_recent:]
            user_buffers[user_id] = deque(recent, maxlen=BUFFER_MAX_SIZE)
        else:
            user_buffers[user_id].clear()

        # Also clear Redis buffer if available
        if persistence_available and working_repo:
            try:
                working_repo.clear_buffer(user_id)
            except Exception as e:
                logger.warning(f"Redis buffer clear failed: {e}")

        cleared = old_count - len(user_buffers[user_id])
        logger.info(f"Cleared {cleared} experiences from user {user_id} buffer")

        return jsonify({
            'status': 'success',
            'cleared': cleared,
            'remaining': len(user_buffers[user_id])
        })

    except Exception as e:
        logger.error(f"Error clearing buffer: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/working/save', methods=['POST'])
def save_model():
    """Save current model weights"""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(working_lnn.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        return jsonify({'status': 'success', 'path': model_path})
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/working/stats', methods=['GET'])
def get_stats():
    """Get service statistics"""
    avg_latency = sum(latency_samples) / len(latency_samples) if latency_samples else 0

    stats = {
        'total_users': len(user_buffers),
        'total_experiences': sum(len(buf) for buf in user_buffers.values()),
        'model_parameters': sum(p.numel() for p in working_lnn.parameters()),
        'time_constants_range': {
            'min': working_lnn.get_time_constants().min().item(),
            'max': working_lnn.get_time_constants().max().item(),
            'mean': working_lnn.get_time_constants().mean().item()
        },
        'performance': {
            'avg_latency_ms': round(avg_latency, 2),
            'samples': len(latency_samples),
            'target_met_ratio': sum(1 for l in latency_samples if l < 50) / len(latency_samples) if latency_samples else 1.0
        },
        'buffer_config': {
            'max_size': BUFFER_MAX_SIZE
        },
        'persistence_available': persistence_available
    }

    # Add Redis stats if available
    if persistence_available and working_repo:
        try:
            stats['redis_info'] = redis.get_info()
        except Exception:
            pass

    return jsonify(stats)


if __name__ == '__main__':
    port = config.working_lnn_port
    # Bind to 0.0.0.0 for Docker container access (port mapping restricts to localhost on host)
    host = '0.0.0.0'

    # Initialize persistence
    init_persistence()

    logger.info(f"Starting Working Memory LNN Service on {host}:{port}")
    logger.info(f"Persistence mode: {'redis' if persistence_available else 'memory'}")
    app.run(host=host, port=port, debug=(config.node_env == 'development'))
