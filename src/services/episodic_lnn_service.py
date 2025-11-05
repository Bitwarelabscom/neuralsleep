"""
Flask API for Episodic Memory LNN

Endpoints:
- POST /episodic/store - Store new experiences
- POST /episodic/extract - Extract patterns
- POST /episodic/query - Query episodic memory
- GET /health - Health check
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

# User episodic buffers
user_episodes = {}  # userId -> list of episode tensors

# Load saved model if exists
model_path = '/opt/neuralsleep/models/episodic_lnn.pt'
if os.path.exists(model_path):
    try:
        episodic_lnn.load_state_dict(torch.load(model_path))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {e}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'episodic_lnn',
        'users_tracked': len(user_episodes),
        'total_episodes': sum(len(eps) for eps in user_episodes.values()),
        'model_loaded': True
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

        if user_id not in user_episodes:
            user_episodes[user_id] = []

        # Convert experiences to tensors
        for exp in experiences:
            exp_tensor = MemoryCoreAdapter.encode_learning_event(exp)
            user_episodes[user_id].append({
                'tensor': exp_tensor,
                'timestamp': exp.get('timestamp', datetime.now().timestamp()),
                'importance': exp.get('importance', 0.5)
            })

        logger.info(f"Stored {len(experiences)} experiences for user {user_id}")

        return jsonify({
            'status': 'success',
            'experiences_stored': len(experiences),
            'total_episodes': len(user_episodes[user_id])
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
        "timeWindow": "24h"  // or "all"
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

        if user_id not in user_episodes or len(user_episodes[user_id]) == 0:
            return jsonify({'patterns': [], 'count': 0})

        episodes = user_episodes[user_id]

        # Filter by time window if needed
        if time_window != 'all':
            # TODO: Implement time-based filtering
            pass

        # Prepare sequences
        if len(episodes) < 2:
            return jsonify({'patterns': [], 'count': 0})

        # Stack tensors and timestamps
        experience_tensors = torch.stack([ep['tensor'] for ep in episodes])
        timestamps = torch.tensor([ep['timestamp'] for ep in episodes], dtype=torch.float32)

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

        if user_id not in user_episodes:
            return jsonify({'episodes': [], 'count': 0})

        episodes = user_episodes[user_id]

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


if __name__ == '__main__':
    port = config.episodic_lnn_port
    logger.info(f"Starting Episodic LNN Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=(config.node_env == 'development'))
