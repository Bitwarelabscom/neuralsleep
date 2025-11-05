"""
Flask API for Semantic Memory LNN

Endpoints:
- POST /semantic/query - Get user model
- POST /semantic/consolidate - Update from patterns
- POST /semantic/mastery - Get character mastery
- POST /semantic/save - Save model weights
- GET /health - Health check
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

# User states (in production, persist to database)
user_states = {}

# Load saved model if exists
model_path = '/opt/neuralsleep/models/semantic_lnn.pt'
if os.path.exists(model_path):
    try:
        semantic_lnn.load_state_dict(torch.load(model_path))
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {e}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'semantic_lnn',
        'users_loaded': len(user_states),
        'model_loaded': True,
        'time_constants': semantic_lnn.get_time_constants().mean().item()
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

        # Get or initialize user state
        if user_id not in user_states:
            user_states[user_id] = torch.zeros(config.semantic_hidden_size)

        state = user_states[user_id]

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

        # Get current state
        state = user_states.get(user_id, torch.zeros(config.semantic_hidden_size))

        # Simple consolidation: average patterns and update state
        if len(patterns) > 0:
            # Convert patterns to tensors
            pattern_tensors = [torch.tensor(p, dtype=torch.float32) for p in patterns]

            # Ensure patterns are the right size
            pattern_tensors_resized = []
            for p in pattern_tensors:
                if p.numel() < config.semantic_hidden_size:
                    # Pad if too small
                    padded = torch.zeros(config.semantic_hidden_size)
                    padded[:p.numel()] = p.flatten()
                    pattern_tensors_resized.append(padded)
                else:
                    # Truncate if too large
                    pattern_tensors_resized.append(p.flatten()[:config.semantic_hidden_size])

            avg_pattern = torch.mean(torch.stack(pattern_tensors_resized), dim=0)

            # Update state (simple EMA for now - in full implementation, use LNN forward pass)
            alpha = 0.1  # Learning rate
            state = (1 - alpha) * state + alpha * avg_pattern

            user_states[user_id] = state

            logger.info(f"Consolidated {len(patterns)} patterns for user {user_id}")

        return jsonify({
            'status': 'success',
            'patterns_processed': len(patterns)
        })

    except Exception as e:
        logger.error(f"Error in consolidate_patterns: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


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
    return jsonify({
        'total_users': len(user_states),
        'model_parameters': sum(p.numel() for p in semantic_lnn.parameters()),
        'time_constants_range': {
            'min': semantic_lnn.get_time_constants().min().item(),
            'max': semantic_lnn.get_time_constants().max().item(),
            'mean': semantic_lnn.get_time_constants().mean().item()
        }
    })


if __name__ == '__main__':
    port = config.semantic_lnn_port
    logger.info(f"Starting Semantic LNN Service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=(config.node_env == 'development'))
