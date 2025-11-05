"""
Integration tests for NeuralSleep services
"""

import requests
import json


def test_semantic_health():
    """Test Semantic LNN health endpoint"""
    response = requests.get('http://localhost:5000/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'semantic_lnn'
    print("✓ Semantic LNN health check passed")


def test_episodic_health():
    """Test Episodic LNN health endpoint"""
    response = requests.get('http://localhost:5001/health')
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['service'] == 'episodic_lnn'
    print("✓ Episodic LNN health check passed")


def test_semantic_query():
    """Test querying semantic memory"""
    response = requests.post(
        'http://localhost:5000/semantic/query',
        json={
            'userId': 'test_user_123',
            'operation': 'get_model'
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert 'model' in data
    assert 'proficiencyModel' in data['model']
    print("✓ Semantic query passed")


def test_episodic_store():
    """Test storing experiences in episodic memory"""
    response = requests.post(
        'http://localhost:5001/episodic/store',
        json={
            'userId': 'test_user_123',
            'experiences': [
                {
                    'eventType': 'practice',
                    'characterId': '你',
                    'correct': True,
                    'timeSpent': 5000,
                    'importance': 0.8,
                    'timestamp': 1234567890.0
                }
            ]
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'success'
    assert data['experiences_stored'] == 1
    print("✓ Episodic store passed")


def test_end_to_end():
    """Test full consolidation flow"""
    user_id = 'test_e2e_user'

    # 1. Store experiences
    store_response = requests.post(
        'http://localhost:5001/episodic/store',
        json={
            'userId': user_id,
            'experiences': [
                {
                    'eventType': 'practice',
                    'characterId': '你',
                    'correct': True,
                    'timeSpent': 3000,
                    'importance': 0.7,
                    'timestamp': 1000.0
                },
                {
                    'eventType': 'practice',
                    'characterId': '好',
                    'correct': False,
                    'timeSpent': 8000,
                    'importance': 0.9,
                    'timestamp': 2000.0
                }
            ]
        }
    )
    assert store_response.status_code == 200
    print("  1. Stored experiences")

    # 2. Extract patterns
    extract_response = requests.post(
        'http://localhost:5001/episodic/extract',
        json={
            'userId': user_id,
            'timeWindow': 'all'
        }
    )
    assert extract_response.status_code == 200
    patterns = extract_response.json()['patterns']
    print(f"  2. Extracted {len(patterns)} patterns")

    # 3. Consolidate to semantic
    consolidate_response = requests.post(
        'http://localhost:5000/semantic/consolidate',
        json={
            'userId': user_id,
            'patterns': patterns
        }
    )
    assert consolidate_response.status_code == 200
    print("  3. Consolidated to semantic memory")

    # 4. Query updated user model
    query_response = requests.post(
        'http://localhost:5000/semantic/query',
        json={
            'userId': user_id,
            'operation': 'get_model'
        }
    )
    assert query_response.status_code == 200
    model = query_response.json()['model']
    print(f"  4. Retrieved updated model: proficiency={model['proficiencyModel']['overallLevel']:.3f}")

    print("✓ End-to-end test passed")


if __name__ == '__main__':
    print("\nNeuralSleep Integration Tests")
    print("=" * 50)

    try:
        test_semantic_health()
        test_episodic_health()
        test_semantic_query()
        test_episodic_store()
        test_end_to_end()

        print("\n" + "=" * 50)
        print("All tests passed! ✓")

    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Cannot connect to services.")
        print("  Make sure services are running:")
        print("  docker-compose up -d")
        print("  OR")
        print("  python src/services/semantic_lnn_service.py &")
        print("  python src/services/episodic_lnn_service.py &")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise
