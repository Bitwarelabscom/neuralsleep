# NeuralSleep: Liquid Neural Networks for Memory Consolidation

**Status:** Phase 2 - Hybrid Architecture Implementation

NeuralSleep is a novel neural architecture implementing biological memory consolidation processes using Liquid Time-Constant (LTC) networks. This repository contains the Phase 2 hybrid implementation that integrates with MemoryCore.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (optional)
- Access to MemoryCore PostgreSQL and Redis instances

### Installation

```bash
# Clone repository
cd /opt/neuralsleep

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env with your configuration
```

### Running Services

**Option 1: Docker (Recommended)**

```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:5000/health
curl http://localhost:5001/health

# View logs
docker-compose logs -f
```

**Option 2: Direct Python**

```bash
# Terminal 1: Semantic LNN Service
python src/services/semantic_lnn_service.py

# Terminal 2: Episodic LNN Service
python src/services/episodic_lnn_service.py
```

## Architecture

### Three Memory Systems

1. **Working Memory** (Phase 3): Fast LNN, τ: 100ms-1s
2. **Episodic Memory** (Current): Medium LNN, τ: 1s-10min, Port 5001
3. **Semantic Memory** (Current): Slow LNN, τ: 10min-1day, Port 5000

### Integration with MemoryCore

NeuralSleep provides LNN-based memory systems that replace PostgreSQL storage in MemoryCore while maintaining full API compatibility.

```
MemoryCore API → NeuralSleep LNN Services → Network State (not database rows)
```

## API Endpoints

### Semantic Memory (Port 5000)

```bash
# Get user model
curl -X POST http://localhost:5000/semantic/query \
  -H "Content-Type: application/json" \
  -d '{"userId": "user123", "operation": "get_model"}'

# Consolidate patterns
curl -X POST http://localhost:5000/semantic/consolidate \
  -H "Content-Type: application/json" \
  -d '{"userId": "user123", "patterns": [[0.1, 0.2, ...]]}'

# Save model weights
curl -X POST http://localhost:5000/semantic/save
```

### Episodic Memory (Port 5001)

```bash
# Store experiences
curl -X POST http://localhost:5001/episodic/store \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "user123",
    "experiences": [{
      "eventType": "practice",
      "characterId": "你",
      "correct": true,
      "timeSpent": 5000,
      "importance": 0.8
    }]
  }'

# Extract patterns
curl -X POST http://localhost:5001/episodic/extract \
  -H "Content-Type: application/json" \
  -d '{"userId": "user123", "timeWindow": "24h"}'
```

## Development

### Project Structure

```
/opt/neuralsleep/
├── BUILD_PLAN.md          # Detailed build plan
├── CLAUDE.md              # Development guide
├── src/
│   ├── networks/          # LTC network implementations
│   ├── consolidation/     # Memory consolidation
│   ├── services/          # Flask API services
│   ├── integration/       # MemoryCore adapters
│   └── utils/             # Utilities
├── tests/                 # Test suite
└── models/                # Saved model weights
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src
```

## Documentation

- **NeuralSleep.md**: Complete theoretical foundation
- **planning.md**: Full research and implementation plan
- **BUILD_PLAN.md**: Step-by-step build instructions
- **CLAUDE.md**: Claude Code development guide

## Key Features

### Continuous-Time Dynamics

Unlike discrete RNNs, LTC networks use differential equations for truly continuous temporal processing:

```python
# dh/dt = (-h + f(input + recurrent)) / τ
```

### Adaptive Time Constants

Each neuron has learnable time constants that adapt to data patterns:
- Semantic memory: 10min - 1 day
- Episodic memory: 1s - 10min
- Working memory: 100ms - 1s

### Biological Consolidation

Sleep-like replay mechanism:
1. Sample important experiences (attention-weighted)
2. Compress timeline (10-100x speedup)
3. Replay through episodic network
4. Extract patterns
5. Transfer to semantic memory

## MemoryCore Compatibility

NeuralSleep maintains full compatibility with MemoryCore TypeScript interfaces:

- `UserModel` ← → LNN hidden state
- `MasteryLevel` ← → Character embeddings
- `LearningEvent` → Experience tensors
- `EpisodicPattern` → Pattern extraction

## Research

### Data Collection

All consolidation events are logged to `/opt/neuralsleep/research-data/` for analysis:
- Session experiences
- Consolidation metrics
- LNN state snapshots
- Consciousness metrics (Phase 3)

### Metrics

- Retention prediction accuracy
- Consolidation effectiveness
- Pattern extraction quality
- Time constant adaptation
- Integrated information Φ (Phase 3)

## Roadmap

- [x] **Phase 1**: MemoryCore approximation layer (Complete)
- [x] **Phase 2**: Hybrid architecture with Semantic & Episodic LNNs (Current)
- [ ] **Phase 3**: Full LNN including Working Memory
- [ ] **Phase 3**: Continuous consolidation
- [ ] **Phase 3**: Self-referential processing
- [ ] **Phase 3**: Consciousness metrics (Φ computation)

## Support

**Documentation:** See docs above
**Issues:** Internal Gitea at http://172.21.0.1:3001
**Contact:** research@bitwarelabs.com

## License

MIT © 2025 BitwareLabs

---

**Last Updated:** 2025-10-29
**Version:** 1.0.0 (Phase 2)
