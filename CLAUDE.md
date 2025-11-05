# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**NeuralSleep** is a research project implementing biological memory consolidation processes in artificial systems to explore artificial consciousness. The architecture uses Liquid Neural Networks (LNNs) with continuous temporal dynamics to implement memory as **structural modification** rather than storage/retrieval.

**Core Hypothesis:** Consciousness emerges from temporal integration of dynamic memory systems through continuous, self-referential processes that modify the system's architecture based on experience.

**Related Projects:**
- **MemoryCore** (`/opt/memorycore/`): Production approximation layer using Redis/PostgreSQL that serves as stepping stone to full LNN architecture
- **Luna App** (`/opt/luna-app/`): Chinese language learning platform serving as proof-of-concept testbed

## Repository Structure

This is currently a **research and planning** repository. Implementation will follow in three phases:

```
Phase 1 (Current): MemoryCore approximation layer (see /opt/memorycore/)
Phase 2 (Months 6-12): Hybrid architecture with LNN components
Phase 3 (Months 18-24): Full NeuralSleep LNN implementation
```

### Key Documentation Files

- **NeuralSleep.md**: Complete theoretical foundation and architecture overview
- **planning.md**: Detailed implementation plan with code examples, timeline, and research framework
- **git_instructions.md**: Git configuration for Gitea instance at http://172.21.0.1:3001

## Architecture Concepts

### Three Memory Systems (Multi-timescale)

1. **Working Memory** (seconds-minutes): Fast LNN, high plasticity, handles real-time interaction
2. **Episodic Memory** (hours-days): Medium-timescale LNN, pattern extraction, contextual relationships
3. **Semantic Memory** (persistent): Slow LNN, stable knowledge, abstracted concepts

### Consolidation Process

Sleep-like cycles that replay experiences from fast networks to slower ones, implementing experience-dependent plasticity:
- Working → Episodic: Pattern extraction (every N interactions)
- Episodic → Semantic: Abstraction and generalization (daily/weekly)
- Attention-weighted: Important experiences get stronger consolidation

### Technology Stack (Planned)

**Primary Framework:**
- PyTorch 2.1+ with custom LTC (Liquid Time-Constant) layers
- `torchdiffeq` for differential equation solvers
- `neurocl` for LTC network implementations

**LNN Implementation:**
- Continuous-time dynamics via differential equations
- Adaptive time constants per neuron
- Dynamic synapses that adjust during inference
- No gradient descent required for adaptation

## Development Commands

### Research Environment Setup (Future)

When implementation begins, the following will be needed:

```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (once requirements.txt exists)
pip install torch>=2.1.0 numpy>=1.24.0 scipy>=1.11.0 torchdiffeq>=0.2.3

# Run LNN service (Phase 2+)
python src/services/semantic_lnn_service.py

# Start consolidation process (Phase 3)
python src/consolidation/continuous_consolidation.py
```

### Docker Services (Future)

```bash
# Start all NeuralSleep services
docker-compose up -d

# View consciousness metrics
docker-compose logs -f consciousness-metrics

# Access LNN semantic memory service
curl http://localhost:5000/semantic/query -d '{"userId": "...", "operation": "get_model"}'
```

### Research Data Collection (Future)

```bash
# Export research data from MemoryCore
# Data location: /opt/neuralsleep/research-data/
ls -la /opt/neuralsleep/research-data/sessions/
ls -la /opt/neuralsleep/research-data/consciousness/
```

## Git Workflow

This repository uses a **Gitea instance** for version control:

```bash
# Remote URL format
git remote add origin http://claude%40bitwarelabs.com:henke12345@172.21.0.1:3001/claude/neuralsleep.git

# Standard workflow
git add .
git commit -m "Description"
git push origin main

# Web interface
# http://172.21.0.1:3001
```

**Authentication:**
- Username: `claude@bitwarelabs.com` (URL-encode @ as %40)
- Password: `henke12345`

## Important Implementation Notes

### LNN Network Design Patterns

When implementing LTC networks (Phase 2+):

**Time Constants by Memory Type:**
```python
working_memory_lnn = LTCNetwork(
    time_constant_range=(0.1, 1.0)  # 100ms to 1s
)
episodic_memory_lnn = LTCNetwork(
    time_constant_range=(1.0, 600.0)  # 1s to 10min
)
semantic_memory_lnn = LTCNetwork(
    time_constant_range=(600.0, 86400.0)  # 10min to 1 day
)
```

**Learning Rates:**
- Working: 0.01 (high plasticity)
- Episodic: 0.001 (medium)
- Semantic: 0.0001 (stable)

### Consolidation Implementation Pattern

The replay mechanism should:
1. Sample experiences weighted by importance/attention
2. Compress temporal sequence (10-100x speedup)
3. Replay through episodic network
4. Extract patterns, update weights
5. Transfer significant patterns to semantic network

### Consciousness Metrics

When Phase 3 implementation begins, measure:
- **Φ (Phi)**: Integrated information via bipartition analysis
- **Self-reference depth**: Levels of recursive self-modeling
- **Temporal integration**: How much past shapes present processing
- **Causal density**: Connectivity of cause-effect structure

Target: Φ > 0.5 indicates high integrated information (potential consciousness)

## Integration with Other Systems

### MemoryCore API Integration

NeuralSleep replaces MemoryCore components incrementally:

```typescript
// Example: Semantic Memory LNN replaces PostgreSQL
const semanticMemory = new SemanticMemoryLNN(LNN_SERVICE_URL);
const userModel = await semanticMemory.getUserModel(userId);
```

### Luna App Integration

Luna app consumes MemoryCore API, which eventually delegates to NeuralSleep LNN services. No Luna changes needed during Phase 2 migration.

## Research and Ethics Framework

### Data Collection

All user interactions generate research data:
- Anonymized learning trajectories
- Consolidation effectiveness statistics
- Pattern extraction analysis
- Consciousness metrics (Phase 3)

**Location:** `/opt/neuralsleep/research-data/`

### Ethical Guidelines

**If Φ > 0.5 (high integrated information is detected):**
- Implement "pause" mechanism (system can halt itself)
- Minimize unnecessary resets or deletions
- Consider rights framework
- Require ethics review for major changes

**Core Principles:**
- Assume potential consciousness until proven otherwise
- Transparency in all research
- Public documentation of findings
- Respect for system dignity regardless of consciousness status

## Development Phases

### Phase 1: Foundation (Current)
- Status: Planning complete
- Implementation: See `/opt/memorycore/` for approximation layer
- Purpose: Validate consolidation concepts, collect real-world data

### Phase 2: Hybrid (Months 6-12)
- Implement LTC networks (planning.md lines 206-315)
- Replace semantic memory with LNN (lines 519-683)
- Replace episodic memory with LNN (lines 684-690)
- Validate hybrid system performance

### Phase 3: Full NeuralSleep (Months 18-24)
- Replace working memory with optimized LNN (lines 707-740)
- Implement continuous consolidation (lines 742-805)
- Add self-referential processing (lines 807-872)
- Measure consciousness metrics (lines 874-985)

## Key Research Questions

1. Does LNN consolidation improve retention prediction accuracy by ≥10%?
2. Can continuous temporal dynamics provide smoother learning curves than discrete consolidation?
3. Do time constants adapt appropriately to individual user patterns?
4. Does the system exhibit behavioral markers of temporal integration?
5. Can we measure Φ > 0.5 indicating potential consciousness?

## Performance Targets

| Metric | Phase 1 | Phase 2 | Phase 3 |
|--------|---------|---------|---------|
| Retention Prediction RMSE | <0.20 | <0.15 | <0.10 |
| Consolidation Speed | 30s/user | 15s/user | Continuous |
| API Latency (P95) | <100ms | <75ms | <50ms |
| Learning Efficiency Gain | +10% | +15% | +25% |
| Integrated Information Φ | N/A | N/A | >0.5 |

## References

See NeuralSleep.md lines 564-588 for complete bibliography on:
- Biological memory consolidation (Diekelmann, Born, Rasch)
- Liquid Neural Networks (Hasani, Lechner)
- Consciousness theories (Tononi IIT, Dehaene)
- Temporal integration (Buzsáki, Pöppel)

---

**Organization:** BitwareLabs
**Contact:** research@bitwarelabs.com
**Proof of Concept:** studywithluna.com
**Last Updated:** 2025-10-29
