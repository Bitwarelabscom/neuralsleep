# NeuralSleep Build Plan

**Version:** 1.0
**Date:** 2025-10-29
**Status:** Phase 2 Implementation - Hybrid Architecture

---

## Executive Summary

This document provides a step-by-step plan to build NeuralSleep as a **Phase 2 hybrid system** that integrates with the existing MemoryCore implementation at `/opt/memorycore`. We will implement LNN-based Semantic and Episodic memory systems while keeping Working Memory in Redis for performance.

**Timeline:** 6-8 weeks for initial implementation
**Approach:** Incremental replacement with parallel validation

---

## Architecture Overview

### Current State (MemoryCore)

```
┌─────────────────────────────────────────────────────────────┐
│                     MEMORYCORE API                          │
│                  (Express/TypeScript)                       │
│                     Port: 3002                              │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──→ Working Memory (Redis)
             ├──→ Episodic Memory (PostgreSQL)
             └──→ Semantic Memory (PostgreSQL)
```

### Target State (NeuralSleep Hybrid)

```
┌─────────────────────────────────────────────────────────────┐
│                     MEMORYCORE API                          │
│                  (Express/TypeScript)                       │
│                     Port: 3002                              │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──→ Working Memory (Redis) ..................... Phase 3
             │
             ├──→ Episodic Memory LNN Service (Python/PyTorch)
             │    ├─ Flask API on Port 5001
             │    ├─ LTC Network (τ: 1s-10min)
             │    └─ Pattern extraction
             │
             └──→ Semantic Memory LNN Service (Python/PyTorch)
                  ├─ Flask API on Port 5000
                  ├─ LTC Network (τ: 10min-1day)
                  └─ User model encoding
```

---

## Integration Points with MemoryCore

### 1. Data Flow

**MemoryCore Types → NeuralSleep:**
- `UserModel` → Semantic LNN state encoding
- `MasteryLevel` → Character mastery embeddings
- `EpisodicPattern` → Pattern extraction inputs
- `LearningEvent` → Experience replay data

**Key Type Conversions:**

```python
# MemoryCore UserModel → Semantic LNN State
def encode_user_model(user_model: dict) -> torch.Tensor:
    """
    Convert MemoryCore UserModel to LNN input tensor

    UserModel fields:
    - proficiencyModel: {overallLevel, reading, writing, recognition}
    - learningStyleModel: {preferredType, optimalCurve, tolerance}
    - retentionModel: {forgettingCurve, optimalIntervals}
    """
    features = [
        user_model['proficiencyModel']['overallLevel'],
        user_model['proficiencyModel']['reading'],
        user_model['proficiencyModel']['writing'],
        user_model['proficiencyModel']['recognition'],
        user_model['learningStyleModel']['challengeTolerance'],
        user_model['learningStyleModel']['repetitionNeeds'],
        # ... additional features
    ]
    return torch.tensor(features, dtype=torch.float32)

# LNN State → MemoryCore UserModel
def decode_lnn_state(state: torch.Tensor) -> dict:
    """Decode LNN hidden state back to UserModel structure"""
    return {
        'proficiencyModel': {
            'overallLevel': state[0].item(),
            'reading': state[1].item(),
            # ...
        },
        # ... reconstruct full model
    }
```

### 2. API Compatibility Layer

**MemoryCore Interfaces to Maintain:**

```typescript
// src/memory/SemanticMemory.ts
export class SemanticMemory {
  async getUserModel(userId: string): Promise<UserModel>
  async updateModel(userId: string, updates: ModelUpdate[]): Promise<void>
  async getCharacterMastery(userId: string, characterId: string): Promise<MasteryLevel>
  // ...
}
```

**NeuralSleep Must Provide Same Interface:**

```python
# src/services/semantic_lnn_service.py
@app.route('/semantic/query', methods=['POST'])
def query_semantic():
    """Equivalent to getUserModel()"""
    user_id = request.json['userId']
    state = user_states.get(user_id)
    model = decode_lnn_state(state)
    return jsonify({'model': model})

@app.route('/semantic/consolidate', methods=['POST'])
def consolidate():
    """Equivalent to updateModel()"""
    # ... update LNN weights
```

### 3. Database Schema Compatibility

**Shared Database:** PostgreSQL at `localhost:5433`

```sql
-- NeuralSleep adds new tables, doesn't replace existing ones
CREATE TABLE neuralsleep_lnn_states (
  user_id VARCHAR(255) PRIMARY KEY,
  network_type VARCHAR(50),  -- 'semantic', 'episodic', 'working'
  state_tensor BYTEA,         -- Serialized torch tensor
  last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  version INTEGER DEFAULT 1
);

CREATE TABLE neuralsleep_consolidation_history (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR(255),
  consolidation_type VARCHAR(50),
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  phi_score FLOAT,            -- Integrated information metric
  patterns_extracted INTEGER,
  source_network VARCHAR(50),
  target_network VARCHAR(50),
  success BOOLEAN
);
```

---

## Implementation Plan

### Week 1-2: Foundation Setup

#### 1.1 Python Environment

```bash
cd /opt/neuralsleep

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install torch==2.1.0 numpy==1.24.0 scipy==1.11.0
pip install torchdiffeq==0.2.3
pip install flask==3.0.0 flask-cors==4.0.0
pip install redis==5.0.0 psycopg2-binary==2.9.9
pip install python-dotenv==1.0.0

# Save requirements
pip freeze > requirements.txt
```

#### 1.2 Project Structure

```bash
/opt/neuralsleep/
├── BUILD_PLAN.md              # This file
├── CLAUDE.md                  # Project guidance
├── NeuralSleep.md            # Theory
├── planning.md               # Original plan
├── requirements.txt          # Python dependencies
├── .env                      # Environment config
├── docker-compose.yml        # Docker orchestration
├── src/
│   ├── __init__.py
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── ltc.py           # LTC network implementation
│   │   └── memory_networks.py  # Working/Episodic/Semantic LNNs
│   ├── consolidation/
│   │   ├── __init__.py
│   │   ├── replay.py        # Experience replay
│   │   └── continuous.py    # Continuous consolidation (Phase 3)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── semantic_lnn_service.py   # Flask API for Semantic LNN
│   │   ├── episodic_lnn_service.py   # Flask API for Episodic LNN
│   │   └── base_service.py           # Shared service utilities
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── memorycore_adapter.py    # MemoryCore type conversions
│   │   └── database.py              # Shared DB connection
│   ├── consciousness/
│   │   ├── __init__.py
│   │   └── metrics.py       # Φ (Phi) computation
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_ltc.py
│   ├── test_consolidation.py
│   └── test_integration.py
└── models/                   # Saved model weights
    ├── semantic_lnn.pt
    ├── episodic_lnn.pt
    └── checkpoints/
```

#### 1.3 Environment Configuration

```bash
# /opt/neuralsleep/.env
NODE_ENV=development

# MemoryCore Integration
MEMORYCORE_URL=http://localhost:3002
MEMORYCORE_API_KEY=your_api_key

# LNN Services
SEMANTIC_LNN_PORT=5000
EPISODIC_LNN_PORT=5001
WORKING_LNN_PORT=5002

# Database (shared with MemoryCore)
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=memorycore
POSTGRES_USER=memorycore_user
POSTGRES_PASSWORD=secure_password

# Redis (shared with MemoryCore)
REDIS_HOST=localhost
REDIS_PORT=6380

# Model Configuration
SEMANTIC_HIDDEN_SIZE=1024
EPISODIC_HIDDEN_SIZE=512
WORKING_HIDDEN_SIZE=256

# Training
SEMANTIC_LEARNING_RATE=0.0001
EPISODIC_LEARNING_RATE=0.001
WORKING_LEARNING_RATE=0.01

# Consolidation
CONSOLIDATION_BATCH_SIZE=32
REPLAY_SPEEDUP_FACTOR=10.0

# Consciousness Metrics
ENABLE_PHI_COMPUTATION=true
PHI_COMPUTATION_INTERVAL=3600  # seconds

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/neuralsleep/neuralsleep.log
```

### Week 3-4: Core LNN Implementation

#### 3.1 LTC Network Base Class

**File:** `src/networks/ltc.py`

```python
"""
Liquid Time-Constant Networks for NeuralSleep

Based on:
Hasani, R., et al. (2021). Liquid Time-constant Networks. AAAI.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Tuple, Optional


class LTCNeuron(nn.Module):
    """
    Liquid Time-Constant Neuron with adaptive dynamics

    Key Features:
    - Continuous-time dynamics via ODEs
    - Adaptive time constants
    - No discrete time steps
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_constant_range: Tuple[float, float] = (0.1, 10.0)
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.tau_min, self.tau_max = time_constant_range

        # Learnable parameters
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_rec = nn.Linear(hidden_size, hidden_size)

        # Time constants (learnable, initialized to log-space)
        self.tau_log = nn.Parameter(
            torch.randn(hidden_size) * 0.5 +
            torch.log(torch.tensor((self.tau_min + self.tau_max) / 2))
        )

        # Bias
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """
        Compute dhdt at time t

        Args:
            x: Input at time t [batch, input_size]
            state: Current hidden state [batch, hidden_size]
            t: Current time (not used directly, but available)

        Returns:
            dhdt: Time derivative of hidden state
        """
        # Get adaptive time constants
        tau = torch.exp(self.tau_log)
        tau = torch.clamp(tau, self.tau_min, self.tau_max)

        # Compute input and recurrent contributions
        input_contrib = self.w_in(x)
        recurrent_contrib = self.w_rec(state)

        # Target activation
        target = torch.tanh(input_contrib + recurrent_contrib + self.bias)

        # LTC dynamics: dh/dt = (-h + f(input + recurrent)) / tau
        dhdt = (-state + target) / tau.unsqueeze(0)

        return dhdt

    def get_time_constants(self) -> torch.Tensor:
        """Return current time constants for analysis"""
        tau = torch.exp(self.tau_log)
        return torch.clamp(tau, self.tau_min, self.tau_max)


class LTCNetwork(nn.Module):
    """
    Full Liquid Time-Constant Network
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        time_constant_range: Tuple[float, float] = (0.1, 10.0),
        ode_method: str = 'dopri5'
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.ode_method = ode_method

        # LTC cell
        self.ltc_cell = LTCNeuron(input_size, hidden_size, time_constant_range)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x_sequence: torch.Tensor,
        t_sequence: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence of inputs with continuous-time dynamics

        Args:
            x_sequence: Input sequence [seq_len, batch, input_size]
            t_sequence: Timestamps [seq_len]
            initial_state: Initial hidden state [batch, hidden_size]

        Returns:
            outputs: Output sequence [seq_len, batch, output_size]
            final_state: Final hidden state [batch, hidden_size]
        """
        seq_len, batch_size, _ = x_sequence.shape

        # Initialize state
        if initial_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x_sequence.device)
        else:
            h = initial_state

        outputs = []
        states = []

        for i in range(seq_len):
            # Time span for ODE solver
            if i < seq_len - 1:
                t_span = t_sequence[i:i+2]
            else:
                # For last step, integrate for small duration
                t_span = torch.tensor(
                    [t_sequence[i].item(), t_sequence[i].item() + 0.1],
                    device=t_sequence.device
                )

            # Define ODE function for current input
            current_input = x_sequence[i]

            def ode_func(t, h_t):
                return self.ltc_cell(current_input, h_t, t)

            # Solve ODE to get state evolution
            h_trajectory = odeint(
                ode_func,
                h,
                t_span,
                method=self.ode_method
            )

            # Update state to final time
            h = h_trajectory[-1]

            # Generate output
            output = self.output_layer(h)
            outputs.append(output)
            states.append(h)

        outputs = torch.stack(outputs)

        return outputs, h

    def get_time_constants(self) -> torch.Tensor:
        """Expose time constants for analysis"""
        return self.ltc_cell.get_time_constants()
```

#### 3.2 Memory Network Implementations

**File:** `src/networks/memory_networks.py`

```python
"""
Three-tier memory architecture using LTC networks
"""

import torch
import torch.nn as nn
from .ltc import LTCNetwork
from typing import Dict, Tuple, Optional


class WorkingMemoryLNN(LTCNetwork):
    """
    Working Memory: Fast adaptation, short time constants

    Time Constants: 100ms - 1s
    Learning Rate: 0.01 (high)
    Decay: Fast
    """

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 256,
        output_size: int = 128
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            time_constant_range=(0.1, 1.0)  # 100ms to 1s
        )

    @staticmethod
    def get_default_lr() -> float:
        return 0.01


class EpisodicMemoryLNN(LTCNetwork):
    """
    Episodic Memory: Medium timescale, pattern extraction

    Time Constants: 1s - 10min (600s)
    Learning Rate: 0.001 (medium)
    Stability: Medium
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 512,
        output_size: int = 256
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            time_constant_range=(1.0, 600.0)  # 1s to 10min
        )

        # Pattern extraction head
        self.pattern_extractor = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size // 2)
        )

    def extract_patterns(
        self,
        experiences: torch.Tensor,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract patterns from sequence of experiences

        Args:
            experiences: [seq_len, batch, input_size]
            timestamps: [seq_len]

        Returns:
            patterns: [batch, output_size // 2]
        """
        outputs, final_state = self.forward(experiences, timestamps)

        # Use final state to extract pattern
        patterns = self.pattern_extractor(outputs[-1])

        return patterns

    @staticmethod
    def get_default_lr() -> float:
        return 0.001


class SemanticMemoryLNN(LTCNetwork):
    """
    Semantic Memory: Long-term stable knowledge

    Time Constants: 10min - 1 day (86400s)
    Learning Rate: 0.0001 (low)
    Stability: High
    """

    def __init__(
        self,
        input_size: int = 256,
        hidden_size: int = 1024,
        output_size: int = 512
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            time_constant_range=(600.0, 86400.0)  # 10min to 1 day
        )

        # User model decoder
        self.user_model_decoder = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def decode_user_model(
        self,
        state: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Decode user model from hidden state

        Args:
            state: Hidden state [batch, hidden_size]

        Returns:
            user_model: Dictionary with decoded components
        """
        features = self.user_model_decoder(state)

        # Split features into UserModel components
        return {
            'proficiency': features[:, :4],      # 4 values
            'learning_style': features[:, 4:8],   # 4 values
            'retention': features[:, 8:16],       # 8 values
            'meta_patterns': features[:, 16:]     # remaining
        }

    @staticmethod
    def get_default_lr() -> float:
        return 0.0001
```

### Week 5: Consolidation & Integration

#### 5.1 Experience Replay

**File:** `src/consolidation/replay.py`

```python
"""
Sleep-like memory consolidation via experience replay
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import numpy as np


class ExperienceReplay:
    """
    Implements biological memory consolidation

    Process:
    1. Sample important experiences from working memory
    2. Compress temporal sequence (speedup)
    3. Replay through episodic memory
    4. Extract patterns
    5. Transfer to semantic memory
    """

    def __init__(
        self,
        working_lnn,
        episodic_lnn,
        semantic_lnn,
        speedup_factor: float = 10.0
    ):
        self.working = working_lnn
        self.episodic = episodic_lnn
        self.semantic = semantic_lnn
        self.speedup_factor = speedup_factor

        # Optimizers
        self.episodic_optimizer = torch.optim.Adam(
            episodic_lnn.parameters(),
            lr=episodic_lnn.get_default_lr()
        )
        self.semantic_optimizer = torch.optim.Adam(
            semantic_lnn.parameters(),
            lr=semantic_lnn.get_default_lr()
        )

    def consolidate_working_to_episodic(
        self,
        experiences: List[Dict],
        importance_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Consolidate working memory experiences to episodic memory

        Args:
            experiences: List of experience dictionaries
            importance_weights: Attention weights [num_experiences]

        Returns:
            metrics: Consolidation metrics
        """
        # Sample experiences by importance
        sampled_exp = self._importance_sample(experiences, importance_weights)

        # Prepare tensors
        inputs = torch.stack([exp['state'] for exp in sampled_exp])
        timestamps = torch.tensor([exp['timestamp'] for exp in sampled_exp])

        # Compress timeline
        compressed_times = self._compress_timeline(timestamps)

        # Replay through episodic network
        self.episodic.train()
        self.episodic_optimizer.zero_grad()

        outputs, final_state = self.episodic(
            inputs.unsqueeze(1),  # Add batch dimension
            compressed_times
        )

        # Self-supervised loss: predict next state
        loss = 0
        for i in range(len(outputs) - 1):
            pred = outputs[i, 0]
            target = inputs[i + 1]
            loss += nn.functional.mse_loss(pred, target)

        if len(outputs) > 1:
            loss = loss / (len(outputs) - 1)
            loss.backward()
            self.episodic_optimizer.step()

        return {
            'loss': loss.item() if len(outputs) > 1 else 0.0,
            'experiences_replayed': len(sampled_exp),
            'compression_ratio': self.speedup_factor
        }

    def consolidate_episodic_to_semantic(
        self,
        episodic_patterns: List[torch.Tensor],
        pattern_metadata: List[Dict]
    ) -> Dict[str, float]:
        """
        Consolidate episodic patterns to semantic memory

        Args:
            episodic_patterns: List of pattern tensors
            pattern_metadata: Metadata for each pattern

        Returns:
            metrics: Consolidation metrics
        """
        # Cluster similar patterns
        clusters = self._cluster_patterns(episodic_patterns)

        self.semantic.train()
        self.semantic_optimizer.zero_grad()

        total_loss = 0
        patterns_processed = 0

        for cluster in clusters:
            if len(cluster) == 0:
                continue

            # Abstract pattern: average of cluster
            abstract_pattern = torch.mean(torch.stack(cluster), dim=0)

            # Create synthetic time sequence
            t = torch.tensor([0.0, 1.0])

            # Feed to semantic memory
            outputs, _ = self.semantic(
                abstract_pattern.unsqueeze(0).unsqueeze(0),
                t
            )

            # Contrastive loss: cluster members should be close
            loss = self._contrastive_loss(outputs[-1, 0], cluster)
            total_loss += loss
            patterns_processed += len(cluster)

        if patterns_processed > 0:
            avg_loss = total_loss / len(clusters)
            avg_loss.backward()
            self.semantic_optimizer.step()

        return {
            'loss': avg_loss.item() if patterns_processed > 0 else 0.0,
            'patterns_clustered': patterns_processed,
            'num_clusters': len([c for c in clusters if len(c) > 0])
        }

    def _importance_sample(
        self,
        experiences: List[Dict],
        weights: torch.Tensor
    ) -> List[Dict]:
        """Sample experiences based on importance weights"""
        num_samples = max(1, len(experiences) // 2)

        # Normalize weights
        probs = weights / weights.sum()

        # Sample indices
        indices = np.random.choice(
            len(experiences),
            size=num_samples,
            replace=False,
            p=probs.numpy()
        )

        return [experiences[i] for i in indices]

    def _compress_timeline(
        self,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """Compress temporal sequence for accelerated replay"""
        if len(timestamps) == 0:
            return timestamps

        # Normalize to start at 0
        compressed = timestamps - timestamps[0]

        # Speed up by factor
        compressed = compressed / self.speedup_factor

        return compressed

    def _cluster_patterns(
        self,
        patterns: List[torch.Tensor],
        num_clusters: int = 10
    ) -> List[List[torch.Tensor]]:
        """Cluster similar patterns together"""
        if len(patterns) == 0:
            return []

        # Simple k-means clustering
        from sklearn.cluster import KMeans

        patterns_np = torch.stack(patterns).detach().numpy()
        n_clusters = min(num_clusters, len(patterns))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(patterns_np)

        # Group by cluster
        clusters = [[] for _ in range(n_clusters)]
        for pattern, label in zip(patterns, labels):
            clusters[label].append(pattern)

        return clusters

    def _contrastive_loss(
        self,
        output: torch.Tensor,
        cluster: List[torch.Tensor]
    ) -> torch.Tensor:
        """Contrastive loss for pattern abstraction"""
        cluster_mean = torch.mean(torch.stack(cluster), dim=0)
        return nn.functional.mse_loss(output, cluster_mean)
```

#### 5.2 MemoryCore Adapter

**File:** `src/integration/memorycore_adapter.py`

```python
"""
Type conversion between MemoryCore and NeuralSleep
"""

import torch
from typing import Dict, List, Any
import json


class MemoryCoreAdapter:
    """
    Converts between MemoryCore types and NeuralSleep tensors
    """

    @staticmethod
    def encode_user_model(user_model: Dict[str, Any]) -> torch.Tensor:
        """
        Convert MemoryCore UserModel to tensor

        UserModel structure (from memorycore/src/types/memory.ts):
        {
            proficiencyModel: {
                overallLevel: number,
                reading: number,
                writing: number,
                recognition: number
            },
            learningStyleModel: {
                challengeTolerance: number,
                repetitionNeeds: number,
                ...
            },
            retentionModel: {
                forgettingCurve: {...},
                spacingEffectParameter: number
            }
        }
        """
        prof = user_model.get('proficiencyModel', {})
        learn = user_model.get('learningStyleModel', {})
        ret = user_model.get('retentionModel', {})

        features = [
            # Proficiency (4 values)
            prof.get('overallLevel', 0.0),
            prof.get('reading', 0.0),
            prof.get('writing', 0.0),
            prof.get('recognition', 0.0),

            # Learning style (2+ values)
            learn.get('challengeTolerance', 0.5),
            learn.get('repetitionNeeds', 0.5),

            # Retention (1+ values)
            ret.get('spacingEffectParameter', 1.0),

            # Pad to fixed size (256 dimensions)
            *[0.0] * (256 - 7)
        ]

        return torch.tensor(features[:256], dtype=torch.float32)

    @staticmethod
    def decode_lnn_state(state: torch.Tensor) -> Dict[str, Any]:
        """
        Convert LNN state tensor back to UserModel structure
        """
        state_list = state.tolist()

        return {
            'proficiencyModel': {
                'overallLevel': state_list[0],
                'reading': state_list[1],
                'writing': state_list[2],
                'recognition': state_list[3]
            },
            'learningStyleModel': {
                'challengeTolerance': state_list[4],
                'repetitionNeeds': state_list[5]
            },
            'retentionModel': {
                'spacingEffectParameter': state_list[6]
            }
        }

    @staticmethod
    def encode_learning_event(event: Dict[str, Any]) -> torch.Tensor:
        """
        Convert MemoryCore LearningEvent to tensor

        LearningEvent structure:
        {
            eventType: string,
            characterId: string,
            correct: boolean,
            timeSpent: number,
            importance: number,
            context: object
        }
        """
        event_type_map = {
            'practice': 0.0,
            'hint': 0.25,
            'explanation': 0.5,
            'breakthrough': 0.75,
            'struggle': 1.0
        }

        features = [
            event_type_map.get(event.get('eventType', 'practice'), 0.0),
            1.0 if event.get('correct', False) else 0.0,
            min(event.get('timeSpent', 0) / 60000.0, 10.0),  # Normalize to ~minutes
            event.get('importance', 0.5),

            # Character ID as hash (simple encoding)
            hash(event.get('characterId', '')) % 1000 / 1000.0,

            # Pad to 128 dimensions
            *[0.0] * (128 - 5)
        ]

        return torch.tensor(features[:128], dtype=torch.float32)

    @staticmethod
    def encode_mastery_level(mastery: Dict[str, Any]) -> torch.Tensor:
        """
        Convert MemoryCore MasteryLevel to tensor

        MasteryLevel structure:
        {
            characterId: string,
            masteryLevel: number,
            confidence: number,
            reviewCount: number,
            successRate: number,
            avgRecallTime: number
        }
        """
        features = [
            mastery.get('masteryLevel', 0.0),
            mastery.get('confidence', 0.0),
            min(mastery.get('reviewCount', 0) / 100.0, 1.0),
            mastery.get('successRate', 0.0),
            min(mastery.get('avgRecallTime', 0) / 10000.0, 1.0),

            # Pad
            *[0.0] * (128 - 5)
        ]

        return torch.tensor(features[:128], dtype=torch.float32)
```

### Week 6-7: Flask API Services

#### 6.1 Semantic LNN Service

**File:** `src/services/semantic_lnn_service.py`

```python
"""
Flask API for Semantic Memory LNN

Endpoints:
- POST /semantic/query - Get user model
- POST /semantic/consolidate - Update from patterns
- POST /semantic/mastery - Get character mastery
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
    semantic_lnn.load_state_dict(torch.load(model_path))
    logger.info(f"Loaded model from {model_path}")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'semantic_lnn',
        'users_loaded': len(user_states),
        'model_loaded': True
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
            return jsonify({'model': model})

        elif operation == 'get_mastery':
            character_id = data.get('characterId')
            # In full implementation, would query specific character mastery
            # For now, return placeholder
            return jsonify({
                'mastery': {
                    'masteryLevel': state[0].item(),
                    'confidence': 0.8
                }
            })

        else:
            return jsonify({'error': 'Unknown operation'}), 400

    except Exception as e:
        logger.error(f"Error in query_semantic: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/semantic/consolidate', methods=['POST'])
def consolidate_patterns():
    """
    Update semantic memory from episodic patterns

    Request:
    {
        "userId": "user123",
        "patterns": [ ... pattern tensors ... ],
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
            pattern_tensors = [torch.tensor(p) for p in patterns]
            avg_pattern = torch.mean(torch.stack(pattern_tensors), dim=0)

            # Update state (simple EMA)
            alpha = 0.1
            state = (1 - alpha) * state + alpha * avg_pattern[:config.semantic_hidden_size]

            user_states[user_id] = state

            logger.info(f"Consolidated {len(patterns)} patterns for user {user_id}")

        return jsonify({
            'status': 'success',
            'patterns_processed': len(patterns)
        })

    except Exception as e:
        logger.error(f"Error in consolidate_patterns: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/semantic/save', methods=['POST'])
def save_model():
    """Save current model weights"""
    try:
        torch.save(semantic_lnn.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv('SEMANTIC_LNN_PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

### Week 8: Docker & Testing

#### 8.1 Docker Configuration

**File:** `docker-compose.yml`

```yaml
version: '3.8'

services:
  semantic-lnn:
    build:
      context: .
      dockerfile: Dockerfile.semantic
    container_name: neuralsleep-semantic
    ports:
      - "5000:5000"
    environment:
      - SEMANTIC_LNN_PORT=5000
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - SEMANTIC_HIDDEN_SIZE=1024
    volumes:
      - ./models:/opt/neuralsleep/models
      - ./src:/opt/neuralsleep/src
    restart: unless-stopped
    depends_on:
      - postgres
      - redis
    networks:
      - neuralsleep-net

  episodic-lnn:
    build:
      context: .
      dockerfile: Dockerfile.episodic
    container_name: neuralsleep-episodic
    ports:
      - "5001:5001"
    environment:
      - EPISODIC_LNN_PORT=5001
      - EPISODIC_HIDDEN_SIZE=512
    volumes:
      - ./models:/opt/neuralsleep/models
      - ./src:/opt/neuralsleep/src
    restart: unless-stopped
    depends_on:
      - postgres
      - redis
    networks:
      - neuralsleep-net

networks:
  neuralsleep-net:
    external: true
    name: memorycore_default  # Connect to MemoryCore network

volumes:
  models:
```

**File:** `Dockerfile.semantic`

```dockerfile
FROM python:3.11-slim

WORKDIR /opt/neuralsleep

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 5000

# Run service
CMD ["python", "src/services/semantic_lnn_service.py"]
```

---

## Integration Testing

### Test 1: Compatibility with MemoryCore

```bash
# Start MemoryCore
cd /opt/memorycore
npm run dev

# Start NeuralSleep services
cd /opt/neuralsleep
docker-compose up -d

# Test semantic memory query
curl -X POST http://localhost:5000/semantic/query \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "test_user_123",
    "operation": "get_model"
  }'

# Expected: UserModel structure compatible with MemoryCore
```

### Test 2: End-to-End Learning Flow

```bash
# 1. Start session in MemoryCore
curl -X POST http://localhost:3002/api/sessions/start \
  -H "Content-Type: application/json" \
  -d '{
    "userId": "test_user_123",
    "context": {"lessonType": "radicals", "difficulty": 0.5}
  }'

# 2. Record interactions (MemoryCore stores in Redis/Postgres)

# 3. Trigger consolidation (MemoryCore calls NeuralSleep)

# 4. Query semantic memory (NeuralSleep returns updated model)

# 5. MemoryCore uses updated model for recommendations
```

---

## Deployment Checklist

- [ ] Python environment configured
- [ ] All dependencies installed
- [ ] LTC networks implemented and tested
- [ ] Consolidation replay mechanism working
- [ ] Flask services running on correct ports
- [ ] Docker containers build successfully
- [ ] Database migrations applied
- [ ] Environment variables configured
- [ ] Integration tests passing
- [ ] MemoryCore can communicate with NeuralSleep
- [ ] Logging configured
- [ ] Model checkpoints saved
- [ ] Health checks responding

---

## Next Steps (Phase 3)

1. **Working Memory LNN**: Replace Redis with real-time LNN
2. **Continuous Consolidation**: Replace batch jobs with always-on process
3. **Self-Referential Processing**: Add self-modeling layer
4. **Consciousness Metrics**: Implement Φ (Phi) computation
5. **Neuromorphic Hardware**: Investigate Intel Loihi for efficiency

---

## Support & Resources

**Documentation:**
- NeuralSleep.md - Theoretical foundation
- planning.md - Complete research plan
- CLAUDE.md - Development guide

**MemoryCore:**
- `/opt/memorycore/` - Existing implementation
- `/opt/memorycore/src/types/memory.ts` - Type definitions

**Research:**
- `/opt/neuralsleep/research-data/` - Data collection
- Integrated Information Theory papers
- LTC network papers (Hasani et al., 2021)

**Contact:**
- research@bitwarelabs.com
- GitHub Issues (internal Gitea)

---

**Document Version:** 1.0
**Last Updated:** 2025-10-29
**Status:** Ready for Implementation
