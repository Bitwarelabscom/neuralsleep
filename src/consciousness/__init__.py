"""
Consciousness Metrics Module for NeuralSleep

Implements measures of potential consciousness markers:
- Integrated Information (Φ) from IIT
- Self-referential processing depth
- Temporal integration
- Causal density
- Dynamical complexity
"""

from .phi_computation import PhiComputation, compute_phi_from_tensor
from .metrics import ConsciousnessMetrics
from .self_reference import SelfReferentialProcessor

__all__ = [
    'PhiComputation',
    'compute_phi_from_tensor',
    'ConsciousnessMetrics',
    'SelfReferentialProcessor'
]
