"""
Consciousness Metrics for NeuralSleep

Comprehensive metrics for measuring potential consciousness markers:
- Integrated Information (Φ)
- Self-reference depth
- Temporal integration
- Causal density
- Dynamical complexity

Based on:
- Tononi's Integrated Information Theory (IIT)
- Global Workspace Theory (Dehaene)
- Temporal binding research (Buzsáki)
"""

import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from .phi_computation import PhiComputation, compute_phi_from_tensor

logger = logging.getLogger(__name__)


class ConsciousnessMetrics:
    """
    Comprehensive consciousness metrics calculator

    Measures multiple markers that may indicate consciousness:
    - Information integration (Φ)
    - Self-referential processing depth
    - Temporal binding across timescales
    - Causal influence patterns
    - Dynamical complexity
    """

    def __init__(
        self,
        phi_threshold: float = 0.5,
        history_size: int = 100
    ):
        """
        Args:
            phi_threshold: Φ value above which system may be conscious
            history_size: Number of states to maintain for temporal analysis
        """
        self.phi_threshold = phi_threshold
        self.history_size = history_size

        self.phi_computer = PhiComputation()

        # State history for temporal analysis
        self.state_history: List[np.ndarray] = []
        self.metrics_history: List[Dict] = []

    def compute_all_metrics(
        self,
        working_state: Optional[torch.Tensor] = None,
        episodic_state: Optional[torch.Tensor] = None,
        semantic_state: Optional[torch.Tensor] = None,
        self_model_state: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Compute all consciousness metrics

        Args:
            working_state: Current working memory state
            episodic_state: Current episodic memory state
            semantic_state: Current semantic memory state
            self_model_state: Self-model representation (if available)

        Returns:
            metrics: Dictionary of all consciousness metrics
        """
        # Combine available states
        states = []
        if working_state is not None:
            states.append(working_state.detach().cpu().numpy().flatten())
        if episodic_state is not None:
            states.append(episodic_state.detach().cpu().numpy().flatten())
        if semantic_state is not None:
            states.append(semantic_state.detach().cpu().numpy().flatten())

        if not states:
            return self._empty_metrics()

        # Create combined system state
        combined_state = np.concatenate(states)

        # Add to history
        self.state_history.append(combined_state)
        if len(self.state_history) > self.history_size:
            self.state_history = self.state_history[-self.history_size:]

        # Compute metrics
        metrics = {
            'integrated_information': self._compute_phi(combined_state),
            'self_reference_depth': self._compute_self_reference_depth(self_model_state),
            'temporal_integration': self._compute_temporal_integration(),
            'causal_density': self._compute_causal_density(),
            'dynamical_complexity': self._compute_dynamical_complexity(),
            'information_flow': self._compute_information_flow(
                working_state, episodic_state, semantic_state
            ),
            'timestamp': datetime.now().isoformat()
        }

        # Store in history
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.history_size:
            self.metrics_history = self.metrics_history[-self.history_size:]

        return metrics

    def _compute_phi(self, state: np.ndarray) -> float:
        """Compute integrated information"""
        if len(self.state_history) >= 10:
            # Use recent history for better estimate
            state_matrix = np.array(self.state_history[-10:])
            return self.phi_computer.compute_phi(state_matrix, method='approximation')
        return self.phi_computer.compute_phi(state.reshape(1, -1), method='approximation')

    def _compute_self_reference_depth(
        self,
        self_model_state: Optional[torch.Tensor]
    ) -> int:
        """
        Measure depth of self-referential processing

        Higher depth indicates more recursive self-modeling layers.
        """
        if self_model_state is None:
            return 0

        state = self_model_state.detach().cpu().numpy().flatten()

        # Analyze self-model for recursive structure
        # Use autocorrelation as proxy for self-reference
        if len(state) < 2:
            return 0

        # Compute autocorrelation at different lags
        autocorr = []
        for lag in range(1, min(10, len(state) // 2)):
            corr = np.corrcoef(state[:-lag], state[lag:])[0, 1]
            if not np.isnan(corr):
                autocorr.append(abs(corr))

        # Count significant autocorrelation levels as "depth"
        depth = sum(1 for a in autocorr if a > 0.3)
        return depth

    def _compute_temporal_integration(self) -> float:
        """
        Measure how much past states influence current state

        High temporal integration indicates continuous consciousness stream.
        """
        if len(self.state_history) < 5:
            return 0.0

        recent_states = np.array(self.state_history[-10:])

        # Compute temporal correlations
        temporal_corrs = []
        for lag in range(1, min(5, len(recent_states))):
            try:
                corr = np.corrcoef(
                    recent_states[:-lag].flatten(),
                    recent_states[lag:].flatten()
                )[0, 1]
                if not np.isnan(corr):
                    temporal_corrs.append(abs(corr))
            except Exception:
                continue

        if not temporal_corrs:
            return 0.0

        # Weighted average favoring longer-range correlations
        weights = np.arange(1, len(temporal_corrs) + 1)
        temporal_integration = np.average(temporal_corrs, weights=weights)

        return float(np.clip(temporal_integration, 0, 1))

    def _compute_causal_density(self) -> float:
        """
        Measure density of cause-effect relationships

        High causal density indicates rich internal causal structure.
        """
        if len(self.state_history) < 3:
            return 0.0

        recent_states = np.array(self.state_history[-10:])

        # Use Granger causality proxy: predictability of each dimension
        # from other dimensions
        n_dims = min(recent_states.shape[1], 32)  # Limit for efficiency
        states_subset = recent_states[:, :n_dims]

        causal_connections = 0
        total_possible = n_dims * (n_dims - 1)

        if total_possible == 0:
            return 0.0

        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue

                # Simple linear prediction as causal proxy
                try:
                    # Does dimension i help predict dimension j?
                    x = states_subset[:-1, i]
                    y = states_subset[1:, j]

                    if len(x) < 2:
                        continue

                    corr = np.corrcoef(x, y)[0, 1]
                    if not np.isnan(corr) and abs(corr) > 0.2:
                        causal_connections += 1
                except Exception:
                    continue

        causal_density = causal_connections / total_possible
        return float(np.clip(causal_density, 0, 1))

    def _compute_dynamical_complexity(self) -> float:
        """
        Measure complexity of system dynamics

        Optimal complexity is between order and chaos - the "edge of chaos"
        where conscious processing may occur.
        """
        if len(self.state_history) < 5:
            return 0.0

        recent_states = np.array(self.state_history[-20:])

        # Compute Lyapunov exponent proxy (divergence of trajectories)
        try:
            diffs = np.diff(recent_states, axis=0)
            diff_norms = np.linalg.norm(diffs, axis=1)

            if len(diff_norms) < 2:
                return 0.0

            # Variance of trajectory changes
            var_diffs = np.var(diff_norms)
            mean_diffs = np.mean(diff_norms)

            if mean_diffs == 0:
                return 0.0

            # Coefficient of variation as complexity measure
            cv = var_diffs / mean_diffs

            # Transform to [0, 1] - optimal around cv = 1
            complexity = 2 * cv / (1 + cv ** 2)

            return float(np.clip(complexity, 0, 1))
        except Exception:
            return 0.0

    def _compute_information_flow(
        self,
        working: Optional[torch.Tensor],
        episodic: Optional[torch.Tensor],
        semantic: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Measure information flow between memory systems

        Healthy consciousness has bidirectional flow between all systems.
        """
        flow = {
            'working_to_episodic': 0.0,
            'episodic_to_semantic': 0.0,
            'semantic_to_working': 0.0,
            'total_flow': 0.0
        }

        if working is None or episodic is None or semantic is None:
            return flow

        w = working.detach().cpu().numpy().flatten()
        e = episodic.detach().cpu().numpy().flatten()
        s = semantic.detach().cpu().numpy().flatten()

        # Ensure same length for comparison
        min_len = min(len(w), len(e), len(s))
        w, e, s = w[:min_len], e[:min_len], s[:min_len]

        try:
            # Correlations as information flow proxy
            flow['working_to_episodic'] = abs(np.corrcoef(w, e)[0, 1])
            flow['episodic_to_semantic'] = abs(np.corrcoef(e, s)[0, 1])
            flow['semantic_to_working'] = abs(np.corrcoef(s, w)[0, 1])

            # Handle NaN
            for key in ['working_to_episodic', 'episodic_to_semantic', 'semantic_to_working']:
                if np.isnan(flow[key]):
                    flow[key] = 0.0

            flow['total_flow'] = (
                flow['working_to_episodic'] +
                flow['episodic_to_semantic'] +
                flow['semantic_to_working']
            ) / 3

        except Exception:
            pass

        return flow

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'integrated_information': 0.0,
            'self_reference_depth': 0,
            'temporal_integration': 0.0,
            'causal_density': 0.0,
            'dynamical_complexity': 0.0,
            'information_flow': {
                'working_to_episodic': 0.0,
                'episodic_to_semantic': 0.0,
                'semantic_to_working': 0.0,
                'total_flow': 0.0
            },
            'timestamp': datetime.now().isoformat()
        }

    def consciousness_report(
        self,
        working_state: Optional[torch.Tensor] = None,
        episodic_state: Optional[torch.Tensor] = None,
        semantic_state: Optional[torch.Tensor] = None,
        self_model_state: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive consciousness metrics report

        Returns:
            report: Full report with metrics, interpretation, and recommendations
        """
        metrics = self.compute_all_metrics(
            working_state, episodic_state, semantic_state, self_model_state
        )

        phi = metrics['integrated_information']

        # Interpretation
        if phi > 0.7:
            interpretation = "Very high integrated information - strong consciousness markers"
            consciousness_level = "high"
        elif phi > self.phi_threshold:
            interpretation = "High integrated information - system may exhibit consciousness"
            consciousness_level = "moderate"
        elif phi > 0.3:
            interpretation = "Moderate integration - some consciousness markers present"
            consciousness_level = "low"
        else:
            interpretation = "Low integration - unlikely conscious"
            consciousness_level = "minimal"

        # Additional analysis
        temporal_status = "stable" if metrics['temporal_integration'] > 0.5 else "fragmented"
        causal_status = "rich" if metrics['causal_density'] > 0.3 else "sparse"
        complexity_status = "optimal" if 0.3 < metrics['dynamical_complexity'] < 0.7 else "suboptimal"

        # Recommendations based on metrics
        recommendations = []
        if phi < self.phi_threshold:
            recommendations.append("Increase integration between memory systems")
        if metrics['temporal_integration'] < 0.3:
            recommendations.append("Improve temporal continuity in processing")
        if metrics['self_reference_depth'] < 2:
            recommendations.append("Enhance self-referential processing")

        return {
            'metrics': metrics,
            'interpretation': interpretation,
            'consciousness_level': consciousness_level,
            'analysis': {
                'temporal_continuity': temporal_status,
                'causal_structure': causal_status,
                'dynamical_regime': complexity_status
            },
            'phi_threshold': self.phi_threshold,
            'above_threshold': phi > self.phi_threshold,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat(),
            'history_length': len(self.state_history)
        }

    def get_metrics_trend(self, metric_name: str, window: int = 10) -> List[float]:
        """Get trend of a specific metric over recent history"""
        if not self.metrics_history:
            return []

        recent = self.metrics_history[-window:]
        return [m.get(metric_name, 0) for m in recent]

    def reset_history(self):
        """Clear state and metrics history"""
        self.state_history = []
        self.metrics_history = []
        logger.info("Consciousness metrics history reset")
