"""
Integrated Information Theory (IIT) - Phi (Φ) Computation

Implements measures of integrated information based on Tononi's IIT.
Φ measures how much information the system as a whole contains beyond
what its parts contain independently.

Reference:
- Tononi, G. (2008). Consciousness as Integrated Information
- Oizumi, M., Albantakis, L., & Tononi, G. (2014). From the phenomenology
  to the mechanisms of consciousness: Integrated Information Theory 3.0
"""

import torch
import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class PhiComputation:
    """
    Compute Φ (Phi) - Integrated Information measure

    A high Φ indicates that the system contains more information
    as a whole than the sum of its parts - a potential marker
    of consciousness.
    """

    def __init__(self, max_partition_size: int = 16):
        """
        Args:
            max_partition_size: Maximum number of dimensions to consider
                               for full partition analysis (exponential complexity)
        """
        self.max_partition_size = max_partition_size

    def compute_phi(
        self,
        system_state: np.ndarray,
        method: str = 'approximation'
    ) -> float:
        """
        Compute integrated information (Φ)

        Args:
            system_state: System state matrix [time_steps, dimensions]
            method: 'exact' for full IIT (exponential),
                   'approximation' for practical computation

        Returns:
            phi: Integrated information value (higher = more integrated)
        """
        if system_state.ndim == 1:
            system_state = system_state.reshape(1, -1)

        n_dims = system_state.shape[1]

        # For large systems, use approximation
        if method == 'exact' and n_dims <= self.max_partition_size:
            return self._compute_phi_exact(system_state)
        else:
            return self._compute_phi_approximation(system_state)

    def _compute_phi_exact(self, system_state: np.ndarray) -> float:
        """
        Exact Φ computation (exponential complexity)

        Computes minimum information partition (MIP) across all
        possible bipartitions of the system.
        """
        n_dims = system_state.shape[1]

        # Full system effective information
        ei_whole = self._effective_information(system_state)

        if ei_whole == 0:
            return 0.0

        # Generate all possible bipartitions
        min_phi = float('inf')

        for partition in self._generate_bipartitions(n_dims):
            part_a, part_b = partition

            # Skip empty partitions
            if len(part_a) == 0 or len(part_b) == 0:
                continue

            # Effective information of parts
            state_a = system_state[:, list(part_a)]
            state_b = system_state[:, list(part_b)]

            ei_a = self._effective_information(state_a)
            ei_b = self._effective_information(state_b)

            # Information lost at the cut
            ei_parts = ei_a + ei_b

            # Normalized by cut size (number of connections severed)
            cut_size = len(part_a) * len(part_b)
            if cut_size > 0:
                phi_partition = (ei_whole - ei_parts) / cut_size
            else:
                phi_partition = 0

            min_phi = min(min_phi, phi_partition)

        return max(0, min_phi)

    def _compute_phi_approximation(self, system_state: np.ndarray) -> float:
        """
        Approximation of Φ for large systems

        Uses several heuristics:
        1. SVD-based dimensionality analysis
        2. Mutual information estimation
        3. Complexity measures
        """
        # 1. Effective dimensionality (SVD-based)
        effective_dims = self._effective_dimensionality(system_state)

        # 2. Integration measure (mutual information proxy)
        integration = self._compute_integration(system_state)

        # 3. Complexity measure
        complexity = self._compute_complexity(system_state)

        # Combine measures (weighted average)
        # Higher effective dims + high integration + moderate complexity = high Φ
        n_dims = system_state.shape[1]
        dim_ratio = effective_dims / n_dims if n_dims > 0 else 0

        phi_approx = 0.4 * dim_ratio + 0.4 * integration + 0.2 * complexity

        return float(phi_approx)

    def _effective_information(self, state: np.ndarray) -> float:
        """
        Compute effective information of a state

        Based on the cause-effect repertoire analysis from IIT 3.0.
        Simplified to use entropy of the state distribution.
        """
        if state.size == 0:
            return 0.0

        # Compute covariance matrix
        if state.shape[0] < 2:
            return float(state.shape[1])  # Default to dimensionality

        try:
            cov = np.cov(state.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            elif cov.ndim == 1:
                cov = np.diag(cov)

            # Singular value decomposition
            _, s, _ = svd(cov)

            # Effective information ~ sum of significant singular values
            threshold = 0.01 * s[0] if len(s) > 0 and s[0] > 0 else 1e-10
            effective_dimensions = np.sum(s > threshold)

            return float(effective_dimensions)
        except Exception as e:
            logger.warning(f"Error computing effective information: {e}")
            return float(state.shape[1])

    def _effective_dimensionality(self, state: np.ndarray) -> float:
        """
        Compute effective dimensionality using participation ratio

        PR = (Σλ_i)² / Σλ_i²

        Higher values indicate more dimensions are actively participating.
        """
        if state.shape[0] < 2:
            return float(state.shape[1])

        try:
            cov = np.cov(state.T)
            if cov.ndim == 0:
                return 1.0

            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

            sum_eigenvalues = np.sum(eigenvalues)
            sum_squared = np.sum(eigenvalues ** 2)

            if sum_squared > 0:
                participation_ratio = (sum_eigenvalues ** 2) / sum_squared
                return float(participation_ratio)
            return 1.0
        except Exception:
            return 1.0

    def _compute_integration(self, state: np.ndarray) -> float:
        """
        Estimate integration level through mutual information proxy

        Compares total entropy with sum of marginal entropies.
        """
        if state.shape[0] < 2 or state.shape[1] < 2:
            return 0.0

        try:
            # Total entropy (joint)
            cov = np.cov(state.T)
            if cov.ndim == 0:
                return 0.0

            det_cov = np.linalg.det(cov)
            if det_cov <= 0:
                det_cov = 1e-10

            joint_entropy = 0.5 * np.log(det_cov + 1e-10)

            # Sum of marginal entropies
            marginal_entropy = 0
            for i in range(state.shape[1]):
                var = np.var(state[:, i])
                if var > 0:
                    marginal_entropy += 0.5 * np.log(2 * np.pi * np.e * var)

            # Multi-information (integration)
            multi_info = marginal_entropy - joint_entropy

            # Normalize to [0, 1]
            max_integration = state.shape[1] * 0.5 * np.log(2 * np.pi * np.e)
            if max_integration > 0:
                integration = np.clip(multi_info / max_integration, 0, 1)
            else:
                integration = 0

            return float(integration)
        except Exception:
            return 0.0

    def _compute_complexity(self, state: np.ndarray) -> float:
        """
        Compute neural complexity (Tononi-Sporns-Edelman complexity)

        Balances segregation and integration - maximized when
        the system has both local specialization and global integration.
        """
        if state.shape[0] < 2 or state.shape[1] < 2:
            return 0.0

        n_dims = state.shape[1]

        try:
            cov = np.cov(state.T)
            if cov.ndim == 0:
                return 0.0

            # Compute mutual information for subsets of increasing size
            complexity = 0

            for k in range(1, min(n_dims, 8)):  # Limit subset size
                # Sample subsets of size k
                subset_mi = []

                for subset in combinations(range(n_dims), k):
                    subset = list(subset)
                    subcov = cov[np.ix_(subset, subset)]

                    det = np.linalg.det(subcov)
                    if det > 0:
                        subset_entropy = 0.5 * np.log(det)
                        subset_mi.append(subset_entropy)

                if subset_mi:
                    avg_mi = np.mean(subset_mi)
                    complexity += avg_mi * (k / n_dims)

            # Normalize
            complexity = np.clip(complexity / n_dims, 0, 1)
            return float(complexity)
        except Exception:
            return 0.0

    def _generate_bipartitions(self, n: int) -> List[Tuple[frozenset, frozenset]]:
        """Generate all unique bipartitions of n elements"""
        partitions = []
        elements = list(range(n))

        # Generate partitions using binary encoding
        for i in range(1, 2 ** (n - 1)):  # Only need half due to symmetry
            part_a = frozenset(j for j in range(n) if (i >> j) & 1)
            part_b = frozenset(j for j in range(n) if not ((i >> j) & 1))
            partitions.append((part_a, part_b))

        return partitions

    def compute_phi_over_time(
        self,
        state_history: List[np.ndarray],
        window_size: int = 10
    ) -> List[float]:
        """
        Compute Φ over a sliding window of states

        Args:
            state_history: List of state vectors
            window_size: Number of states per window

        Returns:
            phi_values: Φ value for each window
        """
        phi_values = []

        for i in range(len(state_history) - window_size + 1):
            window = np.array(state_history[i:i + window_size])
            phi = self.compute_phi(window)
            phi_values.append(phi)

        return phi_values


def compute_phi_from_tensor(tensor: torch.Tensor, method: str = 'approximation') -> float:
    """
    Convenience function to compute Φ from a PyTorch tensor
    """
    phi_computer = PhiComputation()
    np_array = tensor.detach().cpu().numpy()
    return phi_computer.compute_phi(np_array, method=method)
