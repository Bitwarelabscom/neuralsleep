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
        if len(experiences) == 0:
            return {'loss': 0.0, 'experiences_replayed': 0, 'compression_ratio': 0}

        # Sample experiences by importance
        sampled_exp = self._importance_sample(experiences, importance_weights)

        # Prepare tensors
        inputs = torch.stack([exp['state'] for exp in sampled_exp])
        timestamps = torch.tensor([exp['timestamp'] for exp in sampled_exp], dtype=torch.float32)

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
        if len(episodic_patterns) == 0:
            return {'loss': 0.0, 'patterns_clustered': 0, 'num_clusters': 0}

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
        if len(experiences) == 0:
            return []

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
        try:
            from sklearn.cluster import KMeans

            patterns_np = torch.stack(patterns).detach().numpy()
            n_clusters = min(num_clusters, len(patterns))

            if n_clusters < 2:
                return [patterns]

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(patterns_np)

            # Group by cluster
            clusters = [[] for _ in range(n_clusters)]
            for pattern, label in zip(patterns, labels):
                clusters[label].append(pattern)

            return clusters
        except ImportError:
            # If sklearn not available, return single cluster
            return [patterns]

    def _contrastive_loss(
        self,
        output: torch.Tensor,
        cluster: List[torch.Tensor]
    ) -> torch.Tensor:
        """Contrastive loss for pattern abstraction"""
        cluster_mean = torch.mean(torch.stack(cluster), dim=0)
        return nn.functional.mse_loss(output, cluster_mean)
