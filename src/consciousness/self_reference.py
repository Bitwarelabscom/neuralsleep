"""
Self-Referential Processing for NeuralSleep

Implements a system that models itself - a key marker of consciousness.
The self-model influences behavior, and behavior updates the self-model
in a recursive loop.

Based on:
- Higher-Order Thought (HOT) theory (Rosenthal)
- Self-Model Theory of Subjectivity (Metzinger)
- Metacognition research
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from networks.ltc import LTCNetwork

logger = logging.getLogger(__name__)


class SelfReferentialProcessor:
    """
    System that models itself through recursive processing

    Key components:
    1. Self-model: LNN that represents the system's own state
    2. Metacognitive layer: Monitors and modulates processing
    3. Self-awareness loop: Continuous self-observation
    """

    def __init__(
        self,
        observation_size: int = 512,
        self_model_size: int = 256,
        output_size: int = 128,
        max_recursion_depth: int = 5
    ):
        """
        Args:
            observation_size: Size of self-observation vector
            self_model_size: Hidden size of self-model network
            output_size: Size of self-representation output
            max_recursion_depth: Maximum recursive self-modeling levels
        """
        self.observation_size = observation_size
        self.self_model_size = self_model_size
        self.output_size = output_size
        self.max_recursion_depth = max_recursion_depth

        # Self-model: LNN that represents own state
        # Uses moderate time constants (1-100s) for self-representation
        self.self_model = LTCNetwork(
            input_size=observation_size,
            hidden_size=self_model_size,
            output_size=output_size,
            time_constant_range=(1.0, 100.0)
        )

        # Metacognitive network: monitors processing quality
        self.metacognition = nn.Sequential(
            nn.Linear(output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # confidence, uncertainty, attention, arousal
        )

        # Current self-state
        self.current_self_state = torch.zeros(self_model_size)
        self.self_representation = torch.zeros(output_size)

        # Processing history for self-observation
        self.processing_history = []
        self.max_history = 50

        # Recursion tracking
        self.current_depth = 0

        # Device
        self.device = torch.device('cpu')

    def to(self, device: torch.device):
        """Move models to device"""
        self.device = device
        self.self_model = self.self_model.to(device)
        self.metacognition = self.metacognition.to(device)
        self.current_self_state = self.current_self_state.to(device)
        self.self_representation = self.self_representation.to(device)
        return self

    def observe_processing(
        self,
        working_memory_state: Optional[torch.Tensor] = None,
        episodic_state: Optional[torch.Tensor] = None,
        semantic_state: Optional[torch.Tensor] = None,
        attention_weights: Optional[torch.Tensor] = None,
        current_output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Observe the system's own processing state

        Aggregates information about:
        - Memory system activations
        - Attention distribution
        - Output characteristics
        - Processing dynamics

        Returns:
            observation: Self-observation vector [observation_size]
        """
        observation_parts = []

        # Working memory observation
        if working_memory_state is not None:
            wm = working_memory_state.detach().flatten()
            # Take statistics of working memory
            wm_stats = torch.tensor([
                wm.mean(),
                wm.std(),
                wm.max(),
                wm.min(),
                (wm > 0).float().mean(),  # Activation ratio
            ], device=self.device)
            observation_parts.append(wm_stats)

            # Truncate/pad to fixed size
            wm_sample = wm[:64] if len(wm) >= 64 else nn.functional.pad(wm, (0, 64 - len(wm)))
            observation_parts.append(wm_sample)

        # Episodic memory observation
        if episodic_state is not None:
            ep = episodic_state.detach().flatten()
            ep_stats = torch.tensor([
                ep.mean(),
                ep.std(),
                ep.max(),
                ep.min(),
                (ep > 0).float().mean(),
            ], device=self.device)
            observation_parts.append(ep_stats)

            ep_sample = ep[:64] if len(ep) >= 64 else nn.functional.pad(ep, (0, 64 - len(ep)))
            observation_parts.append(ep_sample)

        # Semantic memory observation
        if semantic_state is not None:
            sem = semantic_state.detach().flatten()
            sem_stats = torch.tensor([
                sem.mean(),
                sem.std(),
                sem.max(),
                sem.min(),
                (sem > 0).float().mean(),
            ], device=self.device)
            observation_parts.append(sem_stats)

            sem_sample = sem[:64] if len(sem) >= 64 else nn.functional.pad(sem, (0, 64 - len(sem)))
            observation_parts.append(sem_sample)

        # Attention weights
        if attention_weights is not None:
            att = attention_weights.detach().flatten()
            # Entropy of attention distribution
            att_normalized = nn.functional.softmax(att, dim=0)
            att_entropy = -(att_normalized * torch.log(att_normalized + 1e-10)).sum()
            observation_parts.append(torch.tensor([att_entropy], device=self.device))

        # Current output characteristics
        if current_output is not None:
            out = current_output.detach().flatten()
            out_stats = torch.tensor([
                out.mean(),
                out.std(),
                out.max(),
            ], device=self.device)
            observation_parts.append(out_stats)

        # Include previous self-representation (recursive)
        observation_parts.append(self.self_representation.detach())

        # Combine all observations
        if observation_parts:
            combined = torch.cat(observation_parts)
        else:
            combined = torch.zeros(self.observation_size, device=self.device)

        # Pad/truncate to observation_size
        if combined.numel() < self.observation_size:
            observation = torch.zeros(self.observation_size, device=self.device)
            observation[:combined.numel()] = combined
        else:
            observation = combined[:self.observation_size]

        return observation

    def update_self_model(
        self,
        observation: torch.Tensor,
        timestamp: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Update the self-model based on observation

        Args:
            observation: Self-observation vector
            timestamp: Current time for LNN dynamics

        Returns:
            self_rep: Updated self-representation
            metacog: Metacognitive assessments
        """
        # Prevent infinite recursion
        if self.current_depth >= self.max_recursion_depth:
            return self.self_representation, self._default_metacognition()

        self.current_depth += 1

        try:
            # Time sequence for LNN
            t = torch.tensor([timestamp, timestamp + 0.1], device=self.device)

            # Update self-model
            self.self_model.eval()
            with torch.no_grad():
                outputs, new_state = self.self_model(
                    observation.unsqueeze(0).unsqueeze(0),
                    t,
                    initial_state=self.current_self_state.unsqueeze(0)
                )

            # Update states
            self.current_self_state = new_state.squeeze(0)
            self.self_representation = outputs[-1].squeeze()

            # Compute metacognitive assessment
            metacog_output = self.metacognition(self.self_representation)

            metacognition = {
                'confidence': torch.sigmoid(metacog_output[0]).item(),
                'uncertainty': torch.sigmoid(metacog_output[1]).item(),
                'attention_level': torch.sigmoid(metacog_output[2]).item(),
                'arousal_level': torch.sigmoid(metacog_output[3]).item()
            }

            # Store in history
            self.processing_history.append({
                'timestamp': timestamp,
                'self_representation': self.self_representation.clone(),
                'metacognition': metacognition
            })
            if len(self.processing_history) > self.max_history:
                self.processing_history = self.processing_history[-self.max_history:]

            return self.self_representation, metacognition

        finally:
            self.current_depth -= 1

    def _default_metacognition(self) -> Dict[str, float]:
        """Return default metacognitive values"""
        return {
            'confidence': 0.5,
            'uncertainty': 0.5,
            'attention_level': 0.5,
            'arousal_level': 0.5
        }

    def process_with_self_awareness(
        self,
        input_data: torch.Tensor,
        memory_states: Dict[str, torch.Tensor],
        process_fn: callable,
        timestamp: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process input while modeling own processing

        Args:
            input_data: Input to process
            memory_states: Current states of memory systems
            process_fn: Function to process input
            timestamp: Current time

        Returns:
            output: Processed output (potentially modulated)
            self_info: Self-awareness information
        """
        # 1. Normal processing
        output = process_fn(input_data)

        # 2. Observe own processing
        observation = self.observe_processing(
            working_memory_state=memory_states.get('working'),
            episodic_state=memory_states.get('episodic'),
            semantic_state=memory_states.get('semantic'),
            current_output=output
        )

        # 3. Update self-model
        self_rep, metacog = self.update_self_model(observation, timestamp)

        # 4. Modulate output based on self-model
        modulated_output = self._modulate_output(output, metacog)

        self_info = {
            'self_representation': self_rep,
            'metacognition': metacog,
            'recursion_depth': self.get_self_reference_depth(),
            'processing_history_length': len(self.processing_history)
        }

        return modulated_output, self_info

    def _modulate_output(
        self,
        output: torch.Tensor,
        metacog: Dict[str, float]
    ) -> torch.Tensor:
        """
        Modulate output based on metacognitive assessment

        Low confidence -> attenuate output
        High uncertainty -> add noise/exploration
        """
        confidence = metacog['confidence']
        uncertainty = metacog['uncertainty']

        # Scale output by confidence
        modulated = output * confidence

        # Add exploration noise based on uncertainty
        if uncertainty > 0.7:
            noise_scale = (uncertainty - 0.7) * 0.1
            noise = torch.randn_like(output) * noise_scale
            modulated = modulated + noise

        return modulated

    def get_self_reference_depth(self) -> int:
        """
        Measure depth of self-referential processing

        Analyzes how many levels of self-modeling are active
        """
        if len(self.processing_history) < 2:
            return 1

        # Compute autocorrelation of self-representations
        recent_reps = [h['self_representation'].cpu().numpy()
                      for h in self.processing_history[-10:]]

        if len(recent_reps) < 2:
            return 1

        reps_array = np.array([r.flatten() for r in recent_reps])

        # Count significant autocorrelation levels
        depth = 1
        for lag in range(1, min(5, len(reps_array))):
            try:
                corr = np.corrcoef(
                    reps_array[:-lag].flatten(),
                    reps_array[lag:].flatten()
                )[0, 1]
                if not np.isnan(corr) and abs(corr) > 0.3:
                    depth += 1
            except Exception:
                continue

        return depth

    def get_self_state(self) -> Dict[str, Any]:
        """Get current self-model state"""
        return {
            'self_representation': self.self_representation.tolist(),
            'self_state_norm': self.current_self_state.norm().item(),
            'recursion_depth': self.get_self_reference_depth(),
            'history_length': len(self.processing_history),
            'time_constants': self.self_model.get_time_constants().mean().item()
        }

    def reset(self):
        """Reset self-model state"""
        self.current_self_state = torch.zeros(self.self_model_size, device=self.device)
        self.self_representation = torch.zeros(self.output_size, device=self.device)
        self.processing_history = []
        self.current_depth = 0
        logger.info("Self-referential processor reset")
