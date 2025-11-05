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
