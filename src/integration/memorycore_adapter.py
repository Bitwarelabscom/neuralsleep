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
        if len(state.shape) > 1:
            # Remove batch dimension if present
            state = state.squeeze(0)

        state_list = state.tolist()

        return {
            'proficiencyModel': {
                'overallLevel': float(state_list[0]),
                'reading': float(state_list[1]),
                'writing': float(state_list[2]),
                'recognition': float(state_list[3])
            },
            'learningStyleModel': {
                'challengeTolerance': float(state_list[4]),
                'repetitionNeeds': float(state_list[5]),
                'preferredExplanationType': 'mixed',
                'optimalDifficultyCurve': []
            },
            'retentionModel': {
                'spacingEffectParameter': float(state_list[6]),
                'forgettingCurve': {
                    'timePoints': [],
                    'retentionRates': []
                },
                'optimalReviewIntervals': {}
            },
            'persistentPatterns': {
                'commonErrors': [],
                'strengths': []
            },
            'teachingModel': {
                'effectiveExplanations': {},
                'engagingExerciseTypes': [],
                'optimalSessionLength': 1800,
                'idealDifficultyProgression': []
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

    @staticmethod
    def decode_mastery_level(tensor: torch.Tensor, character_id: str) -> Dict[str, Any]:
        """
        Convert tensor back to MasteryLevel structure
        """
        if len(tensor.shape) > 1:
            tensor = tensor.squeeze(0)

        values = tensor.tolist()

        return {
            'userId': '',  # To be filled by caller
            'characterId': character_id,
            'masteryLevel': float(values[0]),
            'confidence': float(values[1]),
            'reviewCount': int(values[2] * 100),
            'successRate': float(values[3]),
            'avgRecallTime': float(values[4] * 10000),
            'timeConstant': 1000.0,
            'lastReviewed': None,
            'lastConsolidated': None
        }
