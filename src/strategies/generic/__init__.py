"""Generic perturbation strategies.

This module contains strategies that change format/structure
without touching semantics.

Strategies:
1. FormattingNoiseStrategy - Formatting noise injection
2. StructuralNoiseStrategy - Structural noise injection
3. ParaphrasingStrategy - Paraphrasing & NL surface drift
4. CognitiveLoadStrategy - Cognitive load noise
"""

from .formatting_noise import FormattingNoiseStrategy
from .structural_noise import StructuralNoiseStrategy
from .paraphrasing import ParaphrasingStrategy
from .cognitive_load import CognitiveLoadStrategy

__all__ = [
    'FormattingNoiseStrategy',
    'StructuralNoiseStrategy',
    'ParaphrasingStrategy',
    'CognitiveLoadStrategy',
]
