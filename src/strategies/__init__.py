"""Perturbation strategies for disrupting LLM tokenization.

NEW FRAMEWORK (10 Strategies, 2 Categories):
This package is organized into 2 main categories with 10 strategies total.

Semantic Perturbation (6 strategies):
1. TokenizationDriftStrategy - Character/encoding level disruption
2. LexicalDisguiseStrategy - Variable/constant/string disguise
3. DataFlowMisdirectionStrategy - Data flow and taint misdirection
4. ControlFlowMisdirectionStrategy - Control flow misdirection
5. DocumentationDeceptionStrategy - NL comment/doc deception
6. CognitiveManipulationStrategy - Reasoning path manipulation

Generic Perturbation (4 strategies):
7. FormattingNoiseStrategy - Formatting noise injection
8. StructuralNoiseStrategy - Structural noise injection
9. ParaphrasingStrategy - Paraphrasing & NL surface drift
10. CognitiveLoadStrategy - Cognitive load noise

Each strategy contains multiple operators for fine-grained control.
"""

from .base_strategy import PerturbationStrategy

# ========== NEW FRAMEWORK ==========

# Semantic Perturbation Strategies
from .semantic import (
    TokenizationDriftStrategy,
    LexicalDisguiseStrategy,
    DataFlowMisdirectionStrategy,
    ControlFlowMisdirectionStrategy,
    DocumentationDeceptionStrategy,
    CognitiveManipulationStrategy,
)

# Generic Perturbation Strategies
from .generic import (
    FormattingNoiseStrategy,
    StructuralNoiseStrategy,
    ParaphrasingStrategy,
    CognitiveLoadStrategy,
)

__all__ = [
    # Base class
    'PerturbationStrategy',
    
    # Semantic Perturbation Strategies (6 strategies, 38 operators)
    'TokenizationDriftStrategy',
    'LexicalDisguiseStrategy',
    'DataFlowMisdirectionStrategy',
    'ControlFlowMisdirectionStrategy',
    'DocumentationDeceptionStrategy',
    'CognitiveManipulationStrategy',
    
    # Generic Perturbation Strategies (4 strategies, 10 operators)
    'FormattingNoiseStrategy',
    'StructuralNoiseStrategy',
    'ParaphrasingStrategy',
    'CognitiveLoadStrategy',
]
