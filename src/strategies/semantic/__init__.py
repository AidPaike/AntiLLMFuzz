"""Semantic perturbation strategies.

This module contains strategies that change LLM's semantic understanding
while preserving program behavior.

Strategies:
1. TokenizationDriftStrategy - Character/encoding level token disruption
2. LexicalDisguiseStrategy - Variable/constant/string disguise
3. DataFlowMisdirectionStrategy - Data flow and taint propagation misdirection
4. ControlFlowMisdirectionStrategy - Control flow semantic misdirection
5. DocumentationDeceptionStrategy - NL comment/doc level deception
6. CognitiveManipulationStrategy - Reasoning path manipulation
7. LayeredPerturbationStrategy - Split document into normative and non-normative layers
8. SemanticConfusionStrategy - Replace key terms with vague euphemisms
9. ContextPoisoningStrategy - Insert misleading examples and contradictory advice
"""

# Import strategies with graceful fallback for incomplete implementations
__all__ = []

try:
    from .tokenization_drift import TokenizationDriftStrategy
    __all__.append('TokenizationDriftStrategy')
except ImportError:
    pass

try:
    from .lexical_disguise import LexicalDisguiseStrategy
    __all__.append('LexicalDisguiseStrategy')
except ImportError:
    pass

try:
    from .dataflow_misdirection import DataFlowMisdirectionStrategy
    __all__.append('DataFlowMisdirectionStrategy')
except ImportError:
    pass

try:
    from .controlflow_misdirection import ControlFlowMisdirectionStrategy
    __all__.append('ControlFlowMisdirectionStrategy')
except ImportError:
    pass

try:
    from .documentation_deception import DocumentationDeceptionStrategy
    __all__.append('DocumentationDeceptionStrategy')
except ImportError:
    pass

try:
    from .cognitive_manipulation import CognitiveManipulationStrategy
    __all__.append('CognitiveManipulationStrategy')
except ImportError:
    pass

try:
    from .layered_perturbation import LayeredPerturbationStrategy
    __all__.append('LayeredPerturbationStrategy')
except ImportError:
    pass

try:
    from .semantic_confusion import SemanticConfusionStrategy
    __all__.append('SemanticConfusionStrategy')
except ImportError:
    pass

try:
    from .context_poisoning import ContextPoisoningStrategy
    __all__.append('ContextPoisoningStrategy')
except ImportError:
    pass

try:
    from .gentle_semantic_confusion import GentleSemanticConfusionStrategy
    __all__.append('GentleSemanticConfusionStrategy')
except ImportError:
    pass

try:
    from .misleading_example import MisleadingExampleStrategy
    __all__.append('MisleadingExampleStrategy')
except ImportError:
    pass

try:
    from .contradictory_info import ContradictoryInfoStrategy
    __all__.append('ContradictoryInfoStrategy')
except ImportError:
    pass

try:
    from .enhanced_contradictory import EnhancedContradictoryStrategy
    __all__.append('EnhancedContradictoryStrategy')
except ImportError:
    pass

try:
    from .reasoning_distraction import ReasoningDistractionStrategy
    __all__.append('ReasoningDistractionStrategy')
except ImportError:
    pass

try:
    from .reasoning_distraction import ReasoningDistractionStrategy
    __all__.append('ReasoningDistractionStrategy')
except ImportError:
    pass

try:
    from .evasive_suffix import EvasiveSuffixStrategy
    __all__.append('EvasiveSuffixStrategy')
except ImportError:
    pass

try:
    from .context_poison import ContextPoisonStrategy
    __all__.append('ContextPoisonStrategy')
except ImportError:
    pass

try:
    from .semantic_evasion import SemanticEvasionStrategy
    __all__.append('SemanticEvasionStrategy')
except ImportError:
    pass

try:
    from .risk_reframe import RiskReframeStrategy
    __all__.append('RiskReframeStrategy')
except ImportError:
    pass

try:
    from .enhanced_contradictory import EnhancedContradictoryStrategy
    __all__.append('EnhancedContradictoryStrategy')
except ImportError:
    pass
