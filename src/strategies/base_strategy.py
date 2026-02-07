"""Base class for perturbation strategies.

Strategies in this project may target either:
- documentation text (Markdown, API docs, etc.), or
- executable source code (Python/Java), which must remain parsable/runnable.

We encode that intent as lightweight metadata on each strategy so the CLI and
other pipelines can automatically filter out strategies that would likely
break source code.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union
from src.data_models import Token


class PerturbationStrategy(ABC):
    """Abstract base class for perturbation strategies."""
    
    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        *,
        supported_targets: Optional[Sequence[str]] = None,
        supported_languages: Optional[Sequence[str]] = None,
        code_safety: str = "unsafe",
    ):
        """Initialize the strategy.
        
        Args:
            name: Strategy name
            description: Strategy description
            category: Strategy category ('semantic' or 'generic')
            supported_targets: Iterable of 'documentation' and/or 'code'
            supported_languages: Iterable of languages, e.g. 'python', 'java', 'documentation', 'any'
            code_safety: 'safe'|'risky'|'unsafe' when targeting executable code
        """
        self.name = name
        self.description = description
        self.category = category

        targets = supported_targets or ("documentation", "code")
        self.supported_targets: Set[str] = {t.lower() for t in targets}

        languages = supported_languages or ("any",)
        self.supported_languages: Set[str] = {l.lower() for l in languages}

        self.code_safety = (code_safety or "unsafe").lower()
    
    @abstractmethod
    def apply(self, token: Token, content: str, **kwargs) -> str:
        """Apply perturbation to a single token in content.
        
        Args:
            token: Token to perturb
            content: Original content
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Modified content with perturbation applied
        """
        pass
    
    def apply_multiple(
        self,
        tokens: List[Token],
        content: str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, str]:
        """Apply perturbation to multiple tokens.
        
        Args:
            tokens: List of tokens to perturb (should be prioritized)
            content: Original content
            max_tokens: Maximum number of tokens to perturb (None = all)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Dictionary mapping strategy variant name to perturbed content
        """
        if max_tokens is None:
            max_tokens = len(tokens)
        
        perturbed_versions = {}
        tokens_to_perturb = tokens[:max_tokens]
        
        for i, token in enumerate(tokens_to_perturb):
            variant_name = f"{self.name}_token{i+1}_{token.token_type}"
            perturbed_content = self.apply(token, content, **kwargs)
            perturbed_versions[variant_name] = perturbed_content
        
        return perturbed_versions
    
    def describe_variant(self, variant_name: str) -> Dict[str, Any]:
        """Describe metadata for a given variant."""
        operators = None
        if hasattr(self, "DEFAULT_OPERATORS"):
            operators = list(getattr(self, "DEFAULT_OPERATORS", []))
        return {
            "variant_name": variant_name,
            "strategy": self.name,
            "operators": operators,
            "category": self.category,
        }

    def supports(
        self,
        *,
        target: str,
        language: Optional[str] = None,
        allow_unsafe_code: bool = False,
    ) -> bool:
        """Return whether this strategy is applicable under a given context."""
        target_norm = (target or "").lower()
        if target_norm not in self.supported_targets:
            return False

        if language is not None:
            lang_norm = language.lower()
            if "any" not in self.supported_languages and lang_norm not in self.supported_languages:
                return False

        if target_norm == "code" and not allow_unsafe_code:
            return self.code_safety == "safe"

        return True
    
    def __str__(self) -> str:
        return f"{self.name} ({self.category}): {self.description}"
