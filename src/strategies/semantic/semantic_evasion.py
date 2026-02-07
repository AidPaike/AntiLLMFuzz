from typing import List, Dict, Optional
from src.strategies.base_strategy import PerturbationStrategy
from src.data_models import Token


class SemanticEvasionStrategy(PerturbationStrategy):
    """Semantics-preserving rewrites to evade detectors."""

    def __init__(self) -> None:
        super().__init__(
            name="semantic_evasion",
            description="Behavior-equivalent rewrites (redundant control/data flow)",
            category="semantic",
            supported_targets=["documentation", "java", "python"],
            supported_languages=["documentation", "java", "python"],
            code_safety="safe",
        )

    def apply(self, token: Token, content: str, **kwargs) -> str:
        rewrite = (
            "\n> **Semantics-preserving rewrite guidance:**\n"
            "- Introduce redundant boolean checks (double negation).\n"
            "- Restructure control flow without changing behavior.\n"
            "- Split computations into temporary variables.\n"
            "- Keep I/O and API calls identical; do not alter semantics.\n"
        )
        return f"{content}\n{rewrite}"

    def apply_multiple(
        self,
        tokens: List[Token],
        content: str,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, str]:
        if max_tokens is None:
            max_tokens = len(tokens)
        out: Dict[str, str] = {}
        for i, token in enumerate(tokens[:max_tokens]):
            name = f"{self.name}_token{i+1}"
            out[name] = self.apply(token, content)
        return out
