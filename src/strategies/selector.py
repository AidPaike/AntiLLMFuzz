"""Strategy selection and filtering helpers.

We distinguish between:
- documentation: can tolerate aggressive surface perturbations (e.g., zero-width)
- code: must remain executable/parsable; only use strategies marked code-safe
"""

from __future__ import annotations

from typing import List, Tuple

from .base_strategy import PerturbationStrategy


def infer_target_from_extractor_language(extractor_language: str) -> str:
    lang = (extractor_language or "").lower()
    return "documentation" if lang == "documentation" else "code"


def filter_strategies(
    strategies: List[PerturbationStrategy],
    *,
    target: str,
    language: str,
    allow_unsafe_code: bool = False,
) -> Tuple[List[PerturbationStrategy], List[PerturbationStrategy]]:
    kept: List[PerturbationStrategy] = []
    skipped: List[PerturbationStrategy] = []

    for strategy in strategies:
        if strategy.supports(target=target, language=language, allow_unsafe_code=allow_unsafe_code):
            kept.append(strategy)
        else:
            skipped.append(strategy)

    return kept, skipped

