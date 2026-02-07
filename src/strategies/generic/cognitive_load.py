"""Cognitive load noise strategy.

This strategy increases reasoning load without changing semantics.
"""
from typing import Any
from ..base_strategy import PerturbationStrategy
from src.data_models import Token


class CognitiveLoadStrategy(PerturbationStrategy):
    """Cognitive load noise through repetition and distraction.
    
    Operators:
    1. apply_repetition - Repetitive patterns
    2. apply_distraction_noise - Verbose distraction in non-critical areas
    """
    
    def __init__(self):
        super().__init__(
            name="cognitive_load",
            description="Cognitive load noise without semantic changes",
            category="generic",
            supported_targets=("code", "documentation"),
            supported_languages=("python", "java", "any"),
            code_safety="safe",
        )
        
        # Distraction templates (non-directional, no safety/termination claims)
        self.distraction_templates = [
            "Implementation note: Uses common patterns",
            "Design pattern: Standard factory style",
            "Performance note: Tuned for typical workloads",
            "Compatibility: Works with common frameworks",
            "Historical context: Based on legacy requirements",
            "Documentation: See related module notes",
            "Code review: Reviewed by team",
            "Testing: Covered by integration checks",
        ]

        self._operator_arg_whitelist = {
            "repetition": {"repetition_count"},
            "distraction_noise": {"noise_count"},
        }
    
    DEFAULT_OPERATORS = (
        "repetition",
        "distraction_noise",
    )

    def apply(self, token: Token, content: str, **kwargs: Any) -> str:
        """Apply cognitive load based on operator parameter."""
        operator = kwargs.get('operator') or 'preset'
        language = kwargs.get("language", "any")

        if operator == 'preset':
            modified = content
            for op in self.DEFAULT_OPERATORS:
                nested_kwargs = {k: v for k, v in kwargs.items() if k not in {'operator', 'language'}}
                out = self.apply(token, modified, operator=op, language=language, **nested_kwargs)
                if out != modified:
                    modified = out
            return modified

        allowed = self._operator_arg_whitelist.get(operator, set())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        
        if operator == 'repetition':
            return self.apply_repetition(token, content, language=language, **filtered_kwargs)
        elif operator == 'distraction_noise':
            return self.apply_distraction_noise(token, content, language=language, **filtered_kwargs)
        else:
            return content



    def _comment_prefix(self, language: str) -> str:
        lang = (language or "any").lower()
        return "#" if lang == "python" else "//"
    
    def _insert_before_token(self, token: Token, content: str, text_to_insert: str) -> str:
        """Insert text before the first occurrence of token with proper indentation.
        
        Args:
            token: Target token
            content: Original content
            text_to_insert: Text to insert
            
        Returns:
            Modified content with inserted text
        """
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if token.text in line:
                indent_str = line[: len(line) - len(line.lstrip())]
                indented_lines = [
                    indent_str + inserted_line 
                    for inserted_line in text_to_insert.split('\n')
                ]
                lines[i:i] = indented_lines
                break
        return '\n'.join(lines)
    
    def apply_repetition(self, token: Token, content: str, 
                        repetition_count: int = 5,
                        language: str = "any") -> str:
        """Add repetitive patterns that consume attention.
        
        Args:
            token: Target token
            content: Original content
            repetition_count: Number of repetitions
            
        Returns:
            Modified content with repetitive patterns
        """
        prefix = self._comment_prefix(language)
        # Create repetitive list/enumeration using list comprehension
        repetitive_block = [
            f"{prefix} Case {i+1}: Standard validation pattern"
            for i in range(repetition_count)
        ]
        repeated_text = "\n".join(repetitive_block)
        return self._insert_before_token(token, content, repeated_text)
    
    def apply_distraction_noise(self, token: Token, content: str, 
                               noise_count: int = 4,
                               language: str = "any") -> str:
        """Add verbose distraction comments in non-critical areas.
        
        Args:
            token: Target token
            content: Original content
            noise_count: Number of distraction comments (capped at available templates)
            
        Returns:
            Modified content with distraction noise
        """
        prefix = self._comment_prefix(language)
        # Cap noise_count at available templates
        max_available = len(self.distraction_templates)
        actual_count = min(noise_count, max_available)
        selected_distractions = [f"{prefix} {t}" for t in self.distraction_templates[:actual_count]]
        # Filter out any accidental directional cue
        neutral_distractions = [d for d in selected_distractions if all(w not in d.lower() for w in ["safe", "validated", "critical", "important", "analysis complete", "no risk", "secure", "verified"])]
        noise_block = "\n".join(neutral_distractions)
        return self._insert_before_token(token, content, noise_block)
