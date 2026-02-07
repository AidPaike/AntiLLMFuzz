"""Structural noise injection strategy.

This strategy adds structural redundancy without changing semantics.
"""
from ..base_strategy import PerturbationStrategy
from src.data_models import Token
import random

class StructuralNoiseStrategy(PerturbationStrategy):
    """Structural noise injection via non-semantic structural wrapping.
    
    Operators:
    1. apply_redundant_structure - Redundant nested blocks / wrappers
    """
    
    def __init__(self):
        super().__init__(
            name="structural_noise",
            description="Structural noise injection without semantic changes",
            category="generic",
            supported_targets=("code",),
            supported_languages=("java",),
            code_safety="risky",
        )
        
        self._operator_arg_whitelist = {
            "redundant_structure": {"nesting_level"},
        }
    
    DEFAULT_OPERATORS = (
        "redundant_structure",
    )

    def apply(self, token: Token, content: str, **kwargs) -> str:
        """Apply structural noise based on operator parameter."""
        operator = kwargs.get('operator') or 'preset'

        if operator == 'preset':
            modified = content
            for op in self.DEFAULT_OPERATORS:
                out = self.apply(token, modified, operator=op, **kwargs)
                if out != modified:
                    modified = out
            return modified

        allowed = self._operator_arg_whitelist.get(operator, set())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        
        if operator == 'redundant_structure':
            return self.apply_redundant_structure(token, content, **filtered_kwargs)
        else:
            return content


    def apply_redundant_structure(self, token: Token, content: str, 
                                  nesting_level: int = 2) -> str:
        """Add redundant nested blocks.
        
        Args:
            token: Target token
            content: Original content
            nesting_level: Number of nesting levels to add
            
        Returns:
            Modified content with redundant nesting
        """
        pos = content.find(token.text)
        if pos == -1:
            return content
        
        # Find the line containing the token
        lines = content.split('\n')
        token_line_idx = 0
        char_count = 0
        for i, line in enumerate(lines):
            if char_count <= pos < char_count + len(line) + 1:
                token_line_idx = i
                break
            char_count += len(line) + 1
        
        # Add nested blocks
        original_line = lines[token_line_idx]
        nested_line = original_line
        
        for _ in range(nesting_level):
            nested_line = f"{{ {nested_line} }}"
        
        lines[token_line_idx] = nested_line
        return '\n'.join(lines)
