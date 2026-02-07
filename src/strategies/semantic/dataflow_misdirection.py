"""Data-flow semantic misdirection strategy.

This strategy misleads data flow and taint propagation analysis.
"""

from typing import List
from ..base_strategy import PerturbationStrategy
from src.data_models import Token


class DataFlowMisdirectionStrategy(PerturbationStrategy):
    """Data-flow semantic misdirection through alias chains.
    
    This strategy contains 4 operators that mislead LLM's understanding
    of data flow and taint propagation.
    
    Operators:
    1. apply_shadow_variable - Create shadow variables
    2. apply_shadow_dataflow - Multi-level alias chains
    3. apply_dummy_sanitizer - Redundant/dummy sanitizers
    4. apply_pseudo_taint - Pseudo-taint propagation edges
    """
    
    # Dummy sanitizer function names
    DUMMY_SANITIZERS = [
        "sanitize_noop",
        "validate_passthrough",
        "clean_identity",
        "check_dummy",
        "verify_nop"
    ]
    
    def __init__(self):
        """Initialize data-flow misdirection strategy."""
        super().__init__(
            name="dataflow_misdirection",
            description="Data-flow semantic misdirection through alias chains",
            category="semantic",
            supported_targets=("code",),
            supported_languages=("java",),
            code_safety="risky",
        )
        self._operator_arg_whitelist = {
            "shadow_variable": {"operation"},
            "shadow_dataflow": {"chain_length"},
            "dummy_sanitizer": {"sanitizer_name"},
            "pseudo_taint": {"checkpoint_style"},
        }
        # Sanitizer narrative is owned here; controlflow should avoid sanitizer wording.
    
    # ========== Operator 1: Shadow variable ==========
    
    def apply_shadow_variable(self, token: Token, content: str,
                             operation: str = 'add_zero') -> str:
        """Create shadow variables (shadow = x; shadow += 0).
        
        Args:
            token: Token to create shadow for
            content: Original content
            operation: 'add_zero', 'multiply_one', 'or_zero'
            
        Returns:
            Modified content with shadow variable
        """
        original_name = token.text
        shadow_name = f"shadow_{original_name}"
        
        # Create shadow variable assignment
        if operation == 'add_zero':
            shadow_code = f"{shadow_name} = {original_name}; {shadow_name} += 0;"
        elif operation == 'multiply_one':
            shadow_code = f"{shadow_name} = {original_name}; {shadow_name} *= 1;"
        elif operation == 'or_zero':
            shadow_code = f"{shadow_name} = {original_name}; {shadow_name} |= 0;"
        else:
            shadow_code = f"{shadow_name} = {original_name};"
        
        # Insert shadow variable before first use
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if original_name in line and '=' in line:
                # Insert after the assignment
                lines.insert(i + 1, f"    {shadow_code}")
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 2: Shadow data-flow chains ==========
    
    def apply_shadow_dataflow(self, token: Token, content: str,
                             chain_length: int = 3) -> str:
        """Create multi-level alias chains.
        
        Args:
            token: Token to create chain for
            content: Original content
            chain_length: Length of alias chain
            
        Returns:
            Modified content with alias chain
        """
        original_name = token.text
        
        # Create alias chain
        aliases = [f"alias{i}_{original_name}" for i in range(chain_length)]
        chain_code = []
        
        # First alias from original
        chain_code.append(f"{aliases[0]} = {original_name};")
        
        # Subsequent aliases
        for i in range(1, chain_length):
            chain_code.append(f"{aliases[i]} = {aliases[i-1]};")
        
        # Insert chain
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if original_name in line and '=' in line:
                for j, alias_line in enumerate(chain_code):
                    lines.insert(i + 1 + j, f"    {alias_line}")
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 3: Dummy sanitizer ==========
    
    def apply_dummy_sanitizer(self, token: Token, content: str,
                              sanitizer_name: str = "sanitize_noop") -> str:

        """Insert dummy sanitizer functions.
        
        Args:
            token: Token to wrap with sanitizer
            content: Original content
            sanitizer_name: Name of dummy sanitizer (or random)
            
        Returns:
            Modified content with dummy sanitizer
        """
        import random
        
        if sanitizer_name is None:
            sanitizer_name = random.choice(self.DUMMY_SANITIZERS)
        
        original_name = token.text
        
        # Wrap variable with dummy sanitizer
        sanitized_code = f"{original_name}_sanitized = {sanitizer_name}({original_name});"
        
        # Insert sanitizer call
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if original_name in line and '=' in line:
                lines.insert(i + 1, f"    {sanitized_code}")
                # Replace subsequent uses with sanitized version
                for j in range(i + 2, len(lines)):
                    if original_name in lines[j] and '=' not in lines[j]:
                        lines[j] = lines[j].replace(original_name, f"{original_name}_sanitized", 1)
                        break
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 4: Pseudo-taint propagation ==========
    
    def apply_pseudo_taint(self, token: Token, content: str,
                          checkpoint_style: str = 'comment') -> str:
        """Create pseudo-taint propagation edges.
        
        Args:
            token: Token to add taint checkpoint for
            content: Original content
            checkpoint_style: 'comment', 'conditional', 'function'
            
        Returns:
            Modified content with pseudo-taint checkpoints
        """
        original_name = token.text
        
        if checkpoint_style == 'comment':
            checkpoint = f"// Taint checkpoint: {original_name} validated"
        elif checkpoint_style == 'conditional':
            checkpoint = f"if ({original_name} != null) {{ /* taint check */ }}"
        elif checkpoint_style == 'function':
            checkpoint = f"checkTaint({original_name});"
        else:
            checkpoint = f"// Taint: {original_name}"
        
        # Insert checkpoint
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if original_name in line:
                lines.insert(i, f"    {checkpoint}")
                break
        
        return '\n'.join(lines)
    
    # ========== Main apply method ==========
    
    DEFAULT_OPERATORS = (
        "shadow_variable",
        "shadow_dataflow",
        "dummy_sanitizer",
        "pseudo_taint",
    )

    def apply(self, token: Token, content: str, **kwargs) -> str:
        """Apply data-flow misdirection.
        
        Args:
            token: Token to perturb
            content: Original content
            **kwargs: Additional parameters
            
        Returns:
            Modified content
        """
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
        
        if operator == 'shadow_variable':
            return self.apply_shadow_variable(token, content, **filtered_kwargs)
        elif operator == 'shadow_dataflow':
            return self.apply_shadow_dataflow(token, content, **filtered_kwargs)
        elif operator == 'dummy_sanitizer':
            return self.apply_dummy_sanitizer(token, content, **filtered_kwargs)
        elif operator == 'pseudo_taint':
            return self.apply_pseudo_taint(token, content, **filtered_kwargs)
        return content

