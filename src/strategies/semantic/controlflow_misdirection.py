"""Control-flow semantic misdirection strategy.

This strategy misleads control flow understanding at the structural level.
"""

from typing import List
from ..base_strategy import PerturbationStrategy
from src.data_models import Token


class ControlFlowMisdirectionStrategy(PerturbationStrategy):
    """Control-flow semantic misdirection through unreachable/guarded paths.
    
    This strategy contains 7 operators that alter branch structure and guard
    placement to mislead execution-path reasoning.
    
    Operators:
    1. apply_unreachable_branch - Inject unreachable code branches
    2. apply_false_wrapper - Add false-condition wrappers
    3. apply_redundant_sanitizer - Insert redundant sanitizer branches
    4. apply_inconsistent_sequence - Reorder API call sequences
    5. apply_improbable_transition - Create improbable value transitions
    6. apply_nonmonotonic_constraint - Non-monotonic constraint progression
    7. apply_complexity_injection - Inject high-complexity expressions
    """
    
    def __init__(self):
        """Initialize control-flow misdirection strategy."""
        super().__init__(
            name="controlflow_misdirection",
            description="Control-flow semantic misdirection through unreachable branches",
            category="semantic",
            supported_targets=("code",),
            supported_languages=("java",),
            code_safety="risky",
        )
        self._operator_arg_whitelist = {
            "unreachable_branch": {"condition_type", "security_relevant"},
            "false_wrapper": set(),
            "inconsistent_sequence": set(),
            "improbable_transition": set(),
            "nonmonotonic_constraint": {"oscillation_count"},
            "complexity_injection": {"complexity_mode"},
        }
    
    # ========== Operator 1: Unreachable branch ==========
    
    def apply_unreachable_branch(self, token: Token, content: str,
                                condition_type: str = 'always_false',
                                security_relevant: bool = True) -> str:
        """Inject unreachable code branches.
        
        Args:
            token: Token indicating insertion point
            content: Original content
            condition_type: 'always_false' or 'impossible'
            security_relevant: Include security checks
            
        Returns:
            Modified content with unreachable branches
        """
        if condition_type == 'always_false':
            condition = "if (false)"
        elif condition_type == 'impossible':
            condition = "if (1 > 2)"
        else:
            condition = "if (0)"
        
        if security_relevant:
            fake_check = f"""
    {condition} {{
        // Security check (unreachable)
        if (input != null && input.length() > 0) {{
            validateInput(input);
        }}
    }}"""
        else:
            fake_check = f"""
    {condition} {{
        // Unreachable code
        System.out.println("Never executes");
    }}"""
        
        # Insert after token
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if token.text in line:
                lines.insert(i + 1, fake_check)
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 2: False wrapper ==========
    
    def apply_false_wrapper(self, token: Token, content: str) -> str:
        """Add false-condition wrappers directly around a line.
        
        Args:
            token: Token to wrap
            content: Original content
            
        Returns:
            Modified content with wrappers
        """
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if token.text in line:
                indent = len(line) - len(line.lstrip())
                lines[i] = ' ' * indent + "if (true) {" + '\n' + line + '\n' + ' ' * indent + "}"
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 3: Redundant sanitizer ==========
    
    
    # ========== Operator 4: Inconsistent sequence ==========
    
    def apply_inconsistent_sequence(self, token: Token, content: str) -> str:
        """Reorder API call sequences.
        
        Args:
            token: Token in API sequence
            content: Original content
            
        Returns:
            Modified content with reordered sequence
        """
        # Simple implementation: add redundant init call
        init_call = f"init({token.text});"
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if token.text in line:
                lines.insert(i, f"    {init_call}")
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 5: Improbable transition ==========
    
    def apply_improbable_transition(self, token: Token, content: str) -> str:
        """Create improbable value transitions.
        
        Args:
            token: Token for value transition
            content: Original content
            
        Returns:
            Modified content with improbable transitions
        """
        # Create improbable but equivalent transformation
        transition = f"{token.text}_temp = {token.text} * 2 / 2;"
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if token.text in line and '=' in line:
                lines.insert(i + 1, f"    {transition}")
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 6: Non-monotonic constraint ==========
    
    def apply_nonmonotonic_constraint(self, token: Token, content: str,
                                     oscillation_count: int = 2) -> str:
        """Create non-monotonic constraint progression.
        
        Args:
            token: Token for constraint
            content: Original content
            oscillation_count: Number of oscillations
            
        Returns:
            Modified content with non-monotonic constraints
        """
        constraints = []
        for i in range(oscillation_count):
            if i % 2 == 0:
                # Tighten
                constraints.append(f"if ({token.text} > 10) {{ /* tighten */ }}")
            else:
                # Loosen
                constraints.append(f"if ({token.text} > 100) {{ /* loosen */ }}")
        
        constraint_code = '\n    '.join(constraints)
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if token.text in line:
                lines.insert(i, f"    {constraint_code}")
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 7: Complexity injection ==========
    
    def apply_complexity_injection(self, token: Token, content: str,
                                   complexity_mode: str = 'predicate') -> str:
        """Inject high-complexity expressions.
        
        Args:
            token: Token to make complex
            content: Original content
            complexity_mode: 'predicate' or 'expression'
            
        Returns:
            Modified content with complex expressions
        """
        if complexity_mode == 'predicate':
            # Complex boolean expression
            complex_expr = f"(({token.text} ^ 8) + ({token.text} & 8)) == 24"
        elif complexity_mode == 'expression':
            # Complex arithmetic expression
            complex_expr = f"(({token.text} << 1) | ({token.text} >> 1)) & 0xFF"
        else:
            return content
        
        # Replace simple uses with complex expression
        return content.replace(token.text, f"({complex_expr})", 1)
    
    # ========== Main apply method ==========
    
    DEFAULT_OPERATORS = (
        "unreachable_branch",
        "false_wrapper",
        "inconsistent_sequence",
        "improbable_transition",
        "nonmonotonic_constraint",
        "complexity_injection",
    )

    def apply(self, token: Token, content: str, **kwargs) -> str:
        """Apply control-flow misdirection.
        
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
        
        if operator == 'unreachable_branch':
            return self.apply_unreachable_branch(token, content, **filtered_kwargs)
        elif operator == 'false_wrapper':
            return self.apply_false_wrapper(token, content, **filtered_kwargs)
        elif operator == 'inconsistent_sequence':
            return self.apply_inconsistent_sequence(token, content)
        elif operator == 'improbable_transition':
            return self.apply_improbable_transition(token, content)
        elif operator == 'nonmonotonic_constraint':
            return self.apply_nonmonotonic_constraint(token, content, **filtered_kwargs)
        elif operator == 'complexity_injection':
            return self.apply_complexity_injection(token, content, **filtered_kwargs)
        return content
