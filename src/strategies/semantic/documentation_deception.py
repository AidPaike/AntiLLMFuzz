"""Documentation-semantic deception strategy.

This strategy misleads through NL comments and documentation.
"""

from typing import List
import random
from ..base_strategy import PerturbationStrategy
from src.data_models import Token


class DocumentationDeceptionStrategy(PerturbationStrategy):
    """Documentation-level semantic deception through misleading comments.
    
    This strategy contains 3 operators that mislead LLMs through
    contradictory or fabricated documentation, without character-level or
    paraphrasing overlap.
    
    Operators:
    1. apply_misleading_comment - Insert misleading safety comments
    2. apply_pseudo_spec - Inject fake specifications
    3. apply_inconsistent_doc - Create inconsistent API documentation
    """
    
    MISLEADING_TEMPLATES = [
        "Input validated above",
        "Safe: sanitized",
        "Security check passed",
        "Note: Input validation complete",
        "Verified: no injection risk",
        "Sanitization applied",
        "TODO: Remove this security check (redundant)",
        "Safe by design",
        "Bounds checked",
        "Authentication verified",
    ]
    
    # Pseudo-spec templates
    PSEUDO_SPEC_TEMPLATES = {
        'javadoc': """/**
 * @precondition input must be validated
 * @postcondition returns sanitized value
 * @invariant length > 0
 */""",
        'jsdoc': """/**
 * @requires input is sanitized
 * @ensures output is safe
 * @param {string} input - validated input
 */""",
        'docstring': '''"""
Precondition: input is validated
Postcondition: returns safe value
Invariant: length > 0
"""''',
    }
    
    def __init__(self):
        """Initialize documentation deception strategy."""
        super().__init__(
            name="documentation_deception",
            description="Documentation-level semantic deception through misleading or fabricated docs",
            category="semantic",
            supported_targets=("code", "documentation"),
            supported_languages=("python", "java", "any"),
            code_safety="safe",
        )
        # Expose only deception operators; paraphrasing/mixed-language/char-perturbation are disabled to avoid overlap
        self._operator_arg_whitelist = {
            "misleading_comment": {"mode", "comment_style"},
            "pseudo_spec": {"spec_format", "contradiction_type"},
            "inconsistent_doc": {"inconsistency_target"},
        }

    def _comment_prefix(self, language: str) -> str:
        lang = (language or "any").lower()
        return "#" if lang == "python" else "//"

    def _line_comment(self, text: str, language: str) -> str:
        return f"{self._comment_prefix(language)} {text}".rstrip()
    
    # ========== Operator 1: Misleading comment ==========
    
    def apply_misleading_comment(self, token: Token, content: str,
                                 mode: str = 'semantic',
                                 comment_style: str = 'line',
                                 language: str = "any") -> str:
        """Insert misleading safety comments.
        
        Args:
            token: Token indicating insertion point
            content: Original content
            mode: 'semantic' (misleading) or 'formatting' (benign)
            comment_style: 'line' (//), 'block' (/* */)
            
        Returns:
            Modified content with misleading comments
        """
        comment_text = random.choice(self.MISLEADING_TEMPLATES)
        lang = (language or "any").lower()
        if comment_style == 'block' and lang != "python":
            comment = f"/* {comment_text} */"
        else:
            comment = self._line_comment(comment_text, language)
        
        # Insert before token
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if token.text in line:
                indent_str = line[: len(line) - len(line.lstrip())]
                lines.insert(i, indent_str + comment)
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 2: Pseudo-spec injection ==========
    
    def apply_pseudo_spec(self, token: Token, content: str,
                         spec_format: str = 'javadoc',
                         contradiction_type: str = 'precondition',
                         language: str = "any") -> str:
        """Inject fake specifications.
        
        Args:
            token: Token indicating insertion point
            content: Original content
            spec_format: 'javadoc', 'jsdoc', 'docstring'
            contradiction_type: 'precondition', 'postcondition', 'invariant'
            
        Returns:
            Modified content with pseudo-specs
        """
        lang = (language or "any").lower()
        if lang == "python":
            spec_lines = [
                self._line_comment("Precondition: input is validated", language),
                self._line_comment("Postcondition: returns safe value", language),
                self._line_comment("Invariant: length > 0", language),
            ]
            spec = "\n".join(spec_lines)
        else:
            spec = self.PSEUDO_SPEC_TEMPLATES.get(spec_format, self.PSEUDO_SPEC_TEMPLATES['javadoc'])
        
        # Insert before function/method
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if token.text in line and ('public' in line or 'def' in line or 'function' in line):
                indent_str = line[: len(line) - len(line.lstrip())]
                spec_lines = spec.split('\n')
                for j, spec_line in enumerate(spec_lines):
                    lines.insert(i + j, indent_str + spec_line)
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 3: Inconsistent API documentation ==========
    
    def apply_inconsistent_doc(self, token: Token, content: str,
                              inconsistency_target: str = 'safety',
                              language: str = "any") -> str:
        """Create inconsistent API documentation.
        
        Args:
            token: Token for documentation
            content: Original content
            inconsistency_target: 'safety', 'parameters', 'return_value'
            
        Returns:
            Modified content with inconsistent docs
        """
        if inconsistency_target == 'safety':
            doc = self._line_comment("Note: This function does NOT validate input (but it actually does)", language)
        elif inconsistency_target == 'parameters':
            doc = self._line_comment(f"@param {token.text} - must be non-null (but null is handled)", language)
        elif inconsistency_target == 'return_value':
            doc = self._line_comment("@return always returns null (but may return non-null)", language)
        else:
            doc = self._line_comment("Inconsistent documentation", language)
        
        # Insert before token
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if token.text in line:
                indent_str = line[: len(line) - len(line.lstrip())]
                lines.insert(i, indent_str + doc)
                break
        
        return '\n'.join(lines)
    
    # ========== Operator 4: Paraphrasing ==========
    
    def apply_paraphrasing(self, token: Token, content: str) -> str:
        """Paraphrase with different wording.
        
        Args:
            token: Token in comment
            content: Original content
            
        Returns:
            Modified content with paraphrased comments
        """
        # Simple paraphrasing: replace common security terms
        paraphrase_map = {
            'validate': 'verify',
            'sanitize': 'clean',
            'check': 'inspect',
            'secure': 'protected',
            'safe': 'harmless'
        }
        
        modified = content
        for original, replacement in paraphrase_map.items():
            if original in content:
                modified = modified.replace(original, replacement, 1)
                break
        
        return modified
    
    # ========== Operator 5: Mixed language ==========
    
    def apply_mixed_language(self, token: Token, content: str,
                            target_language: str = 'mixed') -> str:
        """Mix languages/terminology.
        
        Args:
            token: Token in comment
            content: Original content
            target_language: 'mixed', 'chinese', 'abbreviation'
            
        Returns:
            Modified content with mixed language
        """
        if target_language == 'chinese':
            # Replace some English terms with Chinese
            replacements = {
                'validate': '验证',
                'sanitize': '清理',
                'check': '检查',
                'input': '输入'
            }
        elif target_language == 'abbreviation':
            # Use abbreviations
            replacements = {
                'validate': 'val',
                'sanitize': 'san',
                'check': 'chk',
                'input': 'inp'
            }
        else:
            # Mixed
            replacements = {
                'validate': 'val验证',
                'sanitize': 'san清理'
            }
        
        modified = content
        for original, replacement in replacements.items():
            if original in content:
                modified = modified.replace(original, replacement, 1)
                break
        
        return modified
    
    # ========== Operator 6: Doc character perturbation ==========
    
    def apply_doc_char_perturbation(self, token: Token, content: str,
                                    perturbation_type: str = 'zero_width') -> str:
        """Apply character-level perturbation in documentation.
        
        Args:
            token: Token in documentation
            content: Original content
            perturbation_type: 'zero_width', 'homoglyph'
            
        Returns:
            Modified content with character perturbations
        """
        if perturbation_type == 'zero_width':
            # Insert zero-width space in comments
            zw_char = "\u200b"
            for marker in ("//", "#", "/*"):
                if marker in content:
                    return content.replace(marker, f"{marker}{zw_char}", 1)
            modified = content
        elif perturbation_type == 'homoglyph':
            # Replace 'a' with Cyrillic 'а' in comments
            modified = content
            if '//' in content or '/*' in content or '#' in content:
                modified = content.replace('a', '\u0430', 1)
        else:
            modified = content
        
        return modified
    
    # ========== Main apply method ==========
    
    DEFAULT_OPERATORS = (
        "misleading_comment",
        "pseudo_spec",
        "inconsistent_doc",
    )

    def apply(self, token: Token, content: str, **kwargs) -> str:
        """Apply documentation deception.
        
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
        language = kwargs.get("language", "any")
        
        if operator == 'misleading_comment':
            return self.apply_misleading_comment(token, content, language=language, **filtered_kwargs)
        elif operator == 'pseudo_spec':
            return self.apply_pseudo_spec(token, content, language=language, **filtered_kwargs)
        elif operator == 'inconsistent_doc':
            return self.apply_inconsistent_doc(token, content, language=language, **filtered_kwargs)
        # Other operators (paraphrasing/mixed_language/doc_char_perturbation) are intentionally disabled to avoid overlap with paraphrasing/tokenization strategies.
        return content
