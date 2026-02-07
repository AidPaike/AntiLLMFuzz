"""Formatting noise injection strategy.

This strategy changes formatting and layout without touching semantics.
"""
from ..base_strategy import PerturbationStrategy
from src.data_models import Token
import re

class FormattingNoiseStrategy(PerturbationStrategy):
    """Formatting noise injection without semantic changes.
    
    Operators:
    1. apply_line_ending - Line ending transformation (LF↔CRLF)
    2. apply_tab_space - Tab↔Space replacement
    3. apply_comment_format - Comment formatting perturbation
    """
    
    def __init__(self):
        super().__init__(
            name="formatting_noise",
            description="Formatting noise injection without semantic changes",
            category="generic",
            supported_targets=("code", "documentation"),
            supported_languages=("python", "java", "any"),
            code_safety="safe",
        )
        self._operator_arg_whitelist = {
            "line_ending": {"transform_type"},
            "tab_space": {"transform_type", "spaces_per_tab"},
            "comment_format": {"transform_type"},
        }
    
    DEFAULT_OPERATORS = (
        "line_ending",
        "tab_space",
        "comment_format",
    )

    def apply(self, token: Token, content: str, **kwargs) -> str:
        """Apply formatting noise based on operator parameter."""
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

        if operator == 'line_ending':
            return self.apply_line_ending(content, **filtered_kwargs)
        elif operator == 'tab_space':
            language = (kwargs.get("language") or "any").lower()
            preserve_executability = bool(kwargs.get("preserve_executability", False))
            allow_unsafe_code = bool(kwargs.get("allow_unsafe_code", False))
            if (
                language == "python"
                and preserve_executability
                and not allow_unsafe_code
                and filtered_kwargs.get("transform_type") == "space_to_tab"
            ):
                return content
            return self.apply_tab_space(content, **filtered_kwargs)
        elif operator == 'comment_format':
            language = kwargs.get("language", "any")
            return self.apply_comment_format(content, language=language, **filtered_kwargs)
        else:
            return content

    def apply_line_ending(self, content: str, transform_type: str = 'lf_to_crlf') -> str:

        """Transform line endings.
        
        Args:
            content: Original content
            transform_type: Type of transformation ('lf_to_crlf', 'crlf_to_lf')
            
        Returns:
            Modified content with transformed line endings
        """
        if transform_type == 'lf_to_crlf':
            # LF → CRLF
            normalized = content.replace('\r\n', '\n')
            return normalized.replace('\n', '\r\n')
        elif transform_type == 'crlf_to_lf':
            # CRLF → LF
            return content.replace('\r\n', '\n')
        else:
            return content
    
    def apply_tab_space(self, content: str, transform_type: str = 'tab_to_space', 
                       spaces_per_tab: int = 4) -> str:
        """Transform tabs and spaces.
        
        Args:
            content: Original content
            transform_type: Type of transformation ('tab_to_space', 'space_to_tab')
            spaces_per_tab: Number of spaces per tab
            
        Returns:
            Modified content with transformed whitespace
        """
        if transform_type == 'tab_to_space':
            # Tab → Spaces
            return content.replace('\t', ' ' * spaces_per_tab)
        elif transform_type == 'space_to_tab':
            # Spaces → Tab (at line beginnings only)
            lines = content.split('\n')
            result = []
            for line in lines:
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces >= spaces_per_tab:
                    # Replace leading spaces with tabs
                    tabs = leading_spaces // spaces_per_tab
                    remaining = leading_spaces % spaces_per_tab
                    new_line = '\t' * tabs + ' ' * remaining + line.lstrip(' ')
                    result.append(new_line)
                else:
                    result.append(line)
            return '\n'.join(result)
        else:
            return content
    
    def apply_comment_format(self, content: str, transform_type: str = 'line_to_block', language: str = "any") -> str:
        """Transform comment formatting.
        
        Args:
            content: Original content
            transform_type: Type of transformation ('line_to_block', 'block_to_line', 'reposition')
            
        Returns:
            Modified content with transformed comments
        """
        lang = (language or "any").lower()
        if lang == "python":
            if transform_type == 'reposition':
                lines = content.split('\n')
                result = []
                for line in lines:
                    if '#' in line:
                        code_part, comment_part = line.split('#', 1)
                        code_part = code_part.rstrip()
                        comment_part = comment_part.strip()
                        if code_part and comment_part:
                            result.append(f"# {comment_part}")
                            result.append(code_part)
                        else:
                            result.append(line)
                    else:
                        result.append(line)
                return '\n'.join(result)
            return content

        if transform_type == 'line_to_block':
            # Convert consecutive line comments to block comment
            # // comment1\n// comment2 → /* comment1\n   comment2 */
            lines = content.split('\n')
            result = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.strip().startswith('//'):
                    # Collect consecutive line comments
                    comment_lines = []
                    while i < len(lines) and lines[i].strip().startswith('//'):
                        comment_text = lines[i].strip()[2:].strip()
                        comment_lines.append(comment_text)
                        i += 1
                    # Convert to block comment
                    if len(comment_lines) == 1:
                        result.append(f"/* {comment_lines[0]} */")
                    else:
                        result.append("/*")
                        for comment in comment_lines:
                            result.append(f" * {comment}")
                        result.append(" */")
                else:
                    result.append(line)
                    i += 1
            return '\n'.join(result)
        
        elif transform_type == 'block_to_line':
            # Convert block comments to line comments
            # /* comment */ → // comment
            def replace_block(match):
                comment_text = match.group(1).strip()
                if '\n' in comment_text:
                    # Multi-line block comment
                    lines = comment_text.split('\n')
                    return '\n'.join([f"// {line.strip().lstrip('*').strip()}" for line in lines if line.strip()])
                else:
                    # Single-line block comment
                    return f"// {comment_text}"
            
            return re.sub(r'/\*(.*?)\*/', replace_block, content, flags=re.DOTALL)
        
        elif transform_type == 'reposition':
            # Move inline comments to previous line
            lines = content.split('\n')
            result = []
            for line in lines:
                # Check for inline comment
                if '//' in line:
                    parts = line.split('//', 1)
                    code_part = parts[0].rstrip()
                    comment_part = parts[1].strip()
                    if code_part:  # Has code before comment
                        result.append(f"// {comment_part}")
                        result.append(code_part)
                    else:
                        result.append(line)
                else:
                    result.append(line)
            return '\n'.join(result)
        
        else:
            return content
