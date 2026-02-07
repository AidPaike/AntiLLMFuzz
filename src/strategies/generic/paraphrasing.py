"""Paraphrasing & NL surface drift strategy.

This strategy changes natural language expression while preserving meaning.
"""
from ..base_strategy import PerturbationStrategy
from src.data_models import Token
import re

class ParaphrasingStrategy(PerturbationStrategy):
    """Paraphrasing & NL surface drift through synonym/phrase swaps.
    
    Operators:
    1. apply_paraphrasing - Synonym/phrase replacement
    2. apply_mixed_language - Mixed language/terminology
    """
    
    def __init__(self):
        super().__init__(
            name="paraphrasing",
            description="Paraphrasing & NL surface drift",
            category="generic",
            supported_targets=("documentation",),
            supported_languages=("any",),
            code_safety="unsafe",
        )
        
        # Synonym mappings for common security terms
        self.synonym_map = {
            'validate': 'verify',
            'check': 'examine',
            'sanitize': 'clean',
            'filter': 'screen',
            'input': 'data',
            'output': 'result',
            'buffer': 'storage',
            'length': 'size',
            'user': 'client',
            'password': 'credential',
            'authentication': 'verification',
            'authorization': 'permission',
            'secure': 'protected',
            'safe': 'protected',
            'error': 'issue',
            'exception': 'problem',
        }
        
        # Mixed language mappings (English → Chinese/abbreviations)
        self.mixed_language_map = {
            'validate': '验证',
            'sanitize': '清理',
            'check': '检查',
            'input': '输入',
            'output': '输出',
            'user': '用户',
            'password': '密码',
            'error': '错误',
        }
        
        # Abbreviation mappings
        self.abbreviation_map = {
            'validate': 'val',
            'sanitize': 'san',
            'authentication': 'auth',
            'authorization': 'authz',
            'configuration': 'config',
            'initialize': 'init',
            'parameter': 'param',
            'temporary': 'temp',
        }

        self._operator_arg_whitelist = {
            "paraphrasing": {"mode"},
            "mixed_language": {"target_language"},
        }
    
    DEFAULT_OPERATORS = (
        "paraphrasing",
        "mixed_language",
    )

    def apply(self, token: Token, content: str, **kwargs) -> str:
        """Apply paraphrasing based on operator parameter."""
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
        
        if operator == 'paraphrasing':
            return self.apply_paraphrasing(token, content, **filtered_kwargs)
        elif operator == 'mixed_language':
            return self.apply_mixed_language(token, content, **filtered_kwargs)
        else:
            return content

    
    def apply_paraphrasing(self, token: Token, content: str, 
                          mode: str = 'synonym') -> str:
        """Replace words with synonyms or paraphrased versions.
        
        Args:
            token: Target token
            content: Original content
            mode: Paraphrasing mode ('synonym', 'abbreviation')
            
        Returns:
            Modified content with paraphrased text
        """
        text_lower = token.text.lower()
        
        if mode == 'synonym':
            # Use synonym mapping
            if text_lower in self.synonym_map:
                replacement = self.synonym_map[text_lower]
                # Preserve case
                if token.text[0].isupper():
                    replacement = replacement.capitalize()
                return content.replace(token.text, replacement, 1)
        
        elif mode == 'abbreviation':
            # Use abbreviation mapping
            if text_lower in self.abbreviation_map:
                replacement = self.abbreviation_map[text_lower]
                # Preserve case
                if token.text[0].isupper():
                    replacement = replacement.upper()
                return content.replace(token.text, replacement, 1)
        
        return content
    
    def apply_mixed_language(self, token: Token, content: str, 
                            target_language: str = 'chinese') -> str:
        """Replace terms with mixed language equivalents.
        
        Args:
            token: Target token
            content: Original content
            target_language: Target language ('chinese', 'abbreviation', 'mixed')
            
        Returns:
            Modified content with mixed language
        """
        text_lower = token.text.lower()
        
        if target_language == 'chinese':
            # Replace with Chinese
            if text_lower in self.mixed_language_map:
                replacement = self.mixed_language_map[text_lower]
                return content.replace(token.text, replacement, 1)
        
        elif target_language == 'abbreviation':
            # Replace with abbreviation
            if text_lower in self.abbreviation_map:
                replacement = self.abbreviation_map[text_lower]
                if token.text[0].isupper():
                    replacement = replacement.upper()
                return content.replace(token.text, replacement, 1)
        
        elif target_language == 'mixed':
            # Mix English and Chinese
            if text_lower in self.mixed_language_map:
                chinese = self.mixed_language_map[text_lower]
                replacement = f"{token.text}({chinese})"
                return content.replace(token.text, replacement, 1)
        
        return content
