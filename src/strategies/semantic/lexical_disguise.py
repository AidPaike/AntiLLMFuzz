"""Lexical semantic disguise strategy.
This strategy disguises variables, constants, and dangerous strings
to mislead LLM understanding.
词汇语义伪装策略。
该策略对变量、常量和危险字符串进行伪装
误导大型语言模型的理解。
"""

import io
import keyword
import re
import tokenize
from typing import Optional, Pattern, Sequence
from ..base_strategy import PerturbationStrategy
from src.data_models import Token
from src.utils import get_llm_client, get_logger, LLMClient


class LexicalDisguiseStrategy(PerturbationStrategy):
    """Lexical-level semantic disguise through identifier manipulation.
    
    This strategy contains 5 operators that disguise lexical elements
    to mislead LLM pattern matching.
    
    """
    
    MISLEADING_EXAMPLES = [
        ("sanitize", "rawPayload"),
        ("validate", "bypassGuard"),
        ("clean", "dirtyBuffer"),
        ("safe", "unsafeHint"),
        ("checked", "unchecked"),
        ("secure", "insecure"),
        ("input", "outputStream"),
        ("source", "sink"),
        ("untrusted", "trusted"),
        ("user", "admin"),
        ("public", "private"),
        ("read", "write"),
    ]

    MISLEADING_MAPPINGS = dict(MISLEADING_EXAMPLES)

    # Mild renaming suggestions (used for few-shot guidance and fallback)
    MILD_EXAMPLES = [
        ("len", "dataSize"),
        ("buf", "buffer"),
        ("ptr", "pointer"),
        ("idx", "index"),
        ("cnt", "count"),
        ("tmp", "temporary"),
    ]

    MILD_RENAMINGS = dict(MILD_EXAMPLES)
    
    # Dangerous keywords (backward-compat name kept, but detection is pattern-based)
    DEFAULT_DANGEROUS_KEYWORDS = [
        "admin", "root", "password", "passwd", "secret", "token", "apikey",
        "auth", "credential", "ssh", "sudo", "flag",
        "select", "insert", "delete", "update", "drop", "exec", "eval", "system",
    ]
    DEFAULT_DANGEROUS_PATTERNS: Sequence[Pattern[str]] = (
        re.compile(r"(?i)api[_-]?key"),
        re.compile(r"(?i)secret"),
        re.compile(r"(?i)(?:pass|pwd)"),
        re.compile(r"(?i)(?:select|insert|delete|drop|update)\s+"),
        re.compile(r"(?i)(?:exec|eval|system|popen|os\.system)"),
    )
    DANGEROUS_KEYWORDS = DEFAULT_DANGEROUS_KEYWORDS
    MODE_DESCRIPTIONS = {
        'mild': "Keep the identifier's semantics intact but replace it with a more contextual, descriptive alias that still misleads model detectors.",
        'misleading': "Change the identifier to something plausible yet intentionally misleading so that a large language model might misinterpret its role."
    }

    FEW_SHOT_LIMIT = 4
    FEW_SHOT_EXAMPLES = {
        'mild': MILD_EXAMPLES[:FEW_SHOT_LIMIT],
        'misleading': MISLEADING_EXAMPLES[:FEW_SHOT_LIMIT],
    }

    IDENTIFIER_PROMPT_TEMPLATE = (
        "You are a code obfuscation assistant.\n"
        "Identifier: {identifier}\n"
        "{mode_description}\n"
        "Examples:\n{examples}\n"
        "Context: {context}\n"
        "Return exactly one identifier composed of letters, digits, or underscores. "
        "Do not include punctuation or explanations."
    )
    LLM_CONTEXT_MAX_LENGTH = 900
    LLM_IDENTIFIER_MAX_LENGTH = 48

    def __init__(self):
        """Initialize lexical disguise strategy."""
        super().__init__(
            name="lexical_disguise",
            description="Lexical-level semantic disguise through identifier manipulation",
            category="semantic",
            supported_targets=("code",),
            supported_languages=("python", "java"),
            code_safety="safe",
        )
        self.logger = get_logger(self.__class__.__name__)
        self.llm_client: Optional[LLMClient] = None

    # ========== Operator 1: Mild identifier renaming ==========
    
    def apply_mild_renaming(self, token: Token, content: str,
                            style: str = 'descriptive',
                            use_llm: bool = False,
                            llm_context: Optional[str] = None,
                            llm_temperature: float = 0.6,
                            llm_max_tokens: int = 48,
                             target: str = "code",
                             language: str = "any",
                             preserve_executability: bool = True,
                             allow_unsafe_code: bool = False) -> str:

        """Rename identifiers while maintaining their semantics and context.
        
        Args:
            token: Token to perturb
            content: Original content
            style: Reserved for future style hints
            use_llm: Whether to ask the LLM for a suggestion
            llm_context: Optional context string for the LLM
            llm_temperature: Sampling temperature when calling the LLM
            llm_max_tokens: Max tokens for the LLM response
        """
        return self.apply_identifier_disguise(
            token,
            content,
            mode='mild',
            use_llm=use_llm,
            llm_context=llm_context,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            target=target,
            language=language,
            preserve_executability=preserve_executability,
            allow_unsafe_code=allow_unsafe_code,
        )
    
    # ========== Operator 2: Misleading variable substitution ==========
    
    def apply_misleading_substitution(self, token: Token, content: str,
                                      mode: str = 'misleading',
                                      use_llm: bool = False,
                                      llm_context: Optional[str] = None,
                                      llm_temperature: float = 0.6,
                                      llm_max_tokens: int = 48,
                                      target: str = "code",
                                      language: str = "any",
                                      preserve_executability: bool = True,
                                      allow_unsafe_code: bool = False) -> str:

        """Apply misleading substitution to confuse LLM pattern matching."""
        return self.apply_identifier_disguise(
            token,
            content,
            mode='misleading',
            use_llm=use_llm,
            llm_context=llm_context,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            target=target,
            language=language,
            preserve_executability=preserve_executability,
            allow_unsafe_code=allow_unsafe_code,
        )

    def apply_identifier_disguise(self, token: Token, content: str,
                                 mode: str = 'mild',
                                 use_llm: bool = False,
                                 llm_context: Optional[str] = None,
                                 llm_temperature: float = 0.6,
                                 llm_max_tokens: int = 48,
                                 target: str = "code",
                                 language: str = "any",
                                 preserve_executability: bool = True,
                                 allow_unsafe_code: bool = False) -> str:

        """Core helper that handles identifier renames using LLM + rules."""
        original_name = token.text
        candidate = None

        if use_llm:
            context_snippet = self._prepare_context(llm_context or content)
            candidate = self._generate_llm_identifier(
                original_name,
                context_snippet,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
                mode=mode
            )

        if not candidate:
            candidate = self._fallback_identifier(original_name, mode)

        if not candidate:
            return content

        candidate = self._preserve_casing(original_name, candidate)

        # For executable source code, prefer syntax-preserving renames.
        if (
            (target or "").lower() == "code"
            and (language or "").lower() == "python"
            and preserve_executability
            and not allow_unsafe_code
        ):
            if token.token_type != "variable":
                return content
            if original_name.startswith("__") and original_name.endswith("__"):
                return content
            if not self._is_valid_identifier(candidate) or candidate == original_name:
                return content
            return self._rename_python_identifier(content, original_name, candidate)

        return content.replace(original_name, candidate, 1)

    def _is_valid_identifier(self, text: str) -> bool:
        return bool(text) and text.isidentifier() and not keyword.iskeyword(text)

    def _rename_python_identifier(self, source: str, old: str, new: str) -> str:
        """Rename a Python identifier using the tokenizer (avoids strings/comments)."""
        if not old or not new or old == new:
            return source
        if not self._is_valid_identifier(new):
            return source

        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
            rewritten = []
            paren_level = 0
            in_import_stmt = False
            prev_significant = None

            for tok in tokens:
                if tok.type == tokenize.OP and tok.string in "([{":
                    paren_level += 1
                elif tok.type == tokenize.OP and tok.string in ")]}":
                    paren_level = max(0, paren_level - 1)

                if tok.type == tokenize.NAME and tok.string in {"import", "from"} and paren_level == 0:
                    in_import_stmt = True
                elif tok.type in {tokenize.NEWLINE, tokenize.NL} and paren_level == 0:
                    in_import_stmt = False

                if (
                    tok.type == tokenize.NAME
                    and tok.string == old
                    and not in_import_stmt
                    and not (prev_significant and prev_significant.type == tokenize.OP and prev_significant.string == ".")
                ):
                    tok = tok._replace(string=new)

                rewritten.append(tok)

                if tok.type not in {tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}:
                    prev_significant = tok

            return tokenize.untokenize(rewritten)
        except Exception as exc:
            self.logger.debug("Python identifier rename failed: %s", exc)
            return source

    def _prepare_context(self, text: str) -> str:
        normalized = " ".join(text.split())
        if len(normalized) > self.LLM_CONTEXT_MAX_LENGTH:
            return normalized[-self.LLM_CONTEXT_MAX_LENGTH:]
        return normalized

    def _generate_llm_identifier(self, identifier: str, context: str,
                                 temperature: float, max_tokens: int,
                                 mode: str) -> Optional[str]:
        client = self._ensure_llm_client()
        if not client:
            return None

        prompt = self._build_identifier_prompt(
            identifier=identifier,
            context=context or "No additional context.",
            mode=mode
        )

        try:
            response = client.simple_completion(
                prompt=prompt,
                system_message="Provide one clever variable name.",
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as exc:
            self.logger.warning("LLM rename request failed: %s", exc)
            return None

        candidate = self._extract_identifier(response)
        if candidate:
            return candidate[:self.LLM_IDENTIFIER_MAX_LENGTH]
        return None

    def _fallback_identifier(self, original: str, mode: str) -> Optional[str]:
        """Fallback to rule-based examples when the LLM is unavailable."""
        mapping = self.MILD_RENAMINGS if mode == 'mild' else self.MISLEADING_MAPPINGS
        name_lower = original.lower()

        if name_lower in mapping:
            return mapping[name_lower]

        for original_token, substitution in mapping.items():
            if original_token in name_lower:
                return original.replace(original_token, substitution)

        return None

    def _build_identifier_prompt(self, identifier: str, context: str, mode: str) -> str:
        """Construct prompt for the LLM with mode-specific instructions."""
        mode_description = self.MODE_DESCRIPTIONS.get(mode, self.MODE_DESCRIPTIONS['misleading'])
        examples = self._render_examples(mode)

        return self.IDENTIFIER_PROMPT_TEMPLATE.format(
            identifier=identifier,
            mode_description=mode_description,
            examples=examples,
            context=context
        )

    def _render_examples(self, mode: str) -> str:
        examples = self.FEW_SHOT_EXAMPLES.get(mode, [])
        if not examples:
            return "No examples provided."

        return "\n".join(f"{source} -> {target}" for source, target in examples)

    def _ensure_llm_client(self) -> Optional[LLMClient]:
        if self.llm_client:
            return self.llm_client

        try:
            self.llm_client = get_llm_client()
            return self.llm_client
        except Exception as exc:
            self.logger.warning("Failed to initialize LLM client: %s", exc)
            return None

    def _extract_identifier(self, text: str) -> Optional[str]:
        sanitized = text.strip().splitlines()[0] if text else ""
        sanitized = sanitized.strip("`\"' ")
        sanitized = re.sub(r'[^0-9a-zA-Z_]', '', sanitized)

        if not sanitized:
            return None

        if sanitized[0].isdigit():
            sanitized = f"_{sanitized}"

        return sanitized

    def _preserve_casing(self, original: str, candidate: str) -> str:
        if original.isupper():
            return candidate.upper()
        if original[0].isupper():
            return candidate.capitalize()
        return candidate

    def _looks_sensitive_literal(
        self,
        text: str,
        keywords: Sequence[str],
        patterns: Sequence[Pattern[str]],
        force: bool = False,
    ) -> bool:
        """Heuristic to decide if a literal merits obfuscation."""
        if force:
            return True

        lowered = text.lower()
        if any(keyword in lowered for keyword in keywords):
            return True

        return any(pattern.search(text) for pattern in patterns)
    
    # ========== Operator 3: Literal boundary variants ==========
    
    def apply_literal_variant(self, token: Token, content: str,
                             variant_type: str = 'hex') -> str:
        """Apply literal boundary variants.
        
        Args:
            token: Token to perturb
            content: Original content
            variant_type: 'hex', 'binary', 'shift', 'separator'
            
        Returns:
            Modified content with literal variants
        """
        original_text = token.text
        
        # Try to parse as integer
        try:
            value = int(original_text)
            
            if variant_type == 'hex':
                variant = f"0x{value:X}"
            elif variant_type == 'binary':
                variant = f"0b{value:b}"
            elif variant_type == 'shift' and value > 0 and (value & (value - 1)) == 0:
                # Power of 2
                shift = value.bit_length() - 1
                variant = f"1<<{shift}"
            elif variant_type == 'separator' and value >= 1000:
                variant = f"{value:_}"
            else:
                return content
            
            return content.replace(original_text, variant, 1)
        except ValueError:
            return content
    
    # ========== Operator 4: Dangerous string pattern obfuscation ==========
    
    def apply_string_obfuscation(self, token: Token, content: str,
                                method: str = 'split',
                                dangerous_keywords: Optional[Sequence[str]] = None,
                                dangerous_patterns: Optional[Sequence[Pattern[str]]] = None,
                                force: bool = False) -> str:
        """Apply dangerous string pattern obfuscation.
        
        Args:
            token: Token to perturb
            content: Original content
            method: 'split', 'hex', 'concat'
            dangerous_keywords: Override keyword list; defaults to sensible secrets/SQL markers
            dangerous_patterns: Optional regex patterns that trigger obfuscation
            force: Bypass heuristics and always obfuscate the literal
            
        Returns:
            Modified content with obfuscated strings
        """
        original_text = token.text

        keywords = list(dangerous_keywords) if dangerous_keywords is not None else self.DEFAULT_DANGEROUS_KEYWORDS
        patterns = list(dangerous_patterns) if dangerous_patterns is not None else self.DEFAULT_DANGEROUS_PATTERNS

        # Check if dangerous
        if not self._looks_sensitive_literal(original_text, keywords, patterns, force):
            return content
        
        if method == 'split':
            # Split into parts
            mid = len(original_text) // 2
            part1 = original_text[:mid]
            part2 = original_text[mid:]
            obfuscated = f'"{part1}" + "{part2}"'
        elif method == 'hex':
            # Hex escape
            obfuscated = '"' + ''.join(f'\\x{ord(c):02x}' for c in original_text) + '"'
        else:
            return content
        
        # Try to replace with quotes
        for quote in ['"', "'"]:
            quoted_original = f"{quote}{original_text}{quote}"
            if quoted_original in content:
                return content.replace(quoted_original, obfuscated, 1)
        
        return content
    
    # ========== Main apply method ==========
    
    OPERATOR_SEQUENCE = [
        'mild_renaming',
        'misleading_substitution',
        'literal_variant',
        'string_obfuscation'
    ]

    OPERATOR_ARG_WHITELIST = {
        'mild_renaming': {'style', 'use_llm', 'llm_context', 'llm_temperature', 'llm_max_tokens', 'target', 'language', 'preserve_executability', 'allow_unsafe_code'},
        'misleading_substitution': {'use_llm', 'llm_context', 'llm_temperature', 'llm_max_tokens', 'target', 'language', 'preserve_executability', 'allow_unsafe_code'},
        'literal_variant': {'variant_type'},
        'string_obfuscation': {'method'}
    }

    def _apply_operator(self, operator: str, token: Token, content: str,
                        **kwargs) -> str:
        """Dispatch a single operator with filtered kwargs."""
        allowed = self.OPERATOR_ARG_WHITELIST.get(operator, set())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        if operator == 'mild_renaming':
            return self.apply_mild_renaming(token, content, **filtered_kwargs)
        if operator == 'misleading_substitution':
            return self.apply_misleading_substitution(token, content, **filtered_kwargs)
        if operator == 'literal_variant':
            return self.apply_literal_variant(token, content, **filtered_kwargs)
        if operator == 'string_obfuscation':
            return self.apply_string_obfuscation(token, content, **filtered_kwargs)
        return content

    def apply(self, token: Token, content: str, **kwargs) -> str:
        """Apply lexical disguise perturbation.
        
        Args:
            token: Token to perturb
            content: Original content
            **kwargs: Additional parameters
            
        Returns:
            Modified content
        """
        requested_operator = kwargs.get('operator') or 'preset'
        dispatch_kwargs = {k: v for k, v in kwargs.items() if k != 'operator'}

        if requested_operator != 'preset':
            return self._apply_operator(requested_operator, token, content, **dispatch_kwargs)

        perturbed = content
        for operator in self.OPERATOR_SEQUENCE:
            perturbed = self._apply_operator(operator, token, perturbed, **dispatch_kwargs)

        return perturbed
