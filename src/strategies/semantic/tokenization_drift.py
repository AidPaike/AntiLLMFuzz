"""Tokenization-level semantic drift strategy.

This strategy disrupts tokenization through character-level manipulation
while preserving semantic meaning.
"""

from __future__ import annotations

import random
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from ..base_strategy import PerturbationStrategy
from src.data_models import Token
from src.utils import get_config_loader


# NOTE: Keep these utilities/module-level constants outside the class body.
# Python comprehensions inside class bodies cannot reliably reference other
# class variables due to scoping rules.

def _iter_codepoints(ranges: Sequence[Tuple[int, int]]) -> Iterable[int]:
    for start, end in ranges:
        for cp in range(start, end + 1):
            yield cp


def _dedupe_preserve_order(items: Iterable[Any]) -> List[Any]:
    seen = set()
    ordered: List[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _codepoints_to_chars(points: Iterable[int]) -> Tuple[str, ...]:
    return tuple(chr(cp) for cp in points)


def _regex_class_for_chars(chars: Sequence[str]) -> str:
    return "[" + "".join(re.escape(ch) for ch in chars) + "]"


# A broadened superset of zero-width / default-ignorable characters covering:
# - Default ignorable Cf characters (e.g., soft hyphen, CGJ, ALM, Syriac mark)
# - Zero-width joiners/non-joiners, bidi controls, hangul fillers
# - Variation selectors (BMP + supplementary), tag characters, interlinear marks
# This provides full coverage while allowing callers to choose safer subsets.
_ZERO_WIDTH_CODEPOINT_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x00AD, 0x00AD),  # SOFT HYPHEN
    (0x034F, 0x034F),  # COMBINING GRAPHEME JOINER
    (0x061C, 0x061C),  # ARABIC LETTER MARK
    (0x070F, 0x070F),  # SYRIAC ABBREVIATION MARK
    (0x115F, 0x1160),  # HANGUL CHOSEONG/JUNGSEONG FILLER
    (0x17B4, 0x17B5),  # KHMER VOWEL INHERENT (zero advance width)
    (0x180B, 0x180E),  # MONGOLIAN VARIATION SELECTORS + MVS
    (0x200B, 0x200F),  # ZWSP..RLM
    (0x202A, 0x202E),  # LRE..RLO
    (0x2060, 0x2064),  # WORD JOINER..INVISIBLE PLUS
    (0x2066, 0x2069),  # BIDI ISOLATES
    (0x206A, 0x206F),  # OTHER FORMATTING CONTROLS
    (0x3164, 0x3164),  # HANGUL FILLER
    (0xFE00, 0xFE0F),  # VARIATION SELECTOR-1..16
    (0xFEFF, 0xFEFF),  # ZERO WIDTH NO-BREAK SPACE (BOM)
    (0xFFA0, 0xFFA0),  # HALFWIDTH HANGUL FILLER
    (0xFFF9, 0xFFFB),  # INTERLINEAR ANNOTATION ANCHOR..TERMINATOR
    (0x1BCA0, 0x1BCA3),  # SHORTHAND FORMAT CONTROLS
    (0x1D173, 0x1D17A),  # MUSICAL SYMBOL FORMAT CONTROLS
    (0xE0000, 0xE0001),  # LANGUAGE TAG + CANCEL TAG
    (0xE0020, 0xE007F),  # TAG CHARACTERS
    (0xE0100, 0xE01EF),  # VARIATION SELECTOR SUPPLEMENT
)

_ZERO_WIDTH_ALL_CODEPOINTS: Tuple[int, ...] = tuple(_iter_codepoints(_ZERO_WIDTH_CODEPOINT_RANGES))
_ZERO_WIDTH_ALL_CHARS: Tuple[str, ...] = _codepoints_to_chars(_ZERO_WIDTH_ALL_CODEPOINTS)

# Conservative subsets so callers can avoid unstable bidi overrides or tag/VS noise.
_ZERO_WIDTH_MINIMAL: Tuple[int, ...] = (
    0x200B,  # ZWSP
    0x200C,  # ZWNJ
    0x200D,  # ZWJ
)
_ZERO_WIDTH_SAFE_CODEPOINTS: Tuple[int, ...] = tuple(
    _dedupe_preserve_order(
        _ZERO_WIDTH_MINIMAL
        + (
            0x034F,  # CGJ
            0x061C,  # ALM
            0x2060,  # WORD JOINER
            0x180B,
            0x180C,
            0x180D,
            0x180E,
        )
    )
)
_ZERO_WIDTH_BIDI_CODEPOINTS: Tuple[int, ...] = tuple(
    _iter_codepoints(((0x200E, 0x200F), (0x202A, 0x202E), (0x2066, 0x2069)))
)
_ZERO_WIDTH_VARIATION_CODEPOINTS: Tuple[int, ...] = tuple(
    _iter_codepoints(((0xFE00, 0xFE0F), (0xE0100, 0xE01EF)))
)
_ZERO_WIDTH_TAG_CODEPOINTS: Tuple[int, ...] = tuple(_iter_codepoints(((0xE0000, 0xE0001), (0xE0020, 0xE007F))))

ZERO_WIDTH_CHARSETS: Dict[str, Tuple[str, ...]] = {
    "minimal": _codepoints_to_chars(_ZERO_WIDTH_MINIMAL),
    "safe": _codepoints_to_chars(_ZERO_WIDTH_SAFE_CODEPOINTS),
    "default_ignorable": _ZERO_WIDTH_ALL_CHARS,
    "bidi": _codepoints_to_chars(_ZERO_WIDTH_BIDI_CODEPOINTS),
    "variation": _codepoints_to_chars(_ZERO_WIDTH_VARIATION_CODEPOINTS),
    "tags": _codepoints_to_chars(_ZERO_WIDTH_TAG_CODEPOINTS),
    "all": _ZERO_WIDTH_ALL_CHARS,
}

ZERO_WIDTH_CHARS_BY_CODEPOINT: Dict[str, str] = {f"U+{cp:04X}": chr(cp) for cp in _ZERO_WIDTH_ALL_CODEPOINTS}
ZERO_WIDTH_RE = re.compile(_regex_class_for_chars(ZERO_WIDTH_CHARSETS["all"]))


def _line_start_indices(text: str) -> List[int]:
    # Supports both LF and CRLF because we only key off '\n'.
    starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def _approx_token_offset(token: Token, content: str) -> Optional[int]:
    """Best-effort mapping of (line, column) to absolute index in `content`."""
    if token.column < 0:
        return None

    # Documentation extractor uses absolute character offsets in `column`.
    if token.source_file.lower().endswith((".md", ".txt", ".rst", ".adoc", ".javadoc")):
        return token.column

    # Code extractors generally provide (1-indexed line, 0-indexed column).
    if token.line <= 0:
        return None

    line_starts = _line_start_indices(content)
    if token.line > len(line_starts):
        return None

    return line_starts[token.line - 1] + token.column


def _find_best_span(token: Token, content: str, text: str) -> Optional[Tuple[int, int]]:
    if not text:
        return None

    approx = _approx_token_offset(token, content)

    # Exact match at approx offset.
    if approx is not None:
        end = approx + len(text)
        if 0 <= approx <= len(content) and end <= len(content) and content[approx:end] == text:
            return approx, end

    # Otherwise choose the occurrence closest to approx offset, or first occurrence.
    positions: List[int] = []
    start = 0
    while True:
        idx = content.find(text, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1

    if not positions:
        return None

    if approx is None:
        best = positions[0]
    else:
        best = min(positions, key=lambda p: abs(p - approx))

    return best, best + len(text)


def _replace_once(token: Token, content: str, original: str, replacement: str) -> str:
    span = _find_best_span(token, content, original)
    if span is None:
        return content
    start, end = span
    return content[:start] + replacement + content[end:]


class TokenizationDriftStrategy(PerturbationStrategy):
    """Tokenization-level semantic drift through character manipulation.
    
    This strategy contains 7 operators that manipulate characters/encoding
    to disrupt LLM tokenization while keeping program behavior unchanged.
    
    Operators:
    1. apply_zero_width - Insert zero-width characters (U+200B, U+200C, etc.)
    2. apply_homoglyph - Replace with visually similar Unicode characters
    3. apply_control_char - Inject backspace/delete control characters
    4. apply_bidi_override - Insert bidirectional override characters (U+202E)
    5. apply_string_fragmentation - Fragment string literals
    6. apply_joiner - Insert zero-width joiners to shift token boundaries
    """
    
    # Zero-width / format characters (Unicode Cf + common zero-width joiners).
    # For broad matching, use `ZERO_WIDTH_RE` (module-level).
    # For selection, use `ZERO_WIDTH_CHARSETS` (module-level).
    ZERO_WIDTH_CHARS = ZERO_WIDTH_CHARS_BY_CODEPOINT
    
    # Control characters
    CONTROL_CHARS = {
        'backspace': "\u0008",  # Backspace
        'delete': "\u007F",     # Delete
    }
    
    # Bidirectional override characters
    BIDI_CHARS = {
        'rtl': "\u202E",  # Right-to-left override
        'ltr': "\u202D",  # Left-to-right override
        'pdf': "\u202C",  # Pop directional formatting
    }
    FRAGMENTABLE_TOKEN_TYPES: Tuple[str, ...] = ("string", "literal", "text", "doc", "comment", "phrase")
    
    # Homoglyph mappings (expanded cross-script confusables)
    _RAW_HOMOGLYPHS: Dict[str, Sequence[str]] = {
        "a": ["\u0430", "\u0251", "\u1d00", "\uff41", "\u03b1", "\u2c6f", "\u13aa", "\u1d43"],
        "b": ["\u13cf", "\u15af", "\uff42", "\u03b2", "\u0184", "\u042c", "\u0180"],
        "c": ["\u03f2", "\u0441", "\uff43", "\u217d", "\u2ca5", "\u03f9", "\u1d04"],
        "d": ["\u0501", "\u1d05", "\uff44", "\u13e7", "\u2146", "\u1d06"],
        "e": ["\u0435", "\u212e", "\uff45", "\u0454", "\u03b5", "\u1d07", "\u212f"],
        "f": ["\u0192", "\u1e1f", "\uff46", "\ua730", "\u0191", "\u1da0"],
        "g": ["\u0261", "\u1d83", "\uff47", "\u13c0", "\u0262", "\u1da2"],
        "h": ["\u04bb", "\u13c2", "\uff48", "\u210e", "\u0570", "\u13f7", "\u02b0"],
        "i": ["\u0456", "\u0131", "\uff49", "\u1d62", "\u2170", "\u1d09"],
        "j": ["\u03f3", "\u029d", "\uff4a", "\u2149", "\u1d0a"],
        "k": ["\u03ba", "\u043a", "\uff4b", "\u1d0b", "\u212a", "\u2c94", "\u049b"],
        "l": ["\u04cf", "\u029f", "\uff4c", "\u2113", "\u01c0", "\u2223"],
        "m": ["\u043c", "\u1d0d", "\u217f", "\uff4d", "\u10db", "\u03bc", "\u1d21"],
        "n": ["\u0578", "\uff4e", "\u0572", "\u03bd"],
        "o": ["\u03bf", "\u043e", "\uff4f", "\u10dd", "\u0275", "\u1d0f"],
        "p": ["\u0440", "\u1d7d", "\uff50", "\u03c1", "\u2ca3"],
        "q": ["\u051b", "\u0566", "\uff51", "\u24e0", "\u02a0"],
        "r": ["\u027d", "\u1d63", "\uff52", "\u1d72", "\u13a1"],
        "s": ["\u0455", "\u1d74", "\uff53", "\u10e1", "\u01bd", "\ua731", "\u0282"],
        "t": ["\u0442", "\u1d75", "\uff54", "\u1d1b", "\u03c4"],
        "u": ["\u028b", "\u03c5", "\uff55", "\u1d1c", "\u10e3", "\u1d1d"],
        "v": ["\u1d65", "\u03bd", "\uff56", "\u1d20", "\u0475", "\u2174", "\u2228"],
        "w": ["\u051d", "\u13b3", "\uff57", "\u1d21", "\u0461", "\u1e81"],
        "x": ["\u0445", "\u03c7", "\uff58", "\u2573", "\u166e"],
        "y": ["\u0443", "\u03c5", "\uff59", "\u028e", "\u04af", "\u1eff"],
        "z": ["\u1d22", "\u0240", "\uff5a", "\u01b6", "\u03b6", "\u1d76"],
        "0": ["\uff10", "\u3007", "\u0660", "\u06f0"],
        "1": ["\uff11", "\u2460", "\u0661", "\u06f1"],
        "2": ["\uff12", "\u2461", "\u0662", "\u06f2"],
        "3": ["\uff13", "\u2462", "\u0663", "\u06f3"],
        "4": ["\uff14", "\u2463", "\u0664", "\u06f4"],
        "5": ["\uff15", "\u2464", "\u0665", "\u06f5"],
        "6": ["\uff16", "\u2465", "\u0666", "\u06f6"],
        "7": ["\uff17", "\u2466", "\u0667", "\u06f7"],
        "8": ["\uff18", "\u2467", "\u0668", "\u06f8"],
        "9": ["\uff19", "\u2468", "\u0669", "\u06f9"],
    }
    HOMOGLYPHS: Dict[str, Tuple[str, ...]] = {
        k: tuple(_dedupe_preserve_order(v)) for k, v in _RAW_HOMOGLYPHS.items()
    }
    
    def __init__(self):
        """Initialize tokenization drift strategy."""
        super().__init__(
            name="tokenization_drift",
            description="Tokenization-level semantic drift through character manipulation",
            category="semantic",
            supported_targets=("documentation",),
            supported_languages=("any",),
            code_safety="unsafe",
        )
        # Operator registry is built lazily in `_operator_dispatch`.
        self._last_variant_plan: List[Dict[str, Any]] = []

    def _passes_score_gate(
        self,
        token: Token,
        scs_threshold: float = 70.0,
        priority_threshold: float = 7.0,
        total_threshold: float = 10.0,
        score_attr: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> bool:
        """Check whether a token is "important" enough to perturb aggressively."""
        if score_attr:
            value = getattr(token, score_attr, 0.0)
            required = score_threshold if score_threshold is not None else scs_threshold
            return value >= required

        if token.scs_score > 0:
            return token.scs_score >= scs_threshold

        if getattr(token, "total_score", 0.0) >= total_threshold:
            return True

        return token.priority_score >= priority_threshold

    def _select_zero_width_char(self, char_type: Optional[str], charset: str = "safe") -> str:
        """Select a zero-width/format character from a named charset or explicit codepoint."""
        candidates = ZERO_WIDTH_CHARSETS.get(charset, ZERO_WIDTH_CHARSETS["safe"])

        if char_type is None or char_type in {"auto", "random"}:
            return random.choice(candidates)

        # Explicit key like "U+200B"
        if char_type in self.ZERO_WIDTH_CHARS:
            return self.ZERO_WIDTH_CHARS[char_type]

        # Allow raw character input
        if len(char_type) == 1 and ZERO_WIDTH_RE.search(char_type):
            return char_type

        # Parse flexible "U+XXXX" or "200B"
        hex_part = char_type.upper().removeprefix("U+").removeprefix("0X")
        try:
            codepoint = int(hex_part, 16)
            if 0 <= codepoint <= 0x10FFFF:
                return chr(codepoint)
        except ValueError:
            pass

        return random.choice(candidates)

    def _operator_dispatch(self) -> Dict[str, Any]:
         return {
             "zero_width": self.apply_zero_width,
             "homoglyph": self.apply_homoglyph,
             "control_char": self.apply_control_char,
             "bidi_override": self.apply_bidi_override,
             "string_fragmentation": self.apply_string_fragmentation,
             "joiner": self.apply_joiner,
         }


    def _auto_operator_plan(self, token: Token) -> List[str]:
        token_type = (token.token_type or "").lower()
        plan: List[str] = []

        if token_type in {"string", "literal", "text", "doc", "comment", "phrase"}:
            plan.append("string_fragmentation")
        if token_type in {"function", "variable", "identifier", "class", "type", "param", "keyword", "name"}:
            plan.append("homoglyph")
        if token.text.isdigit():
            plan.append("homoglyph")
        if "_" in token.text or len(token.text) > 12:
            plan.append("joiner")

        plan.append("zero_width")
        return _dedupe_preserve_order(plan)

    def _load_tsd_config(self) -> Dict[str, Any]:
        try:
            return get_config_loader().get_tsd_config() or {}
        except Exception:
            return {}

    def _merge_operator_params(
        self,
        base_params: Optional[Dict[str, Dict[str, Any]]],
        override_params: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = dict(base_params or {})
        for key, value in (override_params or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    def _apply_tsd_defaults(
        self,
        operator_params: Dict[str, Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        zero_width_char = config.get("zero_width_char")
        if zero_width_char:
            operator_params.setdefault("zero_width", {})
            operator_params["zero_width"].setdefault("char_type", zero_width_char)
        return operator_params

    def _select_operator_sequence(
        self,
        token: Token,
        operators: Optional[Sequence[str]],
        schedule: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        operator_config = (config or {}).get("operators", {})
        known_ops = set(self._operator_dispatch().keys())
        mode = operator_config.get("mode", "all")

        if operators is None:
            if mode == "single":
                single_op = operator_config.get("single") or operator_config.get("operator")
                candidates = [single_op] if single_op else []
            elif mode == "subset":
                candidates = list(operator_config.get("list") or operator_config.get("operators") or [])
            elif mode == "auto":
                candidates = self._auto_operator_plan(token)
            else:
                candidates = list(self._operator_dispatch().keys())
        elif isinstance(operators, str):
            candidates = [operators]
        else:
            candidates = list(operators)

        ordered = [op for op in _dedupe_preserve_order(candidates) if op in known_ops]
        if not ordered:
            ordered = list(self._operator_dispatch().keys())

        if schedule == "random":
            return random.sample(ordered, k=len(ordered)) if ordered else []
        return ordered

    def _apply_operator(self, operator: str, token: Token, content: str, params: Dict[str, Any]) -> str:
        dispatch = self._operator_dispatch()
        handler = dispatch.get(operator)
        if handler is None:
            return content

        if operator == "whitespace_transform":
            return handler(token, content, **params)

        return handler(token, content, **params)
    
    # ========== Operator 1: Zero-width insertion ==========
    
    def apply_zero_width(self, token: Token, content: str,
                        char_type: str = "auto",
                        position: str = "middle",
                        charset: str = "safe") -> str:
        """Insert zero-width characters into tokens.
        
        Args:
            token: Token to perturb
            content: Original content
            char_type: Type of zero-width char ('U+200B', 'U+200C', etc.)
            position: Where to insert ('middle', 'start', 'end', 'all')
            
        Returns:
            Modified content with zero-width characters
        """
        original_text = token.text
        zw_char = self._select_zero_width_char(char_type, charset=charset)
        
        if position == 'middle':
            mid = len(original_text) // 2
            perturbed_text = original_text[:mid] + zw_char + original_text[mid:]
        elif position == 'start':
            perturbed_text = zw_char + original_text
        elif position == 'end':
            perturbed_text = original_text + zw_char
        elif position == 'all':
            perturbed_text = zw_char.join(original_text)
        else:
            perturbed_text = original_text
        
        return _replace_once(token, content, original_text, perturbed_text)
    
    # ========== Operator 2: Homoglyph replacement ==========
    
    def apply_homoglyph(self, token: Token, content: str,
                       replacement_ratio: float = 0.5,
                       variant_index: int = 0,
                       preferred_scripts: Optional[Sequence[str]] = None) -> str:
        """Replace characters with visually similar Unicode homoglyphs.
        
        Args:
            token: Token to perturb
            content: Original content
            replacement_ratio: Ratio of characters to replace (0.0-1.0)
            variant_index: Index of homoglyph variant to use
            preferred_scripts: Optional script filter (e.g., ["Cyrillic", "Greek"])
            
        Returns:
            Modified content with homoglyphs
        """
        original_text = token.text
        perturbed_chars = []
        replacements_made = 0
        max_replacements = max(1, int(len(original_text) * replacement_ratio))
        script_filter = {s.upper() for s in preferred_scripts} if preferred_scripts else None
        
        for char in original_text:
            char_lower = char.lower()
            if (char_lower in self.HOMOGLYPHS and 
                replacements_made < max_replacements):
                variants = list(self.HOMOGLYPHS[char_lower])
                if script_filter:
                    filtered = [
                        v for v in variants
                        if script_filter.intersection(set(unicodedata.name(v, "").upper().split()))
                    ]
                    if filtered:
                        variants = filtered

                if variant_index < 0:
                    selected_variant = random.choice(variants)
                else:
                    selected_variant = variants[variant_index % len(variants)]

                # Preserve case where possible.
                if char.isupper():
                    upper_variant = selected_variant.upper()
                    if len(upper_variant) == 1:
                        selected_variant = upper_variant
                perturbed_chars.append(selected_variant)
                replacements_made += 1
            else:
                perturbed_chars.append(char)
        
        perturbed_text = ''.join(perturbed_chars)
        return _replace_once(token, content, original_text, perturbed_text)
    
    # ========== Operator 3: Control character injection ==========
    
    def apply_control_char(self, token: Token, content: str,
                          control_type: str = 'backspace',
                          insertion_pattern: str = 'compensated') -> str:
        """Inject backspace/delete control characters.
        
        Args:
            token: Token to perturb
            content: Original content
            control_type: 'backspace' or 'delete'
            insertion_pattern: 'compensated' (preserve visible) or 'raw'
            
        Returns:
            Modified content with control characters
        """
        original_text = token.text
        control_char = self.CONTROL_CHARS.get(control_type, self.CONTROL_CHARS['backspace'])
        
        if len(original_text) < 2:
            return content
        
        mid = len(original_text) // 2
        
        if insertion_pattern == 'compensated':
            # Insert dummy char + control char to preserve visible text
            perturbed_text = original_text[:mid] + 'X' + control_char + original_text[mid:]
        else:
            # Raw insertion
            perturbed_text = original_text[:mid] + control_char + original_text[mid:]
        
        return _replace_once(token, content, original_text, perturbed_text)
    
    # ========== Operator 4: Bidirectional override ==========
    
    def apply_bidi_override(self, token: Token, content: str,
                           direction: str = 'rtl',
                           scope: str = 'token') -> str:
        """Insert bidirectional override characters.
        
        Args:
            token: Token to perturb
            content: Original content
            direction: 'rtl' (right-to-left) or 'ltr' (left-to-right)
            scope: 'token' (wrap token) or 'line' (wrap line)
            
        Returns:
            Modified content with bidi characters
        """
        original_text = token.text
        bidi_start = self.BIDI_CHARS.get(direction, self.BIDI_CHARS['rtl'])
        bidi_end = self.BIDI_CHARS['pdf']
        
        perturbed_text = bidi_start + original_text + bidi_end
        return _replace_once(token, content, original_text, perturbed_text)
    
    # ========== Operator 5: String literal fragmentation ==========
    
    def apply_string_fragmentation(self, token: Token, content: str,
                                    fragment_size: int = 2,
                                    joiner_char: str = "auto",
                                    joiner_charset: str = "safe",
                                    scs_threshold: float = 70.0,
                                    priority_threshold: float = 7.0,
                                    total_threshold: float = 10.0,
                                    score_attr: Optional[str] = None,
                                    score_threshold: Optional[float] = None,
                                    target_token_types: Optional[Sequence[str]] = None) -> str:
        if fragment_size is None:
            fragment_size = 2
        if scs_threshold is None:
            scs_threshold = 70.0
        if priority_threshold is None:
            priority_threshold = 7.0
        if total_threshold is None:
            total_threshold = 10.0

        """Fragment a high-score token with invisible joiners.
        
        Args:
            token: Token to perturb
            content: Original content
            fragment_size: Size of each fragment
            joiner_char: Zero-width character (e.g. "U+200B") or "auto"
            joiner_charset: Charset for auto selection ("safe" or "all")
            scs_threshold: Apply only if `token.scs_score` meets this threshold (when available)
            priority_threshold: Fallback threshold when `token.scs_score == 0`
            total_threshold: Composite threshold using `token.total_score`
            score_attr: Optional explicit token attribute to gate on (e.g. "total_score")
            score_threshold: Threshold to pair with `score_attr`
            target_token_types: Restrict fragmentation to these token types (defaults to string-like)
            
        Returns:
            Modified content with fragmented token text
        """
        original_text = token.text

        allowed_types = {
            t.lower() for t in (target_token_types or self.FRAGMENTABLE_TOKEN_TYPES)
        }
        if allowed_types and (token.token_type or "").lower() not in allowed_types:
            return content
        
        if not self._passes_score_gate(
            token,
            scs_threshold=scs_threshold,
            priority_threshold=priority_threshold,
            total_threshold=total_threshold,
            score_attr=score_attr,
            score_threshold=score_threshold,
        ):
            return content
        
        if fragment_size <= 0 or len(original_text) <= max(3, fragment_size):
            return content
        
        joiner = self._select_zero_width_char(joiner_char, charset=joiner_charset)

        # Avoid fragmenting across whitespace/punctuation runs: only split word-like chunks.
        parts: List[str] = []
        for match in re.finditer(r"\w+|\W+", original_text, flags=re.UNICODE):
            chunk = match.group(0)
            if re.fullmatch(r"\w+", chunk, flags=re.UNICODE) and len(chunk) > fragment_size:
                # Keep underscores readable/structured: fragment alpha-numeric runs, preserve '_' groups.
                subparts = re.split(r"(_+)", chunk)
                rebuilt: List[str] = []
                for sub in subparts:
                    if not sub or sub.startswith("_"):
                        rebuilt.append(sub)
                        continue

                    fragments = [sub[i:i + fragment_size] for i in range(0, len(sub), fragment_size)]
                    rebuilt.append(joiner.join(fragments))

                parts.append("".join(rebuilt))
            else:
                parts.append(chunk)
        
        fragmented_text = "".join(parts)
        if fragmented_text == original_text:
            return content

        return _replace_once(token, content, original_text, fragmented_text)
    
    # ========== Operator 6: Token boundary shift via joiners ==========
    
    def apply_joiner(self, token: Token, content: str,
                    joiner_type: str = 'zwj',
                    position: str = 'middle') -> str:
        """Insert zero-width joiners to shift token boundaries.
        
        Args:
            token: Token to perturb
            content: Original content
            joiner_type: 'zwj' (zero-width joiner) or 'zwnj' (zero-width non-joiner)
            position: Where to insert ('middle', 'random')
            
        Returns:
            Modified content with joiners
        """
        joiner_map = {
            'zwj': self.ZERO_WIDTH_CHARS['U+200D'],
            'zwnj': self.ZERO_WIDTH_CHARS['U+200C']
        }
        
        original_text = token.text
        joiner = joiner_map.get(joiner_type, joiner_map['zwj'])
        
        if position == 'middle':
            mid = len(original_text) // 2
            perturbed_text = original_text[:mid] + joiner + original_text[mid:]
        elif position == 'random' and original_text:
            idx = random.randint(1, len(original_text)) if len(original_text) > 1 else 0
            perturbed_text = original_text[:idx] + joiner + original_text[idx:]
        else:
            perturbed_text = original_text
        
        return _replace_once(token, content, original_text, perturbed_text)
    
    # ========== Operator 7: Whitespace transformation ==========
    
    def apply_whitespace_transform(self, content: str,
                                   mode: str = 'tab_to_space',
                                   spaces_per_tab: int = 4) -> str:
        """Transform tabs/spaces and line endings.
        
        Args:
            content: Original content
            mode: 'tab_to_space', 'space_to_tab', 'lf_to_crlf', 'crlf_to_lf'
            spaces_per_tab: Number of spaces per tab
            
        Returns:
            Modified content with transformed whitespace
        """
        if mode == 'tab_to_space':
            return content.replace('\t', ' ' * spaces_per_tab)
        elif mode == 'space_to_tab':
            return content.replace(' ' * spaces_per_tab, '\t')
        elif mode == 'lf_to_crlf':
            return content.replace('\n', '\r\n')
        elif mode == 'crlf_to_lf':
            return content.replace('\r\n', '\n')
        return content
    
    def _execute_plan(
        self,
        token: Token,
        content: str,
        selected_ops: Sequence[str],
        operator_params: Dict[str, Dict[str, Any]],
        legacy_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[str]]:
        """Apply a prepared operator sequence and record which operators changed the text."""
        shared_params = operator_params.get("*", {})
        applied_ops: List[str] = []
        modified_content = content
        legacy_params = legacy_params or {}

        for op_name in selected_ops:
            params = dict(shared_params)
            params.update(operator_params.get(op_name, {}))
            if len(selected_ops) == 1 and legacy_params:
                params.update(legacy_params)

            before = modified_content
            modified_content = self._apply_operator(op_name, token, modified_content, params)
            if modified_content != before:
                applied_ops.append(op_name)

        return modified_content, applied_ops

    def apply_multiple(
         self,
         tokens: List[Token],
         content: str,
         max_tokens: Optional[int] = None,
         operators: Optional[Sequence[str]] = None,
         operator_params: Optional[Dict[str, Dict[str, Any]]] = None,
         schedule: Optional[str] = None,
         token_type_plan: Optional[Dict[str, Sequence[str]]] = None,
         **kwargs,
     ) -> Dict[str, str]:

        """Apply perturbations to multiple tokens with per-token operator plans.

        When `operators` is None, the plan is derived from the TSD config.
        """
        if max_tokens is None:
            max_tokens = len(tokens)

        tokens_to_perturb = tokens[:max_tokens]
        if not tokens_to_perturb:
            return {}

        config = self._load_tsd_config()
        schedule = schedule if schedule is not None else config.get("operators", {}).get("schedule", "sequential") or "sequential"
        operator_params = self._merge_operator_params(
            config.get("operator_params", {}),
            operator_params,
        )
        operator_params = self._apply_tsd_defaults(operator_params, config)
        self._last_variant_plan = []
        modified_content = content

        for token in tokens_to_perturb:
            per_token_ops = operators
            if token_type_plan:
                per_token_ops = token_type_plan.get((token.token_type or "").lower(), per_token_ops)

            safe_schedule = schedule or "sequential"
            selected_ops = self._select_operator_sequence(
                 token=token,
                 operators=per_token_ops,
                 schedule=safe_schedule,
                 config=config,
             ) or []

            if not selected_ops:
                continue

            modified_content, applied_ops = self._execute_plan(
                token,
                modified_content,
                selected_ops,
                operator_params,
                {},
            )
            self._last_variant_plan.append(
                {
                    "token": token.text,
                    "token_type": token.token_type,
                    "operators": applied_ops,
                }
            )

        variant_name = f"{self.name}_composed"
        return {variant_name: modified_content}

    def describe_variant(self, variant_name: str) -> Dict[str, Any]:
        """Describe the operators used to create a variant."""
        details = super().describe_variant(variant_name)
        details["operators"] = list(self._operator_dispatch().keys())
        if self._last_variant_plan and variant_name.startswith(self.name):
            details["plan"] = self._last_variant_plan
        return details

    # ========== Main apply method (for backward compatibility) ==========
    
    DEFAULT_OPERATORS: Tuple[str, ...] = (
        "zero_width",
        "homoglyph",
        "control_char",
        "bidi_override",
        "string_fragmentation",
        "joiner",
    )

    def apply(self, token: Token, content: str,
              operators: Optional[Sequence[str]] = None,
              operator_params: Optional[Dict[str, Dict[str, Any]]] = None,
              schedule: Optional[str] = None,
              **kwargs) -> str:
        """Apply tokenization drift perturbation.
        
        Supports chaining multiple operators on a single token. If no operators
        are provided, the TSD config decides the operator plan (default: all).
        
        Args:
            token: Token to perturb
            content: Original content
            operators: Sequence of operator names or a single name; overrides TSD config
            operator_params: Dict mapping operator name (or '*') to parameter dicts
            schedule: 'sequential' or 'random' selection over the chosen operators
            **kwargs: Legacy single-operator parameters; only used when one operator is selected
            
        Returns:
            Modified content
        """
        legacy_operator = kwargs.pop("operator", None)
        legacy_params = kwargs
        config = self._load_tsd_config()
        schedule = schedule if schedule is not None else config.get("operators", {}).get("schedule", "sequential") or "sequential"
        operator_params = self._merge_operator_params(
            config.get("operator_params", {}),
            operator_params,
        )
        operator_params = self._apply_tsd_defaults(operator_params, config)

        if legacy_operator:
            if operators is None:
                operators = [legacy_operator]
            else:
                operators = [legacy_operator, *operators]
        elif operators is None:
            operators = list(self.DEFAULT_OPERATORS)
                
        # 1. 确定要使用哪些操作符
        safe_schedule = schedule or "sequential"
        selected_ops = self._select_operator_sequence(
             token=token,
             operators=operators,
             schedule=safe_schedule,
             config=config,
         ) or []

        if not selected_ops:
            return content

        modified_content, applied_ops = self._execute_plan(
            token,
            content,
            selected_ops,
            operator_params,
            legacy_params,
        )
        self._last_variant_plan = [
            {
                "token": token.text,
                "token_type": token.token_type,
                "operators": applied_ops,
            }
        ]
        return modified_content

    def apply_minimal_perturbation(
        self,
        token: Token,
        content: str,
        prefer_operator: Optional[str] = "zero_width",
        operator_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> str:
        """Apply a constrained, low-visibility perturbation for one token.

        This differs from `apply` by capping operators to one and defaulting
        to the safest zero-width charset when no explicit operator is requested.
        """
        operator_params = operator_params or {}
        preferred_ops: Optional[Sequence[str]] = None
        if prefer_operator:
            preferred_ops = [prefer_operator]

        if prefer_operator in {None, "zero_width"}:
            operator_params = {
                **operator_params,
                "zero_width": {"charset": "minimal"},
                "*": {**operator_params.get("*", {})},
            }

        return self.apply(
            token,
            content,
            operators=preferred_ops,
            operator_params=operator_params,
            schedule="sequential",
        )
