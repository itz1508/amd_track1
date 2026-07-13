"""Shared deterministic arithmetic prompt detection."""

from __future__ import annotations

import re
from typing import Optional


_OPERATOR_PATTERN = re.compile(r"(\*\*|[+\-*/%^])")
_DIGIT_PATTERN = re.compile(r"\d")
_DIRECT_CALCULATION_PATTERN = re.compile(
    r"^\s*(?:calculate|compute|evaluate|solve|what\s+is|what's|how\s+much\s+is)\s*:?\s*(.+?)\s*\??\s*$",
    flags=re.IGNORECASE,
)
_BARE_EXPRESSION_PATTERN = re.compile(r"^[\d\s.+\-*/%^()]+=?\s*$")
_SAFE_FUNCTION_EXPRESSION_PATTERN = re.compile(r"^[\d\s.+\-*/%^(),]*(?:sqrt[\d\s.+\-*/%^(),]*)+$", re.IGNORECASE)
_BLOCKED_CONTEXT_PATTERN = re.compile(
    r"\b(hash|sha|sha-\d+|http|endpoint|url|route|debug|diagnose|error|code|line|program|function)\b",
    flags=re.IGNORECASE,
)


def extract_arithmetic_expression(prompt: str) -> Optional[str]:
    """Extract a calculator-safe arithmetic expression from a prompt."""
    if not prompt:
        return None

    stripped = prompt.strip()

    # Natural-language percentages are policy-sensitive and always remote.
    if "%" in stripped and re.search(r"\bof\b", stripped, re.IGNORECASE):
        return None

    bare_expression = stripped.rstrip("=").strip()
    if _is_calculator_safe(bare_expression):
        return bare_expression

    direct_match = _DIRECT_CALCULATION_PATTERN.match(stripped)
    if not direct_match:
        return None

    expression_text = direct_match.group(1)
    if _BLOCKED_CONTEXT_PATTERN.search(expression_text):
        return None
    if re.search(r"\b(rounded|round|nearest|markup|discount|average speed|travels?|slows?|entire trip)\b", expression_text, re.IGNORECASE):
        return None

    expression = expression_text.strip(" \t.:,;?!")
    if _is_calculator_safe(expression):
        return expression

    return None


def _is_calculator_safe(expression: str) -> bool:
    """Allow numeric operators and the one evaluator-supported sqrt function."""
    syntax_ok = bool(_BARE_EXPRESSION_PATTERN.fullmatch(expression) or _SAFE_FUNCTION_EXPRESSION_PATTERN.fullmatch(expression))
    return syntax_ok and bool(_DIGIT_PATTERN.search(expression)) and bool(_OPERATOR_PATTERN.search(expression))


def has_arithmetic_expression(prompt: str) -> bool:
    """Return True when the prompt contains a deterministic arithmetic expression."""
    try:
        return extract_arithmetic_expression(prompt) is not None
    except Exception:
        return False
