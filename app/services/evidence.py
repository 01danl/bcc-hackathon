"""Verification that quoted evidence exists in the source document.

Quotes are compared after normalization: Markdown emphasis markers are
ignored, whitespace is collapsed and case is not significant. Models
routinely drop ``**`` or backticks when they quote a document, and a
strict comparison would reject correct findings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_MARKERS_RE = re.compile(r"[*_`]+")


def normalize_for_matching(text: str) -> str:
    """Normalize case, whitespace and Markdown markers for evidence checks."""
    without_markers = _MARKDOWN_MARKERS_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", without_markers).strip().casefold()


def find_missing_quotes(quotes: Iterable[str], source_text: str) -> list[str]:
    """Return the quotes that do not occur in ``source_text`` after normalization.

    A quote that is empty after normalization never counts as found.
    """
    haystack = normalize_for_matching(source_text)
    missing: list[str] = []
    for quote in quotes:
        needle = normalize_for_matching(quote)
        if not needle or needle not in haystack:
            missing.append(quote)
    return missing
