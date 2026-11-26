"""Verification and fuzzy alignment of quoted evidence.

Models rarely hallucinate a quote completely, but they often change a
non-breaking space, a dash, a Markdown marker or one letter. Failing the
whole run (or paying for another expensive reasoning call) because of such a
difference is wasteful. This module first normalizes both strings, then uses
``difflib.SequenceMatcher`` to align a near-match and returns the exact slice
of the source document. A retry is only needed for a real hallucination.

Normalization ignores case, whitespace, Markdown emphasis markers and unifies
dashes, quotes and ``ё``/``е``. The returned evidence is always the original
source text, so the report never contains a rewritten quote.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

DEFAULT_FUZZY_THRESHOLD = 0.92

_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_MARKERS_RE = re.compile(r"[*_`]+")
_DASHES_RE = re.compile(r"[‐‑‒–—―]")
_QUOTES_RE = re.compile(r"[«»“”„‟]")
_YO_RE = re.compile(r"[ёЁ]")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NUMBER_RE = re.compile(r"\d[\d\s.,]*\d|\d")
_ENDPOINT_RE = re.compile(r"(?<![\w])/[a-zA-Z0-9_\-/{}.]+")


@dataclass(frozen=True)
class EvidenceMatch:
    """A quote aligned to a concrete slice of the source document."""

    quote: str
    matched_text: str
    score: float
    exact: bool


def normalize_for_matching(text: str) -> str:
    """Normalize case, whitespace, punctuation and Markdown markers."""
    without_markers = _MARKDOWN_MARKERS_RE.sub("", text)
    unified = _DASHES_RE.sub("-", without_markers)
    unified = _QUOTES_RE.sub('"', unified)
    unified = _YO_RE.sub("е", unified)
    return _WHITESPACE_RE.sub(" ", unified).strip().casefold()


class EvidenceAligner:
    """Align many quotes against one source without normalizing it repeatedly.

    A batch audit checks several quotes against the same document. Keeping the
    normalized document and its index map on the instance makes each check a
    simple substring or bounded fuzzy search.
    """

    def __init__(
        self,
        source_text: str,
        *,
        threshold: float = DEFAULT_FUZZY_THRESHOLD,
    ) -> None:
        self._source_text = source_text if isinstance(source_text, str) else ""
        self._threshold = threshold
        self._haystack, self._source_map = _normalize_with_map(self._source_text)

    def align(self, quote: str) -> EvidenceMatch | None:
        """Return the source slice matching ``quote``, or ``None``."""
        if not isinstance(quote, str):
            return None
        needle, _ = _normalize_with_map(quote)
        if not needle:
            return None
        return _align_normalized(
            quote,
            needle,
            self._source_text,
            self._haystack,
            self._source_map,
            self._threshold,
        )

    def canonicalize(
        self, quotes: Iterable[str]
    ) -> tuple[list[str], list[str]]:
        """Return ``(source_slices, unmatched_quotes)`` with duplicates removed."""
        aligned: list[str] = []
        missing: list[str] = []
        seen: set[str] = set()
        for quote in quotes:
            match = self.align(quote)
            if match is None:
                missing.append(quote)
                continue
            if match.matched_text not in seen:
                seen.add(match.matched_text)
                aligned.append(match.matched_text)
        return aligned, missing

    def missing(self, quotes: Iterable[str]) -> list[str]:
        """Return quotes that cannot be aligned with the source text."""
        _, missing = self.canonicalize(quotes)
        return missing


def align_quote(
    quote: str,
    source_text: str,
    *,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> EvidenceMatch | None:
    """Return the source slice matching ``quote``, or ``None``.

    An exact normalized substring wins immediately. Otherwise a bounded
    fuzzy search around matching blocks and word anchors is used. The
    result is accepted only when the similarity reaches ``threshold``.
    """
    if not isinstance(quote, str) or not isinstance(source_text, str):
        return None
    return EvidenceAligner(source_text, threshold=threshold).align(quote)


def canonicalize_quotes(
    quotes: Iterable[str],
    source_text: str,
    *,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> tuple[list[str], list[str]]:
    """Align quotes and return ``(source_slices, unmatched_quotes)``.

    The source is normalized once for the whole batch. Duplicate aligned
    slices are removed while preserving order.
    """
    return EvidenceAligner(source_text, threshold=threshold).canonicalize(quotes)


def find_missing_quotes(
    quotes: Iterable[str],
    source_text: str,
    *,
    threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> list[str]:
    """Return quotes that cannot be aligned with the source text.

    A quote that is empty after normalization never counts as found.
    """
    return EvidenceAligner(source_text, threshold=threshold).missing(quotes)


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalize ``text`` and remember the original index of every character."""
    normalized: list[str] = []
    indices: list[int] = []
    pending_space = False

    for index, character in enumerate(text):
        if character in "*_`":
            continue
        if character.isspace():
            pending_space = True
            continue
        if pending_space and normalized:
            normalized.append(" ")
            indices.append(index)
        pending_space = False

        replacement = _DASHES_RE.sub("-", character)
        replacement = _QUOTES_RE.sub('"', replacement)
        replacement = _YO_RE.sub("е", replacement)
        for item in replacement.casefold():
            normalized.append(item)
            indices.append(index)

    return "".join(normalized), indices


def _align_normalized(
    quote: str,
    needle: str,
    source_text: str,
    haystack: str,
    source_map: list[int],
    threshold: float,
) -> EvidenceMatch | None:
    """Align an already normalized quote against an already normalized source."""
    if not needle or not haystack:
        return None

    position = haystack.find(needle)
    if position != -1:
        return _build_match(
            quote,
            source_text,
            source_map,
            position,
            position + len(needle),
            1.0,
            exact=True,
        )

    best = _fuzzy_window(needle, haystack)
    if best is None or best[0] < threshold:
        return None
    score, start, end = best
    return _build_match(
        quote, source_text, source_map, start, end, score, exact=False
    )


def _build_match(
    quote: str,
    source_text: str,
    source_map: list[int],
    start: int,
    end: int,
    score: float,
    *,
    exact: bool,
) -> EvidenceMatch | None:
    """Map normalized indices back to a verbatim source slice."""
    if start < 0 or end <= start or end > len(source_map):
        return None
    original_start = source_map[start]
    original_end = source_map[end - 1] + 1
    matched = source_text[original_start:original_end].strip()
    if not matched:
        return None
    if not exact and not _fuzzy_is_safe(quote, matched):
        return None
    return EvidenceMatch(
        quote=quote, matched_text=matched, score=score, exact=exact
    )


def _fuzzy_is_safe(quote: str, matched: str) -> bool:
    """Reject a near-match that changed a number or an API endpoint.

    A one-letter typo is harmless, but a different amount, threshold or URL
    is a different statement. Canonicalizing it to the source would hide a
    real hallucination, so those tokens must survive the fuzzy match.
    """
    quote_numbers = {_normalize_number(item) for item in _NUMBER_RE.findall(quote)}
    matched_numbers = {
        _normalize_number(item) for item in _NUMBER_RE.findall(matched)
    }
    if not quote_numbers.issubset(matched_numbers):
        return False

    quote_endpoints = {
        item.casefold() for item in _ENDPOINT_RE.findall(quote)
    }
    matched_endpoints = {
        item.casefold() for item in _ENDPOINT_RE.findall(matched)
    }
    return quote_endpoints.issubset(matched_endpoints)


def _normalize_number(value: str) -> str:
    """Normalize ``1 000,50`` and ``1000.50`` to the same token."""
    return value.replace(" ", "").replace(",", ".").strip(".")


def _fuzzy_window(
    needle: str, haystack: str
) -> tuple[float, int, int] | None:
    """Find the best matching source window around anchors and matching blocks.

    The search is deliberately bounded: a handful of candidate starts around
    the longest matching blocks and the longest quote words, and a few window
    lengths around the quote length. This keeps a large document from turning
    one evidence check into a quadratic scan over the whole text.
    """
    matcher = SequenceMatcher(None, needle, haystack, autojunk=False)
    blocks = [
        block for block in matcher.get_matching_blocks() if block.size >= 3
    ]
    if not blocks:
        return None

    slack = max(8, len(needle) // 5)
    step = max(1, slack // 6)
    starts: set[int] = set()

    # Long matching blocks give the most reliable approximate position.
    for block in [*blocks[:6], *blocks[-6:]]:
        approximate = block.b - block.a
        for delta in range(-slack, slack + 1, step):
            starts.add(approximate + delta)

    # A long word is a cheap anchor when the quote has several small edits.
    for token in sorted(_WORD_RE.findall(needle), key=len, reverse=True)[:3]:
        if len(token) < 4:
            continue
        offset = needle.find(token)
        search_from = 0
        for _ in range(8):
            found = haystack.find(token, search_from)
            if found == -1:
                break
            approximate = found - offset
            for delta in (-2, 0, 2):
                starts.add(approximate + delta)
            search_from = found + 1

    length_delta = max(2, len(needle) // 10)
    lengths = {
        len(needle),
        len(needle) - 2,
        len(needle) + 2,
        len(needle) - length_delta,
        len(needle) + length_delta,
    }
    lengths = {length for length in lengths if length >= max(3, len(needle) // 2)}

    best: tuple[float, int, int] | None = None
    for start in sorted(starts)[:120]:
        if start < 0 or start >= len(haystack):
            continue
        for length in sorted(lengths):
            end = start + length
            if end > len(haystack):
                continue
            score = SequenceMatcher(
                None, needle, haystack[start:end], autojunk=False
            ).ratio()
            if best is None or score > best[0]:
                best = (score, start, end)

    return best
