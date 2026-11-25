"""Resolve a requirement text back to its source location.

The LLM is asked to return a location, but a deterministic lookup in the
original document is more reliable. The locator searches the verbatim
requirement text, then finds the nearest preceding Markdown heading or
the PDF page that contains it.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from app.models.requirement import DocumentFormat, SourceDocument, SourceLocation

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class _HeadingMark:
    offset: int
    path: str
    title: str


class SourceLocator:
    """Locate requirement text inside a loaded source document."""

    def __init__(self, document: SourceDocument) -> None:
        self._document = document
        self._headings = (
            _index_markdown_headings(document.text)
            if document.format is DocumentFormat.MARKDOWN
            else []
        )

    def locate(self, text: str) -> SourceLocation | None:
        """Return the source location of ``text``, or ``None`` if not found."""
        needle = text.strip()
        if not needle:
            return None

        if self._document.format is DocumentFormat.PDF:
            return self._locate_pdf(needle)
        return self._locate_markdown(needle)

    def _locate_markdown(self, needle: str) -> SourceLocation | None:
        offset = self._document.text.find(needle)
        if offset < 0:
            offset = _find_normalized_offset(self._document.text, needle)
        if offset is None or offset < 0:
            return None

        index = bisect_right(
            [heading.offset for heading in self._headings], offset
        ) - 1
        if index < 0:
            return SourceLocation()
        heading = self._headings[index]
        return SourceLocation(section=heading.path, heading=heading.title)

    def _locate_pdf(self, needle: str) -> SourceLocation | None:
        normalized_needle = _normalize(needle)
        for page_number, page_text in enumerate(self._document.pages, start=1):
            if needle in page_text:
                return SourceLocation(page=page_number)
            if normalized_needle in _normalize(page_text):
                return SourceLocation(page=page_number)
        return None


def _index_markdown_headings(text: str) -> list[_HeadingMark]:
    """Build an ordered index of headings with their character offsets."""
    marks: list[_HeadingMark] = []
    stack: list[tuple[int, str]] = []
    offset = 0

    for line in text.splitlines(keepends=True):
        match = _HEADING_RE.match(line.rstrip("\r\n"))
        if match is not None:
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            marks.append(
                _HeadingMark(
                    offset=offset,
                    path=" > ".join(item for _, item in stack),
                    title=title,
                )
            )
        offset += len(line)
    return marks


def _find_normalized_offset(text: str, needle: str) -> int | None:
    """Fallback search that ignores case, whitespace and Markdown markers."""
    normalized_needle = _normalize_markdown(needle)
    for start, paragraph in _iter_paragraphs(text):
        if normalized_needle in _normalize_markdown(paragraph):
            return start
    return None


def _iter_paragraphs(text: str) -> list[tuple[int, str]]:
    """Split text into blank-line-separated paragraphs with start offsets."""
    paragraphs: list[tuple[int, str]] = []
    buffer: list[str] = []
    start: int | None = None
    offset = 0

    for line in text.splitlines(keepends=True):
        if line.strip():
            if start is None:
                start = offset
            buffer.append(line)
        elif buffer:
            assert start is not None
            paragraphs.append((start, "".join(buffer)))
            buffer = []
            start = None
        offset += len(line)

    if buffer:
        assert start is not None
        paragraphs.append((start, "".join(buffer)))
    return paragraphs


def _normalize(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _normalize_markdown(text: str) -> str:
    """Normalize text for matching, ignoring emphasis/list markers."""
    without_markers = re.sub(r"[*_`]+", "", text)
    return _normalize(without_markers)
