"""Heading-aware chunking of requirement documents.

Large documents are split into overlapping fragments before they are sent
to the LLM. The chunker works on whole lines/paragraphs, keeps the
Markdown heading path in the location metadata and never splits a line
unless a single line is larger than the configured limit.

Every chunk is a verbatim slice of the source (overlap repeats the tail
of the previous chunk), so an evidence fragment can still be traced back
to the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.requirement import DocumentFormat, SourceDocument, SourceLocation

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")


@dataclass(frozen=True)
class DocumentChunk:
    """A verbatim document fragment with its source location."""

    index: int
    text: str
    location: SourceLocation


@dataclass(frozen=True)
class _Section:
    """A document section before chunk-level splitting."""

    body: str
    location: SourceLocation
    heading_line: str | None = None


class DocumentChunker:
    """Split Markdown/PDF documents into overlapping, line-safe chunks."""

    def __init__(self, max_chars: int = 6000, overlap_chars: int = 600) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be non-negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, document: SourceDocument) -> list[DocumentChunk]:
        """Return all chunks of ``document`` in reading order."""
        if document.format is DocumentFormat.PDF:
            return self._split_pdf(document)

        units: list[tuple[str, SourceLocation]] = []
        for section in self._markdown_sections(document):
            # Keep the original heading line inside the chunk so the model
            # can see section boundaries when adjacent sections are combined.
            if section.heading_line:
                units.append((section.heading_line, section.location))
            for unit in self._section_units(section.body):
                units.append((unit, section.location))

        return self._chunk_units(units)

    def _split_pdf(self, document: SourceDocument) -> list[DocumentChunk]:
        """Chunk each PDF page separately to keep page numbers exact."""
        chunks: list[DocumentChunk] = []
        for section in self._pdf_sections(document):
            units = [
                (unit, section.location)
                for unit in self._section_units(section.body)
            ]
            for chunk in self._chunk_units(units):
                chunks.append(
                    DocumentChunk(
                        index=len(chunks) + 1,
                        text=chunk.text,
                        location=chunk.location,
                    )
                )
        return chunks

    def _chunk_units(
        self, units: list[tuple[str, SourceLocation]]
    ) -> list[DocumentChunk]:
        """Group units into overlapping chunks, never splitting a unit."""
        if not units:
            return []

        chunks: list[DocumentChunk] = []
        current: list[tuple[str, SourceLocation]] = []
        current_length = 0

        for unit in units:
            separator = 2 if current else 0
            if current and current_length + separator + len(unit[0]) > self.max_chars:
                chunks.append(self._make_chunk(len(chunks) + 1, current))
                current = self._overlap_tail(current)
                current_length = _units_length(current)
                while current and current_length + 2 + len(unit[0]) > self.max_chars:
                    current.pop(0)
                    current_length = _units_length(current)

            current.append(unit)
            current_length = _units_length(current)

        if current:
            chunks.append(self._make_chunk(len(chunks) + 1, current))
        return [chunk for chunk in chunks if chunk.text.strip()]

    @staticmethod
    def _make_chunk(index: int, units: list[tuple[str, SourceLocation]]) -> DocumentChunk:
        return DocumentChunk(
            index=index,
            text="\n\n".join(text for text, _ in units),
            location=units[0][1],
        )

    def _markdown_sections(self, document: SourceDocument) -> list[_Section]:
        """Split Markdown by headings while keeping a heading path."""
        sections: list[_Section] = []
        heading_stack: list[tuple[int, str]] = []
        current_lines: list[str] = []
        current_heading_path: str | None = None
        current_heading_line: str | None = None

        def flush() -> None:
            body = "\n".join(current_lines).strip("\n")
            if body.strip():
                sections.append(
                    _Section(
                        body=body,
                        location=SourceLocation(section=current_heading_path),
                        heading_line=current_heading_line,
                    )
                )

        for line in document.text.splitlines():
            heading = _HEADING_RE.match(line)
            if heading is None:
                current_lines.append(line)
                continue

            flush()
            current_lines = []
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_heading_path = " > ".join(title for _, title in heading_stack)
            current_heading_line = line

        flush()
        return sections

    def _pdf_sections(self, document: SourceDocument) -> list[_Section]:
        """Treat each PDF page as a separate section."""
        sections: list[_Section] = []
        for page_number, page_text in enumerate(document.pages, start=1):
            body = page_text.strip("\n")
            if body.strip():
                sections.append(
                    _Section(
                        body=body,
                        location=SourceLocation(page=page_number),
                    )
                )
        return sections

    def _section_units(self, body: str) -> list[str]:
        """Break a section into paragraph-sized units, then into lines."""
        units: list[str] = []
        for paragraph in _PARAGRAPH_RE.split(body):
            cleaned = paragraph.strip("\n")
            if not cleaned.strip():
                continue
            if len(cleaned) <= self.max_chars:
                units.append(cleaned)
                continue

            for line in cleaned.splitlines():
                if not line.strip():
                    continue
                if len(line) <= self.max_chars:
                    units.append(line)
                else:
                    units.extend(self._split_long_line(line))
        return units

    def _split_long_line(self, line: str) -> list[str]:
        """Split a single oversized line at sentence boundaries, then hard-cut."""
        pieces: list[str] = []
        for sentence in _SENTENCE_RE.split(line):
            if len(sentence) <= self.max_chars:
                pieces.append(sentence)
                continue
            start = 0
            while start < len(sentence):
                pieces.append(sentence[start : start + self.max_chars])
                start += self.max_chars
        return [piece for piece in pieces if piece.strip()]

    def _overlap_tail(
        self, units: list[tuple[str, SourceLocation]]
    ) -> list[tuple[str, SourceLocation]]:
        """Return the trailing units that fit into the overlap budget."""
        if self.overlap_chars <= 0:
            return []

        tail: list[tuple[str, SourceLocation]] = []
        length = 0
        for unit in reversed(units):
            added = len(unit[0]) + (2 if tail else 0)
            if tail and length + added > self.overlap_chars:
                break
            tail.insert(0, unit)
            length += added
        return tail


def _units_length(units: list[tuple[str, SourceLocation]]) -> int:
    """Length of ``"\\n\\n".join(text for text, _ in units)``."""
    if not units:
        return 0
    return sum(len(text) for text, _ in units) + 2 * (len(units) - 1)
