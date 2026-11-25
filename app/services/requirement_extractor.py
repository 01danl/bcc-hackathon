"""Requirement extraction service.

This is a deliberate placeholder. The heuristic treats every Markdown
list item (``* item``, ``- item``, ``1. item``) as a requirement
candidate, assigns sequential internal ids and marks everything as
``OTHER``. It parses no identifiers and performs no classification —
an LLM-based extractor will replace this implementation later.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from app.models.requirement import (
    DocumentFormat,
    Requirement,
    RequirementCollection,
    SourceDocument,
    SourceLocation,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST_ITEM_RE = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)")


class RequirementExtractor:
    """Extract requirement candidates from a loaded document.

    Kept separate from document loading: the loader only produces text,
    this service decides what counts as a requirement candidate.
    """

    id_prefix = "REQ"

    def extract(self, document: SourceDocument) -> RequirementCollection:
        """Extract requirement candidates from ``document``.

        Args:
            document: loaded source document.

        Returns:
            Collection of candidates with sequential internal ids.
        """
        requirements: list[Requirement] = []
        for line, location in self._iter_lines(document):
            if _LIST_ITEM_RE.match(line) is None:
                continue
            text = self._clean_text(line)
            if not text:
                continue
            requirements.append(
                Requirement(
                    id=self._make_id(len(requirements) + 1),
                    text=text,
                    location=location,
                )
            )
        return RequirementCollection(requirements=requirements)

    def _make_id(self, number: int) -> str:
        """Build a sequential internal id such as ``REQ-007``."""
        return f"{self.id_prefix}-{number:03d}"

    def _iter_lines(
        self, document: SourceDocument
    ) -> Iterator[tuple[str, SourceLocation]]:
        """Yield every document line together with its source location.

        Markdown lines are tagged with the closest preceding heading;
        PDF lines are tagged with their 1-based page number.
        """
        if document.format is DocumentFormat.PDF:
            for page_number, page_text in enumerate(document.pages, start=1):
                for line in page_text.splitlines():
                    yield line, SourceLocation(page=page_number)
            return

        section: str | None = None
        for line in document.text.splitlines():
            heading = _HEADING_RE.match(line)
            if heading is not None:
                section = heading.group(2)
                continue
            yield line, SourceLocation(section=section)

    @staticmethod
    def _clean_text(line: str) -> str:
        """Strip list markers and simple bold markup from a line."""
        text = _LIST_ITEM_RE.sub("", line).strip()
        return text.replace("**", "").strip()
