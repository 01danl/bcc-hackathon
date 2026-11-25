"""Pydantic data models for source documents and requirements.

These models are the shared vocabulary between pipeline stages: ingestion
produces a :class:`SourceDocument`, extraction produces a
:class:`RequirementCollection`, and later stages (LLM analysis, report
generation) will consume them.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class DocumentFormat(str, Enum):
    """Formats supported by the ingestion stage."""

    PDF = "pdf"
    MARKDOWN = "markdown"


class RequirementCategory(str, Enum):
    """Coarse category of a requirement.

    The LLM classifies requirements into the first seven values. The
    remaining values are kept for backwards compatibility with documents
    and code written for the earlier placeholder extractor.
    """

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    BUSINESS = "business"
    INTERFACE = "interface"
    INTEGRATION = "integration"
    CONSTRAINT = "constraint"
    DATA = "data"
    ACCEPTANCE = "acceptance"
    UI = "ui"
    OTHER = "other"


class SourceLocation(BaseModel):
    """Where a fragment comes from inside the source document.

    Markdown documents usually provide ``section`` (a heading path); PDF
    documents provide ``page``. ``heading`` is an optional extra heading
    returned by the LLM when it is more precise than the section path.
    """

    page: int | None = Field(default=None, ge=1, description="1-based page number")
    section: str | None = Field(default=None, description="Section heading path")
    heading: str | None = Field(default=None, description="Nearest heading, if any")


class SourceDocument(BaseModel):
    """A requirements document loaded from disk and ready for processing."""

    path: Path
    format: DocumentFormat
    text: str = Field(description="Full extracted text")
    pages: list[str] = Field(
        default_factory=list,
        description="Per-page text; filled for PDF documents only",
    )

    @property
    def page_count(self) -> int:
        """Number of pages with extracted text (0 for Markdown)."""
        return len(self.pages)


class Requirement(BaseModel):
    """A single requirement extracted from a source document."""

    id: str = Field(min_length=1, description="Internal unique id, e.g. 'REQ-001'")
    original_id: str | None = Field(
        default=None,
        description="Identifier used in the source document, if any",
    )
    text: str = Field(min_length=1, description="Requirement text as written")
    category: RequirementCategory = Field(
        default=RequirementCategory.OTHER,
    )
    location: SourceLocation = Field(default_factory=SourceLocation)


class RequirementCollection(BaseModel):
    """All requirements extracted from a single document."""

    requirements: list[Requirement] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.requirements)

    def by_category(self) -> dict[RequirementCategory, list[Requirement]]:
        """Group requirements by category (only non-empty categories)."""
        grouped: dict[RequirementCategory, list[Requirement]] = {}
        for requirement in self.requirements:
            grouped.setdefault(requirement.category, []).append(requirement)
        return grouped

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> RequirementCollection:
        """Reject collections with duplicated internal ids."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for requirement in self.requirements:
            if requirement.id in seen:
                duplicates.add(requirement.id)
            seen.add(requirement.id)
        if duplicates:
            raise ValueError(
                f"Duplicate requirement ids: {', '.join(sorted(duplicates))}"
            )
        return self
