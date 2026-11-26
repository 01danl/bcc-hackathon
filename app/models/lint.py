"""Deterministic lint findings produced before any LLM call.

The linter is deliberately cheap and conservative: it finds obvious empty
templates, stubs, placeholders, vague wording and internal API mismatches
with regular expressions and Markdown table parsing. Its findings are passed
to the auditor as ``pre_detected_flags``; the auditor can confirm or reject
them, but confirmed flags are always kept in the report.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.models.analysis import QualityCriterion, Severity


class LintCategory(str, Enum):
    """Kind of a deterministic defect."""

    VAGUE_WORDING = "vague_wording"
    EMPTY_TEMPLATE = "empty_template"
    STUB_TEXT = "stub_text"
    EMPTY_TABLE_CELL = "empty_table_cell"
    HTTP_STATUS_MISMATCH = "http_status_mismatch"
    HTTP_METHOD_MISMATCH = "http_method_mismatch"
    UNFILLED_PLACEHOLDER = "unfilled_placeholder"


class LintFlag(BaseModel):
    """One deterministic finding with a verbatim quote from the source."""

    flag_id: str = Field(min_length=1)
    category: LintCategory
    severity: Severity
    criterion: QualityCriterion
    message: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description="Verbatim source fragment")
    recommendation: str = Field(min_length=1)
    requirement_id: str | None = Field(
        default=None,
        description="Internal REQ-ID when the flag belongs to one requirement",
    )
    related_req_ids: list[str] = Field(default_factory=list)


class LintResult(BaseModel):
    """All flags found in one document, split by their scope."""

    requirement_flags: list[LintFlag] = Field(default_factory=list)
    document_flags: list[LintFlag] = Field(default_factory=list)

    @property
    def all_flags(self) -> list[LintFlag]:
        """Every flag in report order: document flags, then requirements."""
        return [*self.document_flags, *self.requirement_flags]

    @property
    def suspicious_ids(self) -> set[str]:
        """Requirements that must reach the deep auditor because of a flag."""
        return {
            flag.requirement_id
            for flag in self.requirement_flags
            if flag.requirement_id
        }

    def flags_for_requirement(self, requirement_id: str) -> list[LintFlag]:
        """Return requirement-level flags bound to ``requirement_id``."""
        return [
            flag
            for flag in self.requirement_flags
            if flag.requirement_id == requirement_id
        ]
