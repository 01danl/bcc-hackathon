"""Pydantic models for the audit of a requirement set as a whole.

As with the per-requirement analysis, two layers are used:
``LLMSetAuditResponse`` validates the raw JSON returned by the auditor
model, and :class:`RequirementSetAudit` adds the final verdict. The
verdict is derived deterministically in Python from finding severities
and the per-requirement statuses; it is never taken from the model.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from app.models.analysis import Severity, normalize_severity


def _string_list(value: object) -> object:
    """Coerce ``None`` or a bare string into a list of non-empty strings.

    Other values are returned untouched so that Pydantic reports them
    against the ``list[str]`` annotation.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return value


class SetFindingCategory(str, Enum):
    """Kind of a defect found at the level of the whole document."""

    INTEGRATION_MISMATCH = "integration_mismatch"
    CONTRADICTION = "contradiction"
    DUPLICATE = "duplicate"
    MISSING_REQUIREMENT = "missing_requirement"
    INCOMPLETE_SECTION = "incomplete_section"
    GOAL_COVERAGE = "goal_coverage"


class VerdictStatus(str, Enum):
    """Whether the document can go to development and testing."""

    READY = "READY"
    NEEDS_REVISION = "NEEDS_REVISION"
    BLOCKED = "BLOCKED"


class SetFinding(BaseModel):
    """One defect returned by the set audit prompt."""

    category: SetFindingCategory
    severity: Severity
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    related_req_ids: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes from the source document",
    )

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, value: object) -> object:
        if isinstance(value, SetFindingCategory):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in SetFindingCategory._value2member_map_:
                return SetFindingCategory(normalized)
        raise ValueError(
            "category must be one of: integration_mismatch, contradiction, "
            "duplicate, missing_requirement, incomplete_section, goal_coverage"
        )

    @field_validator("severity", mode="before")
    @classmethod
    def _normalize_severity(cls, value: object) -> object:
        return normalize_severity(value)

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split())
        return value

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("related_req_ids", "evidence", mode="before")
    @classmethod
    def _as_string_list(cls, value: object) -> object:
        return _string_list(value)


class LLMSetAuditResponse(BaseModel):
    """Top-level JSON object returned by the set audit prompt."""

    summary: str
    goal_coverage: str
    findings: list[SetFinding] = Field(default_factory=list)
    priority_actions: list[str] = Field(default_factory=list)

    @field_validator("summary", "goal_coverage", mode="before")
    @classmethod
    def _clean_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("priority_actions", mode="before")
    @classmethod
    def _as_string_list(cls, value: object) -> object:
        return _string_list(value)


class RequirementSetAudit(LLMSetAuditResponse):
    """Audit result for the whole requirement set, with the final verdict."""

    verdict: VerdictStatus
    verdict_reason: str = Field(min_length=1)
    excluded_findings: int = Field(
        default=0,
        description=(
            "Findings dropped because none of their quotes "
            "were found in the document"
        ),
    )
