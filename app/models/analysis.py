"""Pydantic models for LLM extraction and requirement quality analysis.

Two layers of models are used deliberately:

* ``LLM*`` models validate the raw JSON returned by the model.
* The remaining models are the application-level results consumed by the
  CLI and written to the intermediate JSON files.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.requirement import RequirementCategory, SourceLocation


class RequirementStatus(str, Enum):
    """Overall quality status of one requirement."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class Severity(str, Enum):
    """Severity of a single finding."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class QualityCriterion(str, Enum):
    """Quality criteria from ISO/IEC/IEEE 29148 and IEEE Std 830."""

    CLARITY = "clarity"
    VERIFIABILITY = "verifiability"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    NECESSITY = "necessity"
    FEASIBILITY = "feasibility"


class FindingOrigin(str, Enum):
    """Where a finding came from: the auditor model or the static linter."""

    LLM = "llm"
    STATIC_LINTER = "static_linter"


_CATEGORY_ALIASES = {
    "functional": RequirementCategory.FUNCTIONAL,
    "non_functional": RequirementCategory.NON_FUNCTIONAL,
    "nonfunctional": RequirementCategory.NON_FUNCTIONAL,
    "non_functional_requirement": RequirementCategory.NON_FUNCTIONAL,
    "business": RequirementCategory.BUSINESS,
    "interface": RequirementCategory.INTERFACE,
    "integration": RequirementCategory.INTEGRATION,
    "constraint": RequirementCategory.CONSTRAINT,
    "data": RequirementCategory.DATA,
    "acceptance": RequirementCategory.ACCEPTANCE,
    "acceptance_criteria": RequirementCategory.ACCEPTANCE,
    "ui": RequirementCategory.UI,
    "ux": RequirementCategory.UI,
    "other": RequirementCategory.OTHER,
}


def normalize_severity(value: object) -> Severity:
    """Accept any case spelling of a severity; unknown values become MEDIUM."""
    if isinstance(value, Severity):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in Severity.__members__:
            return Severity[normalized]
    return Severity.MEDIUM


class LLMExtractedRequirement(BaseModel):
    """One requirement as returned by the extraction prompt."""

    original_id: str | None = Field(
        default=None, description="Identifier used in the source document, if any"
    )
    text: str = Field(min_length=1, description="Verbatim requirement text")
    category: RequirementCategory = RequirementCategory.OTHER
    source_location: SourceLocation = Field(default_factory=SourceLocation)

    @field_validator("text", mode="before")
    @classmethod
    def _clean_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("original_id", mode="before")
    @classmethod
    def _clean_original_id(cls, value: object) -> object:
        if isinstance(value, str):
            cleaned = value.strip()
            return cleaned or None
        return value

    @field_validator("category", mode="before")
    @classmethod
    def _normalize_category(cls, value: object) -> object:
        if isinstance(value, RequirementCategory):
            return value
        if not isinstance(value, str):
            return RequirementCategory.OTHER
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        return _CATEGORY_ALIASES.get(normalized, RequirementCategory.OTHER)

    @field_validator("source_location", mode="before")
    @classmethod
    def _normalize_location(cls, value: object) -> object:
        if value is None:
            return {}
        return value


class LLMExtractionResponse(BaseModel):
    """Top-level JSON object returned by the extraction prompt."""

    requirements: list[LLMExtractedRequirement] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _wrap_bare_list(cls, data: object) -> object:
        """Accept a bare JSON array as a fallback for imperfect models."""
        if isinstance(data, list):
            return {"requirements": data}
        return data


def _as_string_list(value: object) -> object:
    """Coerce ``None`` or a bare string into a list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, list):
        return [
            item.strip() for item in value if isinstance(item, str) and item.strip()
        ]
    return value


class _FindingValidators(BaseModel):
    """Shared validators for model-returned findings.

    The base model has no fields of its own; ``check_fields=False`` lets the
    same validators serve both requirement-level and batch-level models.
    """

    @field_validator("severity", mode="before", check_fields=False)
    @classmethod
    def _normalize_severity(cls, value: object) -> object:
        return normalize_severity(value)

    @field_validator("criterion", mode="before", check_fields=False)
    @classmethod
    def _normalize_criterion(cls, value: object) -> object:
        if isinstance(value, QualityCriterion):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in QualityCriterion._value2member_map_:
                return QualityCriterion(normalized)
        raise ValueError(
            "criterion must be one of: clarity, verifiability, completeness, "
            "consistency, necessity, feasibility"
        )

    @field_validator("explanation", "evidence", "recommendation", mode="before", check_fields=False)
    @classmethod
    def _clean_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class LLMFinding(_FindingValidators):
    """One defect returned by the analysis prompt.

    Fields are ordered causally: quote first, then the conflict analysis,
    then the criterion and only then severity. Models generate JSON left to
    right, so they reason before judging instead of guessing a severity and
    justifying it afterwards.
    """

    evidence: str = Field(min_length=1, description="Exact fragment from the source")
    explanation: str = Field(min_length=1)
    criterion: QualityCriterion
    severity: Severity
    recommendation: str = Field(min_length=1)
    origin: FindingOrigin = FindingOrigin.LLM


class LLMAnalysisResponse(BaseModel):
    """Top-level JSON object returned by the legacy single-item prompt."""

    findings: list[LLMFinding] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _wrap_bare_list(cls, data: object) -> object:
        if isinstance(data, list):
            return {"findings": data}
        return data


class LLMBatchFinding(_FindingValidators):
    """A finding bound to a concrete ``REQ-ID`` in a batch answer."""

    requirement_id: str = Field(min_length=1)
    evidence: str = Field(min_length=1, description="Exact fragment from the source")
    explanation: str = Field(min_length=1)
    criterion: QualityCriterion
    severity: Severity
    recommendation: str = Field(min_length=1)
    origin: FindingOrigin = FindingOrigin.LLM

    @field_validator("requirement_id", mode="before")
    @classmethod
    def _clean_requirement_id(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class LLMBatchAnalysisResponse(BaseModel):
    """Findings for a whole batch of up to twenty requirements."""

    findings: list[LLMBatchFinding] = Field(default_factory=list)
    rejected_flag_ids: list[str] = Field(
        default_factory=list,
        description="Pre-detected lint flags the model judged to be false positives",
    )

    @model_validator(mode="before")
    @classmethod
    def _wrap_bare_list(cls, data: object) -> object:
        if isinstance(data, list):
            return {"findings": data}
        return data

    @field_validator("rejected_flag_ids", mode="before")
    @classmethod
    def _as_string_list(cls, value: object) -> object:
        return _as_string_list(value)


class RequirementFinding(BaseModel):
    """A validated finding with the requirement id attached."""

    requirement_id: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    criterion: QualityCriterion
    severity: Severity
    recommendation: str = Field(min_length=1)
    status: RequirementStatus
    origin: FindingOrigin = FindingOrigin.LLM


class RequirementAnalysis(BaseModel):
    """Analysis result for one requirement.

    ``status`` is derived from findings, never trusted blindly from the
    model: no findings means PASS, HIGH/CRITICAL means FAIL, and any other
    finding means WARNING.
    """

    requirement_id: str = Field(min_length=1)
    status: RequirementStatus
    findings: list[RequirementFinding] = Field(default_factory=list)

    @classmethod
    def from_findings(
        cls,
        requirement_id: str,
        findings: Sequence[LLMFinding | LLMBatchFinding],
    ) -> RequirementAnalysis:
        built: list[RequirementFinding] = []
        for finding in findings:
            finding_status = (
                RequirementStatus.FAIL
                if finding.severity in {Severity.HIGH, Severity.CRITICAL}
                else RequirementStatus.WARNING
            )
            payload = finding.model_dump()
            # Batch findings already carry their own id; the owning id is the
            # one we derive from, so never let the model override it.
            payload.pop("requirement_id", None)
            built.append(
                RequirementFinding(
                    requirement_id=requirement_id,
                    status=finding_status,
                    **payload,
                )
            )

        if any(finding.status is RequirementStatus.FAIL for finding in built):
            overall = RequirementStatus.FAIL
        elif built:
            overall = RequirementStatus.WARNING
        else:
            overall = RequirementStatus.PASS
        return cls(requirement_id=requirement_id, status=overall, findings=built)

    def severity_counts(self) -> dict[Severity, int]:
        """Count findings by severity."""
        counts: dict[Severity, int] = {}
        for finding in self.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
        return counts


class RequirementAnalysisCollection(BaseModel):
    """All per-requirement analyses for one document."""

    analyses: list[RequirementAnalysis] = Field(default_factory=list)
    excluded_findings: int = Field(
        default=0,
        description="Findings dropped because their quote or requirement id was invalid",
    )
    skipped_requirements: int = Field(
        default=0,
        description="Requirements not deep-audited because triage marked them as low risk",
    )
    deep_audited_requirements: int = Field(
        default=0,
        description="Requirements sent to the auditor model",
    )
    static_findings: int = Field(
        default=0,
        description="Findings added from the deterministic static linter",
    )
    triage_summary: str = Field(
        default="",
        description="Short human-readable summary returned by the triage stage",
    )
    triage_fallback_used: bool = Field(
        default=False,
        description="Triage failed and every requirement was audited",
    )

    def by_status(self) -> dict[RequirementStatus, int]:
        """Count requirements by overall status."""
        counts: dict[RequirementStatus, int] = {}
        for analysis in self.analyses:
            counts[analysis.status] = counts.get(analysis.status, 0) + 1
        return counts

    @property
    def finding_count(self) -> int:
        """Total number of findings across all requirements."""
        return sum(len(analysis.findings) for analysis in self.analyses)
