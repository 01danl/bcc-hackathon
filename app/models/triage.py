"""Models for the cheap triage stage that routes requirements.

The extractor model performs a fast first pass over every requirement and
marks only those that deserve a deep (and expensive) auditor review. The
decision is a routing hint, not a verdict: statuses and the final verdict
are still computed deterministically in Python.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


class TriageRisk(str, Enum):
    """How likely a requirement is to contain a real defect."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class LLMTriageItem(BaseModel):
    """One requirement the triage model wants the auditor to inspect."""

    requirement_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    risk: TriageRisk = TriageRisk.MEDIUM

    @field_validator("requirement_id", "reason", mode="before")
    @classmethod
    def _clean_strings(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("risk", mode="before")
    @classmethod
    def _normalize_risk(cls, value: object) -> object:
        if isinstance(value, TriageRisk):
            return value
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in TriageRisk.__members__:
                return TriageRisk[normalized]
        return TriageRisk.MEDIUM


class LLMTriageResponse(BaseModel):
    """Top-level JSON object returned by the triage prompt."""

    suspicious: list[LLMTriageItem] = Field(default_factory=list)
    summary: str = ""

    @model_validator(mode="before")
    @classmethod
    def _wrap_bare_list(cls, data: object) -> object:
        """Accept a bare array as a fallback for imperfect models."""
        if isinstance(data, list):
            return {"suspicious": data}
        return data

    @field_validator("summary", mode="before")
    @classmethod
    def _clean_summary(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value
