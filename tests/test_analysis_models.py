"""Unit tests for LLM response validation and analysis status derivation."""

import pytest
from pydantic import ValidationError

from app.models.analysis import (
    FindingOrigin,
    LLMBatchAnalysisResponse,
    LLMBatchFinding,
    LLMAnalysisResponse,
    LLMExtractionResponse,
    LLMFinding,
    QualityCriterion,
    RequirementAnalysis,
    RequirementStatus,
    Severity,
)
from app.models.set_audit import SetFinding, SetFindingCategory


def _finding(
    *,
    severity: Severity = Severity.MEDIUM,
    criterion: QualityCriterion = QualityCriterion.CLARITY,
) -> LLMFinding:
    return LLMFinding(
        severity=severity,
        criterion=criterion,
        explanation="Формулировка допускает несколько трактовок.",
        evidence="Система должна работать быстро.",
        recommendation="Указать измеримый предел времени отклика.",
    )


class TestExtractionResponse:
    def test_bare_array_is_wrapped(self) -> None:
        response = LLMExtractionResponse.model_validate(
            [
                {
                    "text": "Система должна хранить порог.",
                    "category": "functional",
                }
            ]
        )

        assert len(response.requirements) == 1
        assert response.requirements[0].original_id is None

    def test_category_aliases_are_normalized(self) -> None:
        response = LLMExtractionResponse.model_validate(
            {
                "requirements": [
                    {"text": "A", "category": "non-functional"},
                    {"text": "B", "category": "nonfunctional"},
                    {"text": "C", "category": "integration"},
                ]
            }
        )

        assert [item.category.value for item in response.requirements] == [
            "non_functional",
            "non_functional",
            "integration",
        ]

    def test_unknown_category_becomes_other(self) -> None:
        response = LLMExtractionResponse.model_validate(
            {"requirements": [{"text": "A", "category": "weird-category"}]}
        )

        assert response.requirements[0].category.value == "other"

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMExtractionResponse.model_validate(
                {"requirements": [{"text": "   "}]}
            )

    def test_null_location_is_accepted(self) -> None:
        response = LLMExtractionResponse.model_validate(
            {"requirements": [{"text": "A", "source_location": None}]}
        )

        assert response.requirements[0].source_location.page is None


class TestFindingValidation:
    def test_empty_evidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMFinding(
                severity="HIGH",
                criterion="clarity",
                explanation="Проблема.",
                evidence="",
                recommendation="Исправить.",
            )

    def test_invalid_criterion_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMFinding(
                severity="HIGH",
                criterion="beauty",
                explanation="Проблема.",
                evidence="Текст.",
                recommendation="Исправить.",
            )

    def test_severity_and_criterion_are_case_insensitive(self) -> None:
        finding = LLMFinding.model_validate(
            {
                "severity": "critical",
                "criterion": "Verifiability",
                "explanation": "Проблема.",
                "evidence": "Текст.",
                "recommendation": "Исправить.",
            }
        )

        assert finding.severity is Severity.CRITICAL
        assert finding.criterion is QualityCriterion.VERIFIABILITY

    def test_analysis_response_accepts_bare_list(self) -> None:
        response = LLMAnalysisResponse.model_validate(
            [
                {
                    "severity": "LOW",
                    "criterion": "clarity",
                    "explanation": "Проблема.",
                    "evidence": "Текст.",
                    "recommendation": "Исправить.",
                }
            ]
        )

        assert len(response.findings) == 1


class TestBatchFindingValidation:
    def test_requirement_id_is_required(self) -> None:
        with pytest.raises(ValidationError):
            LLMBatchFinding(
                severity="HIGH",
                criterion="clarity",
                explanation="Проблема.",
                evidence="Текст.",
                recommendation="Исправить.",
            )

    def test_batch_response_accepts_bare_list(self) -> None:
        response = LLMBatchAnalysisResponse.model_validate(
            [
                {
                    "requirement_id": "REQ-001",
                    "severity": "LOW",
                    "criterion": "clarity",
                    "explanation": "Проблема.",
                    "evidence": "Текст.",
                    "recommendation": "Исправить.",
                }
            ]
        )

        assert response.findings[0].requirement_id == "REQ-001"

    def test_requirement_id_is_stripped(self) -> None:
        finding = LLMBatchFinding.model_validate(
            {
                "requirement_id": "  REQ-007  ",
                "severity": "MEDIUM",
                "criterion": "clarity",
                "explanation": "Проблема.",
                "evidence": "Текст.",
                "recommendation": "Исправить.",
            }
        )

        assert finding.requirement_id == "REQ-007"


class TestCausalFieldOrder:
    def test_batch_finding_starts_with_evidence_and_ends_with_severity(self) -> None:
        fields = list(LLMBatchFinding.model_fields)

        assert fields == [
            "requirement_id",
            "evidence",
            "explanation",
            "criterion",
            "severity",
            "recommendation",
            "origin",
        ]
        assert fields.index("evidence") < fields.index("explanation")
        assert fields.index("explanation") < fields.index("criterion")
        assert fields.index("criterion") < fields.index("severity")

    def test_set_finding_starts_with_evidence(self) -> None:
        fields = list(SetFinding.model_fields)

        assert fields[:4] == ["evidence", "description", "category", "severity"]
        assert fields.index("evidence") < fields.index("severity")


class TestRejectedFlags:
    def test_bare_string_is_accepted(self) -> None:
        response = LLMBatchAnalysisResponse.model_validate(
            {"findings": [], "rejected_flag_ids": "LINT-001"}
        )

        assert response.rejected_flag_ids == ["LINT-001"]

    def test_static_origin_is_preserved(self) -> None:
        finding = LLMBatchFinding(
            requirement_id="REQ-001",
            evidence="Текст.",
            explanation="Проблема.",
            criterion=QualityCriterion.CLARITY,
            severity=Severity.MEDIUM,
            recommendation="Исправить.",
            origin=FindingOrigin.STATIC_LINTER,
        )

        analysis = RequirementAnalysis.from_findings("REQ-001", [finding])

        assert analysis.findings[0].origin is FindingOrigin.STATIC_LINTER

    def test_set_category_accepts_ambiguous_requirement(self) -> None:
        finding = SetFinding(
            evidence=["Текст."],
            description="Описание.",
            category="ambiguous-requirement",
            severity="MEDIUM",
            title="Заголовок",
        )

        assert finding.category is SetFindingCategory.AMBIGUOUS_REQUIREMENT


class TestStatusDerivation:
    def test_no_findings_is_pass(self) -> None:
        analysis = RequirementAnalysis.from_findings("REQ-001", [])

        assert analysis.status is RequirementStatus.PASS
        assert analysis.findings == []

    def test_low_or_medium_is_warning(self) -> None:
        analysis = RequirementAnalysis.from_findings(
            "REQ-001", [_finding(severity=Severity.LOW)]
        )

        assert analysis.status is RequirementStatus.WARNING
        assert analysis.findings[0].status is RequirementStatus.WARNING

    def test_high_is_fail(self) -> None:
        analysis = RequirementAnalysis.from_findings(
            "REQ-001", [_finding(severity=Severity.HIGH)]
        )

        assert analysis.status is RequirementStatus.FAIL
        assert analysis.findings[0].status is RequirementStatus.FAIL

    def test_multiple_findings_are_preserved(self) -> None:
        analysis = RequirementAnalysis.from_findings(
            "REQ-001",
            [
                _finding(severity=Severity.LOW, criterion=QualityCriterion.CLARITY),
                _finding(
                    severity=Severity.CRITICAL,
                    criterion=QualityCriterion.VERIFIABILITY,
                ),
            ],
        )

        assert analysis.status is RequirementStatus.FAIL
        assert len(analysis.findings) == 2
        assert analysis.severity_counts()[Severity.CRITICAL] == 1
