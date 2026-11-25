"""Unit tests for LLM response validation and analysis status derivation."""

import pytest
from pydantic import ValidationError

from app.models.analysis import (
    LLMAnalysisResponse,
    LLMExtractionResponse,
    LLMFinding,
    QualityCriterion,
    RequirementAnalysis,
    RequirementStatus,
    Severity,
)


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
