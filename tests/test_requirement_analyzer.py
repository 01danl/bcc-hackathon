"""Unit tests for per-requirement LLM analysis (fully mocked model)."""

from __future__ import annotations

from typing import Any

import pytest

from app.llm.client import LLMResponseError
from app.models.analysis import (
    LLMAnalysisResponse,
    LLMFinding,
    QualityCriterion,
    RequirementStatus,
    Severity,
)
from app.models.requirement import (
    Requirement,
    RequirementCollection,
    SourceLocation,
)
from app.services.requirement_analyzer import RequirementAnalyzer


class FakeClient:
    def __init__(self, responses: list[LLMAnalysisResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _collection() -> RequirementCollection:
    return RequirementCollection(
        requirements=[
            Requirement(
                id="REQ-001",
                text="Система должна работать быстро.",
                location=SourceLocation(section="Требования"),
            ),
            Requirement(
                id="REQ-002",
                text="Система должна хранить порог.",
                location=SourceLocation(section="Требования"),
            ),
        ]
    )


def _finding(severity: Severity = Severity.MEDIUM) -> LLMFinding:
    return LLMFinding(
        severity=severity,
        criterion=QualityCriterion.CLARITY,
        explanation="Формулировка не содержит измеримого критерия.",
        evidence="работать быстро",
        recommendation="Указать максимальное время отклика.",
    )


class TestRequirementAnalyzer:
    def test_pass_when_model_returns_no_findings(self) -> None:
        collection = _collection()
        client = FakeClient([LLMAnalysisResponse(findings=[])])
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analysis = analyzer.analyze(
            collection.requirements[0],
            collection,
            document_text="\n".join(item.text for item in collection.requirements),
        )

        assert analysis.requirement_id == "REQ-001"
        assert analysis.status is RequirementStatus.PASS
        assert analysis.findings == []

    def test_findings_are_attached_to_requirement(self) -> None:
        collection = _collection()
        client = FakeClient(
            [LLMAnalysisResponse(findings=[_finding(Severity.HIGH)])]
        )
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analysis = analyzer.analyze(
            collection.requirements[0],
            collection,
            document_text="\n".join(item.text for item in collection.requirements),
        )

        assert analysis.status is RequirementStatus.FAIL
        assert analysis.findings[0].requirement_id == "REQ-001"
        assert analysis.findings[0].status is RequirementStatus.FAIL
        assert analysis.findings[0].severity is Severity.HIGH

    def test_evidence_from_another_requirement_is_accepted(self) -> None:
        collection = _collection()
        other_evidence = LLMFinding(
            severity=Severity.MEDIUM,
            criterion=QualityCriterion.CONSISTENCY,
            explanation="Требования конфликтуют.",
            evidence="Система должна хранить порог.",
            recommendation="Согласовать поведение.",
        )
        client = FakeClient([LLMAnalysisResponse(findings=[other_evidence])])
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analysis = analyzer.analyze(
            collection.requirements[0],
            collection,
            document_text="\n".join(item.text for item in collection.requirements),
        )

        assert analysis.status is RequirementStatus.WARNING
        assert analysis.findings[0].criterion is QualityCriterion.CONSISTENCY

    def test_non_verbatim_evidence_is_rejected_and_retried(self) -> None:
        collection = _collection()
        invented = LLMFinding(
            severity=Severity.HIGH,
            criterion=QualityCriterion.CLARITY,
            explanation="Проблема.",
            evidence="этой фразы нет в документе",
            recommendation="Исправить.",
        )
        client = FakeClient(
            [
                LLMAnalysisResponse(findings=[invented]),
                LLMAnalysisResponse(findings=[invented]),
            ]
        )
        analyzer = RequirementAnalyzer(client, max_response_retries=1)  # type: ignore[arg-type]

        with pytest.raises(LLMResponseError, match="exact fragment"):
            analyzer.analyze(
                collection.requirements[0],
                collection,
                document_text="\n".join(item.text for item in collection.requirements),
            )

        assert len(client.calls) == 2

    def test_evidence_ignores_markdown_markers(self) -> None:
        collection = RequirementCollection(
            requirements=[
                Requirement(
                    id="REQ-001",
                    text="Ф-1. Система должна отправлять push.",
                    location=SourceLocation(section="Требования"),
                )
            ]
        )
        finding = LLMFinding(
            severity=Severity.MEDIUM,
            criterion=QualityCriterion.VERIFIABILITY,
            explanation="Не указан проверяемый результат.",
            evidence="Ф-1. Система должна отправлять push.",
            recommendation="Добавить наблюдаемый результат.",
        )
        client = FakeClient([LLMAnalysisResponse(findings=[finding])])
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analysis = analyzer.analyze(
            collection.requirements[0],
            collection,
            document_text="* **Ф-1.** Система должна отправлять push.",
        )

        assert analysis.status is RequirementStatus.WARNING

    def test_analyze_all_calls_progress_and_returns_collection(self) -> None:
        collection = _collection()
        client = FakeClient(
            [
                LLMAnalysisResponse(findings=[]),
                LLMAnalysisResponse(findings=[_finding(Severity.LOW)]),
            ]
        )
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]
        progress: list[tuple[int, int, str]] = []

        analyses = analyzer.analyze_all(
            collection,
            document_text="\n".join(item.text for item in collection.requirements),
            progress=lambda number, total, requirement: progress.append(
                (number, total, requirement.id)
            ),
        )

        assert len(analyses.analyses) == 2
        assert progress == [(1, 2, "REQ-001"), (2, 2, "REQ-002")]
        assert analyses.by_status()[RequirementStatus.WARNING] == 1
        assert analyses.by_status()[RequirementStatus.PASS] == 1
