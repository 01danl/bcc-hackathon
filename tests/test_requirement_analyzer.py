"""Unit tests for batched LLM analysis (fully mocked model)."""

from __future__ import annotations

from typing import Any

import pytest

from app.llm.client import LLMResponseError
from app.models.analysis import (
    FindingOrigin,
    LLMBatchAnalysisResponse,
    LLMBatchFinding,
    QualityCriterion,
    RequirementStatus,
    Severity,
)
from app.models.lint import LintCategory, LintFlag
from app.models.requirement import (
    Requirement,
    RequirementCollection,
    SourceLocation,
)
from app.services.requirement_analyzer import RequirementAnalyzer
from app.services.requirement_triage import TriageResult


class FakeClient:
    def __init__(self, responses: list[LLMBatchAnalysisResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _collection(count: int = 2) -> RequirementCollection:
    texts = [
        "Система должна работать быстро.",
        "Система должна хранить порог.",
        "Система должна отправлять push.",
    ]
    return RequirementCollection(
        requirements=[
            Requirement(
                id=f"REQ-{number:03d}",
                text=texts[(number - 1) % len(texts)],
                location=SourceLocation(section="Требования"),
            )
            for number in range(1, count + 1)
        ]
    )


def _corpus(collection: RequirementCollection) -> str:
    return "\n".join(item.text for item in collection.requirements)


def _finding(
    requirement_id: str,
    severity: Severity = Severity.MEDIUM,
    criterion: QualityCriterion = QualityCriterion.CLARITY,
    evidence: str = "работать быстро",
) -> LLMBatchFinding:
    return LLMBatchFinding(
        requirement_id=requirement_id,
        severity=severity,
        criterion=criterion,
        explanation="Формулировка не содержит измеримого критерия.",
        evidence=evidence,
        recommendation="Указать максимальное время отклика.",
    )


class TestBatchRequirementAnalyzer:
    def test_one_call_for_the_whole_batch_and_pass_statuses(self) -> None:
        collection = _collection()
        client = FakeClient([LLMBatchAnalysisResponse(findings=[])])
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analyses = analyzer.analyze_all(
            collection, document_text=_corpus(collection)
        )

        assert len(analyses.analyses) == 2
        assert all(
            analysis.status is RequirementStatus.PASS
            for analysis in analyses.analyses
        )
        assert len(client.calls) == 1

    def test_finding_is_attached_to_its_requirement_id(self) -> None:
        collection = _collection()
        client = FakeClient(
            [LLMBatchAnalysisResponse(findings=[_finding("REQ-002", Severity.HIGH)])]
        )
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analyses = analyzer.analyze_all(
            collection, document_text=_corpus(collection)
        )

        first, second = analyses.analyses
        assert first.status is RequirementStatus.PASS
        assert second.status is RequirementStatus.FAIL
        assert second.findings[0].requirement_id == "REQ-002"
        assert second.findings[0].status is RequirementStatus.FAIL
        assert second.findings[0].severity is Severity.HIGH

    def test_evidence_from_another_requirement_is_accepted(self) -> None:
        collection = _collection()
        other_evidence = _finding(
            "REQ-001",
            criterion=QualityCriterion.CONSISTENCY,
            evidence="Система должна хранить порог.",
        )
        client = FakeClient([LLMBatchAnalysisResponse(findings=[other_evidence])])
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analyses = analyzer.analyze_all(
            collection, document_text=_corpus(collection)
        )

        assert analyses.analyses[0].status is RequirementStatus.WARNING
        assert (
            analyses.analyses[0].findings[0].criterion is QualityCriterion.CONSISTENCY
        )

    def test_non_verbatim_evidence_is_rejected_and_retried(self) -> None:
        collection = _collection()
        invented = _finding("REQ-001", evidence="этой фразы нет в документе")
        client = FakeClient(
            [
                LLMBatchAnalysisResponse(findings=[invented]),
                LLMBatchAnalysisResponse(findings=[invented]),
            ]
        )
        analyzer = RequirementAnalyzer(client, max_response_retries=1)  # type: ignore[arg-type]

        with pytest.raises(LLMResponseError, match="exact fragment"):
            analyzer.analyze_all(
                collection, document_text=_corpus(collection)
            )

        assert len(client.calls) == 2

    def test_unknown_requirement_id_is_rejected_and_retried(self) -> None:
        collection = _collection()
        unknown = _finding("REQ-999")
        client = FakeClient(
            [
                LLMBatchAnalysisResponse(findings=[unknown]),
                LLMBatchAnalysisResponse(findings=[unknown]),
            ]
        )
        analyzer = RequirementAnalyzer(client, max_response_retries=1)  # type: ignore[arg-type]

        with pytest.raises(LLMResponseError, match="unknown requirement ids"):
            analyzer.analyze_all(
                collection, document_text=_corpus(collection)
            )

        assert len(client.calls) == 2

    def test_evidence_ignores_markdown_markers(self) -> None:
        collection = RequirementCollection(
            requirements=[
                Requirement(id="REQ-001", text="Система должна отправлять push.")
            ]
        )
        finding = _finding(
            "REQ-001",
            criterion=QualityCriterion.VERIFIABILITY,
            evidence="Ф-1. Система должна отправлять push.",
        )
        client = FakeClient([LLMBatchAnalysisResponse(findings=[finding])])
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analyses = analyzer.analyze_all(
            collection,
            document_text="* **Ф-1.** Система должна отправлять push.",
        )

        assert analyses.analyses[0].status is RequirementStatus.WARNING

    def test_large_set_is_split_into_batches_of_twenty(self) -> None:
        collection = _collection(25)
        client = FakeClient(
            [LLMBatchAnalysisResponse(findings=[]), LLMBatchAnalysisResponse(findings=[])]
        )
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]
        progress: list[tuple[int, int, str]] = []

        analyses = analyzer.analyze_all(
            collection,
            document_text=_corpus(collection),
            progress=lambda number, total, requirement: progress.append(
                (number, total, requirement.id)
            ),
        )

        assert len(analyses.analyses) == 25
        assert len(client.calls) == 2
        assert progress[0][0] == 1
        assert progress[-1][0] == 25
        assert progress[-1][1] == 25
        assert {item[2] for item in progress} == {
            f"REQ-{number:03d}" for number in range(1, 26)
        }
        assert all(
            analysis.status is RequirementStatus.PASS
            for analysis in analyses.analyses
        )

    def test_single_requirement_wrapper_uses_the_batch_path(self) -> None:
        collection = _collection(1)
        client = FakeClient([LLMBatchAnalysisResponse(findings=[])])
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analysis = analyzer.analyze(
            collection.requirements[0], collection, document_text=_corpus(collection)
        )

        assert analysis.requirement_id == "REQ-001"
        assert analysis.status is RequirementStatus.PASS
        assert len(client.calls) == 1

    def test_fuzzy_evidence_is_canonicalized_without_a_retry(self) -> None:
        collection = RequirementCollection(
            requirements=[
                Requirement(
                    id="REQ-001",
                    text="Система должна отправлять push-уведомление.",
                )
            ]
        )
        finding = _finding(
            "REQ-001",
            evidence="Система должна отправлять push уведомление.",
        )
        client = FakeClient([LLMBatchAnalysisResponse(findings=[finding])])
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analyses = analyzer.analyze_all(
            collection,
            document_text="* **Ф-1.** Система должна отправлять push-уведомление.",
        )

        assert len(client.calls) == 1
        assert analyses.analyses[0].findings[0].evidence == (
            "Система должна отправлять push-уведомление."
        )

    def test_lint_flags_are_merged_when_the_model_omits_them(self) -> None:
        collection = _collection()
        flag = _flag("LINT-001", "REQ-001", "Система должна работать быстро.")
        client = FakeClient([LLMBatchAnalysisResponse(findings=[])])
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analyses = analyzer.analyze_all(
            collection,
            document_text=_corpus(collection),
            pre_detected_flags=[flag],
        )

        assert analyses.analyses[0].status is RequirementStatus.FAIL
        assert analyses.analyses[0].findings[0].origin is FindingOrigin.STATIC_LINTER
        assert analyses.static_findings == 1

    def test_rejected_lint_flag_is_not_merged(self) -> None:
        collection = _collection()
        flag = _flag("LINT-001", "REQ-001", "Система должна работать быстро.")
        client = FakeClient(
            [
                LLMBatchAnalysisResponse(
                    findings=[], rejected_flag_ids=["LINT-001"]
                )
            ]
        )
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analyses = analyzer.analyze_all(
            collection,
            document_text=_corpus(collection),
            pre_detected_flags=[flag],
        )

        assert analyses.analyses[0].findings == []
        assert analyses.static_findings == 0

    def test_triage_limits_the_deep_audit_to_suspicious_requirements(self) -> None:
        collection = _collection()
        client = FakeClient([LLMBatchAnalysisResponse(findings=[])])
        triage = StubTriage({"REQ-002"})
        analyzer = RequirementAnalyzer(client, triage=triage)  # type: ignore[arg-type]

        analyses = analyzer.analyze_all(
            collection, document_text=_corpus(collection)
        )

        assert len(client.calls) == 1
        prompt = client.calls[0]["user_prompt"]
        assert '"id": "REQ-002"' in prompt
        assert '"id": "REQ-001"' not in prompt
        assert analyses.deep_audited_requirements == 1
        assert analyses.skipped_requirements == 1
        assert analyses.triage_summary == "Проверено triage."

    def test_lint_flag_forces_an_audit_even_when_triage_skips_it(self) -> None:
        collection = _collection()
        flag = _flag("LINT-001", "REQ-001", "Система должна работать быстро.")
        client = FakeClient([LLMBatchAnalysisResponse(findings=[])])
        analyzer = RequirementAnalyzer(client, triage=StubTriage(set()))  # type: ignore[arg-type]

        analyses = analyzer.analyze_all(
            collection,
            document_text=_corpus(collection),
            pre_detected_flags=[flag],
        )

        assert len(client.calls) == 1
        assert "REQ-001" in client.calls[0]["user_prompt"]
        assert analyses.deep_audited_requirements == 1
        assert analyses.skipped_requirements == 1
        assert analyses.analyses[0].findings[0].origin is FindingOrigin.STATIC_LINTER

    def test_document_prefix_is_stable_across_batches(self) -> None:
        collection = _collection(25)
        client = FakeClient(
            [
                LLMBatchAnalysisResponse(findings=[]),
                LLMBatchAnalysisResponse(findings=[]),
            ]
        )
        analyzer = RequirementAnalyzer(client)  # type: ignore[arg-type]

        analyzer.analyze_all(collection, document_text=_corpus(collection))

        first, second = client.calls
        assert first["system_prompt"] == second["system_prompt"]
        assert _corpus(collection) in first["system_prompt"]
        assert _corpus(collection) not in first["user_prompt"]
        assert first["user_prompt"].rstrip().endswith(
            "only findings bound to requirement IDs and rejected lint flag IDs."
        )


class StubTriage:
    def __init__(self, suspicious_ids: set[str]) -> None:
        self._suspicious_ids = frozenset(suspicious_ids)

    def triage(
        self,
        collection: RequirementCollection,
        *,
        document_text: str | None = None,
        pre_detected_flags: Any = (),
    ) -> TriageResult:
        return TriageResult(
            suspicious_ids=self._suspicious_ids,
            reasons={item: "Причина." for item in self._suspicious_ids},
            summary="Проверено triage.",
            triaged_requirements=len(collection),
        )


def _flag(flag_id: str, requirement_id: str, evidence: str) -> LintFlag:
    return LintFlag(
        flag_id=flag_id,
        category=LintCategory.VAGUE_WORDING,
        severity=Severity.HIGH,
        criterion=QualityCriterion.VERIFIABILITY,
        message="Нечёткая скорость без измеримого предела.",
        evidence=evidence,
        recommendation="Указать время отклика.",
        requirement_id=requirement_id,
    )
