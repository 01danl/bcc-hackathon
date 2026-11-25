"""Unit tests for the set-level audit service (fully mocked model)."""

from __future__ import annotations

from typing import Any

import pytest

from app.llm.client import LLMResponseError
from app.models.analysis import (
    RequirementAnalysis,
    RequirementStatus,
    Severity,
)
from app.models.requirement import Requirement, RequirementCollection
from app.models.set_audit import (
    LLMSetAuditResponse,
    SetFinding,
    SetFindingCategory,
    VerdictStatus,
)
from app.services.set_analyzer import RequirementSetAnalyzer

DOCUMENT = (
    "# ТЗ\n\n"
    "## Функциональные требования\n\n"
    "* **Ф-1.** Система отправляет push при достижении порога.\n"
    "* **Ф-2.** Переводы и снятие наличных не учитываются.\n"
)


class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _collection() -> RequirementCollection:
    return RequirementCollection(
        requirements=[
            Requirement(
                id="REQ-001",
                text="Система отправляет push при достижении порога.",
                original_id="Ф-1",
            ),
            Requirement(
                id="REQ-002",
                text="Переводы и снятие наличных не учитываются.",
                original_id="Ф-2",
            ),
        ]
    )


def _analyses(*statuses: RequirementStatus) -> list[RequirementAnalysis]:
    return [
        RequirementAnalysis(requirement_id=f"REQ-{number:03d}", status=status)
        for number, status in enumerate(statuses, start=1)
    ]


def _finding(**overrides: Any) -> SetFinding:
    payload: dict[str, Any] = {
        "category": SetFindingCategory.CONTRADICTION,
        "severity": Severity.MEDIUM,
        "title": "Требования конфликтуют",
        "description": "Ф-1 и Ф-2 описывают один счётчик по-разному.",
        "related_req_ids": ["REQ-001"],
        "evidence": ["Система отправляет push при достижении порога."],
    }
    payload.update(overrides)
    return SetFinding(**payload)


def _response(findings: list[SetFinding] | None = None) -> LLMSetAuditResponse:
    return LLMSetAuditResponse(
        summary="Набор требований согласован частично.",
        goal_coverage="Цель документа покрыта.",
        findings=findings or [],
        priority_actions=["Уточнить формулировки."],
    )


def _analyze(client: FakeClient, analyses: list[RequirementAnalysis]) -> Any:
    return RequirementSetAnalyzer(client).analyze(  # type: ignore[arg-type]
        DOCUMENT, _collection(), analyses
    )


class TestVerdict:
    def test_ready_without_findings_or_remarks(self) -> None:
        client = FakeClient([_response()])

        audit = _analyze(
            client, _analyses(RequirementStatus.PASS, RequirementStatus.PASS)
        )

        assert audit.verdict is VerdictStatus.READY
        assert audit.excluded_findings == 0

    def test_blocked_by_high_set_finding(self) -> None:
        client = FakeClient(
            [_response([_finding(severity=Severity.HIGH)])]
        )

        audit = _analyze(
            client, _analyses(RequirementStatus.PASS, RequirementStatus.PASS)
        )

        assert audit.verdict is VerdictStatus.BLOCKED

    def test_blocked_by_failed_requirement_analysis(self) -> None:
        client = FakeClient([_response()])

        audit = _analyze(
            client, _analyses(RequirementStatus.FAIL, RequirementStatus.PASS)
        )

        assert audit.verdict is VerdictStatus.BLOCKED

    def test_needs_revision_on_medium_finding(self) -> None:
        client = FakeClient([_response([_finding(severity=Severity.MEDIUM)])])

        audit = _analyze(
            client, _analyses(RequirementStatus.PASS, RequirementStatus.PASS)
        )

        assert audit.verdict is VerdictStatus.NEEDS_REVISION

    def test_needs_revision_on_warning_analysis(self) -> None:
        client = FakeClient([_response()])

        audit = _analyze(
            client, _analyses(RequirementStatus.PASS, RequirementStatus.WARNING)
        )

        assert audit.verdict is VerdictStatus.NEEDS_REVISION


class TestEvidenceVerification:
    def test_unverified_quotes_trigger_one_retry(self) -> None:
        bad = _finding(evidence=["этой фразы нет в документе"])
        good = _finding()
        client = FakeClient([_response([bad]), _response([good])])

        audit = _analyze(
            client, _analyses(RequirementStatus.PASS, RequirementStatus.PASS)
        )

        assert len(client.calls) == 2
        assert "этой фразы нет в документе" in client.calls[1]["user_prompt"]
        assert "ОШИБКА ВАЛИДАЦИИ" in client.calls[1]["user_prompt"]
        assert audit.findings[0].evidence == [
            "Система отправляет push при достижении порога."
        ]

    def test_findings_without_quotes_trigger_retry(self) -> None:
        silent = _finding(evidence=[])
        quoted = _finding()
        client = FakeClient([_response([silent]), _response([quoted])])

        audit = _analyze(
            client, _analyses(RequirementStatus.PASS, RequirementStatus.PASS)
        )

        assert len(client.calls) == 2
        assert audit.findings[0].evidence == [
            "Система отправляет push при достижении порога."
        ]

    def test_unsupported_findings_are_excluded_after_the_last_attempt(self) -> None:
        bad = _finding(evidence=["этой фразы нет в документе"])
        client = FakeClient([_response([bad]), _response([bad])])

        audit = _analyze(client, _analyses(RequirementStatus.PASS))

        assert audit.findings == []
        assert audit.excluded_findings == 1
        # Nothing confirmed and nothing blocking, but the run is not clean:
        # the model reported something the document does not back.
        assert audit.verdict is VerdictStatus.NEEDS_REVISION
        assert "исключена" in audit.verdict_reason

    def test_verified_quotes_are_kept_and_unverified_dropped(self) -> None:
        mixed = _finding(
            evidence=[
                "Переводы и снятие наличных не учитываются.",
                "нет такой фразы",
            ]
        )
        client = FakeClient([_response([mixed]), _response([mixed])])

        audit = _analyze(client, [])

        assert audit.findings[0].evidence == [
            "Переводы и снятие наличных не учитываются."
        ]


class TestRelatedIds:
    def test_document_ids_are_mapped_and_unknown_dropped(self) -> None:
        finding = _finding(
            related_req_ids=["Ф-1", "REQ-002", "REQ-999", "Ф-1"],
        )
        client = FakeClient([_response([finding])])

        audit = _analyze(client, [])

        assert audit.findings[0].related_req_ids == ["REQ-001", "REQ-002"]


class TestPromptAndRetries:
    def test_prompt_carries_full_document_and_requirements(self) -> None:
        client = FakeClient([_response()])

        _analyze(client, [])

        prompt = client.calls[0]["user_prompt"]
        assert "### ПОЛНЫЙ ТЕКСТ ДОКУМЕНТА" in prompt
        assert "Переводы и снятие наличных не учитываются." in prompt
        assert '"original_id": "Ф-1"' in prompt
        assert client.calls[0]["system_prompt"].startswith(
            "Ты — старший системный аналитик"
        )

    def test_invalid_json_is_retried(self) -> None:
        client = FakeClient([LLMResponseError("not valid JSON"), _response()])

        audit = _analyze(client, [])

        assert len(client.calls) == 2
        assert audit.summary == "Набор требований согласован частично."

    def test_persistently_invalid_answer_raises(self) -> None:
        client = FakeClient(
            [LLMResponseError("bad"), LLMResponseError("bad")]
        )

        with pytest.raises(LLMResponseError):
            _analyze(client, [])

        assert len(client.calls) == 2
