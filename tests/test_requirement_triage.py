"""Unit tests for the cheap cascading triage stage."""

from __future__ import annotations

from typing import Any

from app.llm.client import LLMRequestError, LLMResponseError
from app.models.analysis import QualityCriterion, Severity
from app.models.lint import LintCategory, LintFlag
from app.models.requirement import Requirement, RequirementCollection
from app.models.triage import LLMTriageItem, LLMTriageResponse, TriageRisk
from app.services.requirement_triage import RequirementTriage


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
                text="Система должна работать быстро.",
                original_id="Ф-1",
            ),
            Requirement(
                id="REQ-002",
                text="Порог задаётся в тиын.",
                original_id="Ф-2",
            ),
        ]
    )


def _response(*ids: str) -> LLMTriageResponse:
    return LLMTriageResponse(
        suspicious=[
            LLMTriageItem(
                requirement_id=requirement_id,
                reason="Есть конкретный риск.",
                risk=TriageRisk.MEDIUM,
            )
            for requirement_id in ids
        ],
        summary="Проверено.",
    )


class TestTriageRouting:
    def test_only_suspicious_requirements_are_returned(self) -> None:
        client = FakeClient([_response("REQ-001")])
        triage = RequirementTriage(client)  # type: ignore[arg-type]

        result = triage.triage(_collection(), document_text="документ")

        assert result.suspicious_ids == frozenset({"REQ-001"})
        assert result.reasons["REQ-001"] == "Есть конкретный риск."
        assert result.triaged_requirements == 2
        assert result.fallback_used is False

    def test_lint_flags_always_reach_the_auditor(self) -> None:
        client = FakeClient([_response()])
        triage = RequirementTriage(client)  # type: ignore[arg-type]
        flag = LintFlag(
            flag_id="LINT-001",
            category=LintCategory.VAGUE_WORDING,
            severity=Severity.MEDIUM,
            criterion=QualityCriterion.CLARITY,
            message="Субъективная оценка.",
            evidence="Порог задаётся в тиын.",
            recommendation="Уточнить.",
            requirement_id="REQ-002",
        )

        result = triage.triage(
            _collection(),
            document_text="документ",
            pre_detected_flags=[flag],
        )

        assert result.suspicious_ids == frozenset({"REQ-002"})
        assert "Линтер" in result.reasons["REQ-002"]

    def test_unknown_ids_are_ignored(self) -> None:
        client = FakeClient([_response("REQ-999", "REQ-001")])
        triage = RequirementTriage(client)  # type: ignore[arg-type]

        result = triage.triage(_collection(), document_text="документ")

        assert result.suspicious_ids == frozenset({"REQ-001"})


class TestSafeFallback:
    def test_malformed_answer_falls_back_to_auditing_everything(self) -> None:
        client = FakeClient(
            [LLMResponseError("bad"), LLMResponseError("bad")]
        )
        triage = RequirementTriage(client, max_response_retries=1)  # type: ignore[arg-type]

        result = triage.triage(_collection(), document_text="документ")

        assert result.suspicious_ids == frozenset({"REQ-001", "REQ-002"})
        assert result.fallback_used is True
        assert "деградировал" in result.summary or "проверен целиком" in result.summary

    def test_network_failure_falls_back_to_auditing_everything(self) -> None:
        client = FakeClient([LLMRequestError("provider is down")])
        triage = RequirementTriage(client)  # type: ignore[arg-type]

        result = triage.triage(_collection(), document_text="документ")

        assert result.suspicious_ids == frozenset({"REQ-001", "REQ-002"})
        assert result.fallback_used is True


class TestPrompt:
    def test_document_is_in_the_stable_system_prefix(self) -> None:
        client = FakeClient([_response()])
        triage = RequirementTriage(client)  # type: ignore[arg-type]
        document = "ПОЛНЫЙ ТЕКСТ ДОКУМЕНТА"

        triage.triage(_collection(), document_text=document)

        system_prompt = client.calls[0]["system_prompt"]
        user_prompt = client.calls[0]["user_prompt"]
        assert document in system_prompt
        assert "### FULL SOURCE DOCUMENT" in system_prompt
        assert document not in user_prompt
        assert "### BATCH REQUIREMENTS (JSON)" in user_prompt
        assert user_prompt.rstrip().endswith(
            "Check the batch and return a JSON object that follows the schema "
            "from the system message."
        )
