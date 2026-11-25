"""Unit tests for the set-audit models: LLM answer validation and verdict."""

import pytest
from pydantic import ValidationError

from app.models.analysis import Severity
from app.models.set_audit import (
    LLMSetAuditResponse,
    RequirementSetAudit,
    SetFinding,
    SetFindingCategory,
    VerdictStatus,
)


def _finding_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "category": "contradiction",
        "severity": "HIGH",
        "title": "Ф-3 и Ф-4 конфликтуют",
        "description": "Требования задают разный момент отправки push.",
        "related_req_ids": ["REQ-003"],
        "evidence": ["На карту приходится максимум один такой push"],
    }
    payload.update(overrides)
    return payload


def _response_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "summary": "В наборе есть противоречия.",
        "goal_coverage": "Цель покрыта частично.",
        "findings": [_finding_payload()],
        "priority_actions": ["Согласовать момент отправки push."],
    }
    payload.update(overrides)
    return payload


class TestSetFindingValidation:
    def test_category_and_severity_are_case_insensitive(self) -> None:
        finding = SetFinding.model_validate(
            _finding_payload(category="Integration-Mismatch", severity="critical")
        )

        assert finding.category is SetFindingCategory.INTEGRATION_MISMATCH
        assert finding.severity is Severity.CRITICAL

    def test_every_category_is_accepted(self) -> None:
        categories = [category.value for category in SetFindingCategory]

        findings = [
            SetFinding.model_validate(_finding_payload(category=category))
            for category in categories
        ]

        assert [finding.category for finding in findings] == list(SetFindingCategory)

    def test_unknown_category_rejected(self) -> None:
        with pytest.raises(ValidationError, match="category must be one of"):
            SetFinding.model_validate(_finding_payload(category="typo"))

    def test_unknown_severity_becomes_medium(self) -> None:
        finding = SetFinding.model_validate(_finding_payload(severity="severe"))

        assert finding.severity is Severity.MEDIUM

    def test_empty_title_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SetFinding.model_validate(_finding_payload(title="   "))

    def test_title_whitespace_is_collapsed(self) -> None:
        finding = SetFinding.model_validate(
            _finding_payload(title="  Ф-3 и   Ф-4\nконфликтуют ")
        )

        assert finding.title == "Ф-3 и Ф-4 конфликтуют"

    @pytest.mark.parametrize("field", ["related_req_ids", "evidence"])
    def test_none_and_bare_string_become_lists(self, field: str) -> None:
        as_none = SetFinding.model_validate(_finding_payload(**{field: None}))
        as_string = SetFinding.model_validate(_finding_payload(**{field: "значение"}))

        assert getattr(as_none, field) == []
        assert getattr(as_string, field) == ["значение"]

    @pytest.mark.parametrize("field", ["related_req_ids", "evidence"])
    def test_blank_items_are_dropped_from_lists(self, field: str) -> None:
        finding = SetFinding.model_validate(
            _finding_payload(**{field: ["  значение  ", "", "  ", None, 4]})
        )

        assert getattr(finding, field) == ["значение"]


class TestSetAuditResponse:
    def test_defaults_are_empty_lists(self) -> None:
        response = LLMSetAuditResponse.model_validate(
            {"summary": "ок", "goal_coverage": "ок"}
        )

        assert response.findings == []
        assert response.priority_actions == []

    def test_missing_summary_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LLMSetAuditResponse.model_validate({"goal_coverage": "ок"})

    def test_bare_string_priority_actions_becomes_a_list(self) -> None:
        response = LLMSetAuditResponse.model_validate(
            _response_payload(priority_actions="Заполнить раздел.")
        )

        assert response.priority_actions == ["Заполнить раздел."]


class TestRequirementSetAudit:
    def test_extends_the_model_answer_with_a_derived_verdict(self) -> None:
        response = LLMSetAuditResponse.model_validate(_response_payload())

        audit = RequirementSetAudit(
            **response.model_dump(),
            verdict=VerdictStatus.BLOCKED,
            verdict_reason="Обнаружены противоречия.",
        )

        assert audit.summary == response.summary
        assert audit.findings == response.findings
        assert audit.verdict is VerdictStatus.BLOCKED
        assert audit.excluded_findings == 0

    def test_dumps_to_plain_json_values(self) -> None:
        audit = RequirementSetAudit(
            **_response_payload(),
            verdict=VerdictStatus.NEEDS_REVISION,
            verdict_reason="Есть замечания.",
        )

        dumped = audit.model_dump(mode="json")

        assert dumped["verdict"] == "NEEDS_REVISION"
        assert dumped["findings"][0]["category"] == "contradiction"
        assert dumped["findings"][0]["severity"] == "HIGH"

    def test_empty_verdict_reason_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RequirementSetAudit(
                **_response_payload(), verdict=VerdictStatus.READY, verdict_reason=""
            )
