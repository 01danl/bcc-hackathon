"""Tests for the root ``agent.py`` one-command entry point."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import agent
from app.config import Settings
from app.models.analysis import (
    LLMBatchAnalysisResponse,
    LLMBatchFinding,
    LLMExtractionResponse,
    LLMExtractedRequirement,
    QualityCriterion,
    Severity,
)
from app.models.requirement import RequirementCategory
from app.models.triage import LLMTriageItem, LLMTriageResponse, TriageRisk
from app.models.set_audit import LLMSetAuditResponse

DOCUMENT = "# Требования\n\n* **Ф-1.** Система должна отправлять push.\n"


class FakeLLMClient:
    """Answer every stage deterministically, without any network access."""

    def __init__(self) -> None:
        self.calls: list[type] = []

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, response_model: type
    ) -> Any:
        self.calls.append(response_model)
        if response_model is LLMExtractionResponse:
            return LLMExtractionResponse(
                requirements=[
                    LLMExtractedRequirement(
                        original_id="Ф-1",
                        text="Система должна отправлять push.",
                        category=RequirementCategory.FUNCTIONAL,
                    )
                ]
            )
        if response_model is LLMTriageResponse:
            return LLMTriageResponse(
                suspicious=[
                    LLMTriageItem(
                        requirement_id="REQ-001",
                        reason="Нужен глубокий аудит.",
                        risk=TriageRisk.MEDIUM,
                    )
                ],
                summary="Одно требование отправлено на аудит.",
            )
        if response_model is LLMBatchAnalysisResponse:
            return LLMBatchAnalysisResponse(
                findings=[
                    LLMBatchFinding(
                        requirement_id="REQ-001",
                        evidence="Система должна отправлять push.",
                        explanation="Не указан наблюдаемый результат.",
                        criterion=QualityCriterion.VERIFIABILITY,
                        severity=Severity.MEDIUM,
                        recommendation="Добавить наблюдаемый результат.",
                    )
                ]
            )
        if response_model is LLMSetAuditResponse:
            return LLMSetAuditResponse(
                summary="Набор требований согласован.",
                goal_coverage="Цель документа покрыта.",
            )
        raise AssertionError(f"unexpected response model: {response_model}")


def _patch_llm(monkeypatch: Any, client: FakeLLMClient) -> None:
    settings = Settings(
        llm_api_key="sk-or-v1-secret-test-key",
        llm_base_url="https://openrouter.ai/api/v1",
        llm_model="test/model",
    )
    monkeypatch.setattr("app.cli.load_settings", lambda: settings)
    monkeypatch.setattr("app.cli.build_llm_client", lambda settings: client)


class TestAgentEntryPoint:
    def test_full_pipeline_runs_without_a_subcommand(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        client = FakeLLMClient()
        _patch_llm(monkeypatch, client)
        input_path = tmp_path / "requirements.md"
        input_path.write_text(DOCUMENT, encoding="utf-8")
        output_path = tmp_path / "nested" / "deeper" / "report.md"

        result = CliRunner().invoke(
            agent.agent_app,
            ["--input", str(input_path), "--output", str(output_path)],
        )

        assert result.exit_code == 0, result.output
        assert output_path.parent.is_dir()
        assert output_path.exists()
        report = output_path.read_text(encoding="utf-8")
        assert "**Вердикт: NEEDS_REVISION.**" in report
        assert "Не указан наблюдаемый результат." in report
        assert Counter(call.__name__ for call in client.calls) == {
            "LLMExtractionResponse": 1,
            "LLMTriageResponse": 1,
            "LLMBatchAnalysisResponse": 1,
            "LLMSetAuditResponse": 1,
        }

    def test_output_in_the_current_directory_needs_no_parent(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        client = FakeLLMClient()
        _patch_llm(monkeypatch, client)
        (tmp_path / "requirements.md").write_text(DOCUMENT, encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = CliRunner().invoke(
            agent.agent_app,
            ["--input", "requirements.md", "--output", "report.md"],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "report.md").exists()
