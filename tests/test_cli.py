"""CLI integration tests with the LLM services stubbed out."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from app.cli import app, get_git_commit
from app.config import Settings
from app.llm.client import LLMRequestError
from app.models.analysis import (
    LLMAnalysisResponse,
    LLMExtractionResponse,
    LLMExtractedRequirement,
    LLMFinding,
    QualityCriterion,
    RequirementAnalysis,
    RequirementAnalysisCollection,
    RequirementStatus,
    Severity,
)
from app.models.requirement import (
    DocumentFormat,
    Requirement,
    RequirementCategory,
    RequirementCollection,
    SourceDocument,
)
from app.models.set_audit import (
    LLMSetAuditResponse,
    RequirementSetAudit,
    SetFinding,
    SetFindingCategory,
    VerdictStatus,
)


def _settings() -> Settings:
    return Settings(
        llm_api_key="sk-or-v1-secret-test-key",
        llm_model="test/model",
        llm_base_url="https://openrouter.ai/api/v1",
    )


def _collection() -> RequirementCollection:
    return RequirementCollection(
        requirements=[
            Requirement(
                id="REQ-001",
                text="Система должна отправлять push.",
                original_id="Ф-1",
            )
        ]
    )


def _set_audit(**overrides: Any) -> RequirementSetAudit:
    """Build a READY set audit, optionally overriding some of its fields."""
    payload: dict[str, Any] = {
        "summary": "Набор требований в целом согласован.",
        "goal_coverage": "Цель документа покрыта требованиями.",
        "verdict": VerdictStatus.READY,
        "verdict_reason": (
            "ТЗ непротиворечиво, полно и готово к передаче "
            "в разработку и тестирование."
        ),
    }
    payload.update(overrides)
    return RequirementSetAudit(**payload)


def _stub_extractor() -> type:
    class StubExtractor:
        def __init__(self, *args: Any) -> None:
            pass

        def extract(self, document: SourceDocument) -> RequirementCollection:
            return _collection()

        def validate(self, collection: RequirementCollection) -> list[str]:
            return []

    return StubExtractor


def _stub_analyzer() -> type:
    class StubAnalyzer:
        def __init__(self, *args: Any) -> None:
            pass

        def analyze_all(self, *args: Any, **kwargs: Any) -> RequirementAnalysisCollection:
            return RequirementAnalysisCollection(
                analyses=[
                    RequirementAnalysis(
                        requirement_id="REQ-001",
                        status=RequirementStatus.PASS,
                    )
                ]
            )

    return StubAnalyzer


def _stub_set_analyzer(
    audit: RequirementSetAudit, calls: list[Any] | None = None
) -> type:
    """Build a set analyzer stub returning ``audit`` and recording its inputs."""

    class StubSetAnalyzer:
        def __init__(self, *args: Any) -> None:
            pass

        def analyze(
            self,
            document_text: str,
            collection: RequirementCollection,
            analyses: Any,
        ) -> RequirementSetAudit:
            if calls is not None:
                calls.append((document_text, collection, analyses))
            return audit

    return StubSetAnalyzer


def _failing_set_analyzer() -> type:
    class FailingSetAnalyzer:
        def __init__(self, *args: Any) -> None:
            pass

        def analyze(self, *args: Any) -> None:
            raise LLMRequestError("set audit provider is down")

    return FailingSetAnalyzer


@pytest.fixture(autouse=True)
def _fixed_commit(monkeypatch: Any) -> None:
    """Keep CLI tests independent of the repository's git state."""
    monkeypatch.setattr("app.cli.get_git_commit", lambda: "abc1234")


def _run(tmp_path: Path) -> tuple[Path, Path, Path, Any]:
    input_path = tmp_path / "requirements.md"
    input_path.write_text(
        "# Требования\n\n* Ф-1. Система должна отправлять push.\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "report.md"
    result = CliRunner().invoke(
        app,
        [
            "analyze",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    return input_path, output_path, tmp_path, result


class TestCliConfiguration:
    def test_missing_configuration_exits_nonzero(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "app.cli.load_settings",
            lambda: Settings(llm_api_key=None, llm_model=None),
        )
        _, output_path, _, result = _run(tmp_path)
        combined = result.output + (result.stderr or "")

        assert result.exit_code == 1
        assert "LLM_API_KEY" in combined
        assert not output_path.exists()

    def test_key_is_not_printed(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        class StubExtractor:
            def __init__(self, *args: Any) -> None:
                pass

            def extract(self, document: SourceDocument) -> RequirementCollection:
                return _collection()

            def validate(self, collection: RequirementCollection) -> list[str]:
                return []

        class StubAnalyzer:
            def __init__(self, *args: Any) -> None:
                pass

            def analyze_all(self, *args: Any, **kwargs: Any) -> RequirementAnalysisCollection:
                return RequirementAnalysisCollection(
                    analyses=[
                        RequirementAnalysis(
                            requirement_id="REQ-001",
                            status=RequirementStatus.PASS,
                        )
                    ]
                )

        monkeypatch.setattr("app.cli.load_settings", _settings)
        monkeypatch.setattr("app.cli.build_llm_client", lambda settings: object())
        monkeypatch.setattr("app.cli.RequirementExtractor", StubExtractor)
        monkeypatch.setattr("app.cli.RequirementAnalyzer", StubAnalyzer)
        monkeypatch.setattr(
            "app.cli.RequirementSetAnalyzer", _stub_set_analyzer(_set_audit())
        )

        _, output_path, tmp_path, result = _run(tmp_path)
        combined = result.output + (result.stderr or "")
        report = output_path.read_text(encoding="utf-8")
        requirements_json = (tmp_path / "report.requirements.json").read_text(
            encoding="utf-8"
        )

        assert result.exit_code == 0
        assert "Готово" in result.output
        assert "sk-or-v1-secret-test-key" not in combined
        assert "sk-or-v1-secret-test-key" not in report
        assert "sk-or-v1-secret-test-key" not in requirements_json
        assert "REQ-001" in report
        assert "test/model" in report


class TestCliFailureHandling:
    def test_extraction_error_writes_no_report(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        class FailingExtractor:
            def __init__(self, *args: Any) -> None:
                pass

            def extract(self, document: SourceDocument) -> RequirementCollection:
                raise LLMRequestError("provider is down")

            def validate(self, collection: RequirementCollection) -> list[str]:
                return []

        monkeypatch.setattr("app.cli.load_settings", _settings)
        monkeypatch.setattr("app.cli.build_llm_client", lambda settings: object())
        monkeypatch.setattr("app.cli.RequirementExtractor", FailingExtractor)

        _, output_path, tmp_path, result = _run(tmp_path)
        combined = result.output + (result.stderr or "")

        assert result.exit_code == 1
        assert "provider is down" in combined
        assert not output_path.exists()
        assert not (tmp_path / "report.analysis.json").exists()

    def test_analysis_error_keeps_extraction_json_only(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        class StubExtractor:
            def __init__(self, *args: Any) -> None:
                pass

            def extract(self, document: SourceDocument) -> RequirementCollection:
                return _collection()

            def validate(self, collection: RequirementCollection) -> list[str]:
                return []

        class FailingAnalyzer:
            def __init__(self, *args: Any) -> None:
                pass

            def analyze_all(self, *args: Any, **kwargs: Any) -> None:
                raise LLMRequestError("analysis provider is down")

        monkeypatch.setattr("app.cli.load_settings", _settings)
        monkeypatch.setattr("app.cli.build_llm_client", lambda settings: object())
        monkeypatch.setattr("app.cli.RequirementExtractor", StubExtractor)
        monkeypatch.setattr("app.cli.RequirementAnalyzer", FailingAnalyzer)

        _, output_path, tmp_path, result = _run(tmp_path)
        extraction_json = tmp_path / "report.requirements.json"

        assert result.exit_code == 1
        assert not output_path.exists()
        assert extraction_json.exists()
        payload = json.loads(extraction_json.read_text(encoding="utf-8"))
        assert payload["requirements"][0]["id"] == "REQ-001"
        assert not (tmp_path / "report.analysis.json").exists()


class TestCliSetAudit:
    def test_report_ends_with_set_audit_and_verdict(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        finding = SetFinding(
            category=SetFindingCategory.CONTRADICTION,
            severity=Severity.MEDIUM,
            title="Ф-3 и Ф-4 описывают разный счётчик",
            description="Требования по-разному определяют накопленную сумму.",
            related_req_ids=["REQ-001"],
            evidence=["Система должна отправлять push."],
        )
        audit = _set_audit(
            findings=[finding],
            priority_actions=["Согласовать определение счётчика."],
            verdict=VerdictStatus.NEEDS_REVISION,
            verdict_reason="ТЗ требует уточнения неоднозначностей.",
        )
        calls: list[Any] = []
        monkeypatch.setattr("app.cli.load_settings", _settings)
        monkeypatch.setattr("app.cli.build_llm_client", lambda settings: object())
        monkeypatch.setattr("app.cli.RequirementExtractor", _stub_extractor())
        monkeypatch.setattr("app.cli.RequirementAnalyzer", _stub_analyzer())
        monkeypatch.setattr(
            "app.cli.RequirementSetAnalyzer", _stub_set_analyzer(audit, calls)
        )

        _, output_path, tmp_path, result = _run(tmp_path)
        report = output_path.read_text(encoding="utf-8")

        assert result.exit_code == 0
        assert "Вердикт: NEEDS_REVISION" in result.output
        sections = [
            "## Паспорт прогона",
            "## Нарезка требований",
            "## Проверка каждого требования",
            "## Проверка всего набора требований",
            "## Сводка",
            "## Итог",
        ]
        positions = [report.index(section) for section in sections]
        assert positions == sorted(positions)
        assert report.rstrip().split("## ")[-1].startswith("Итог")
        assert "Ф-3 и Ф-4 описывают разный счётчик" in report
        assert "REQ-001 (Ф-1)" in report
        assert "«Система должна отправлять push.»" in report
        assert "ТЗ требует уточнения неоднозначностей." in report
        assert "1. Согласовать определение счётчика." in report
        assert "следующий этап" not in report
        assert "Ограничения текущего этапа" not in report

        # The stage sees the full document, the requirements and their analyses.
        document_text, collection, analyses = calls[0]
        assert "Ф-1. Система должна отправлять push." in document_text
        assert isinstance(collection, RequirementCollection)
        assert [item.id for item in collection.requirements] == ["REQ-001"]
        assert [item.requirement_id for item in analyses] == ["REQ-001"]

        payload = json.loads(
            (tmp_path / "report.analysis.json").read_text(encoding="utf-8")
        )
        assert payload["set_audit"]["verdict"] == "NEEDS_REVISION"
        assert payload["set_audit"]["findings"][0]["category"] == "contradiction"
        assert payload["passport"]["commit"] == "abc1234"

    def test_set_audit_error_keeps_intermediate_json_and_writes_no_report(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("app.cli.load_settings", _settings)
        monkeypatch.setattr("app.cli.build_llm_client", lambda settings: object())
        monkeypatch.setattr("app.cli.RequirementExtractor", _stub_extractor())
        monkeypatch.setattr("app.cli.RequirementAnalyzer", _stub_analyzer())
        monkeypatch.setattr(
            "app.cli.RequirementSetAnalyzer", _failing_set_analyzer()
        )

        _, output_path, tmp_path, result = _run(tmp_path)
        combined = result.output + (result.stderr or "")
        analysis_json = tmp_path / "report.analysis.json"

        assert result.exit_code == 1
        assert "set audit provider is down" in combined
        assert not output_path.exists()
        assert (tmp_path / "report.requirements.json").exists()
        assert analysis_json.exists()
        payload = json.loads(analysis_json.read_text(encoding="utf-8"))
        assert payload["analyses"][0]["requirement_id"] == "REQ-001"
        assert "set_audit" not in payload

    def test_each_stage_client_is_built_with_its_own_model(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        settings = Settings(
            llm_api_key="sk-or-v1-secret-test-key",
            llm_base_url="https://openrouter.ai/api/v1",
            extractor_model="fast/model",
            auditor_model="heavy/model",
        )
        models_seen: list[str | None] = []

        def fake_build(pinned: Settings) -> object:
            models_seen.append(pinned.llm_model)
            return object()

        monkeypatch.setattr("app.cli.load_settings", lambda: settings)
        monkeypatch.setattr("app.cli.build_llm_client", fake_build)
        monkeypatch.setattr("app.cli.RequirementExtractor", _stub_extractor())
        monkeypatch.setattr("app.cli.RequirementAnalyzer", _stub_analyzer())
        monkeypatch.setattr(
            "app.cli.RequirementSetAnalyzer", _stub_set_analyzer(_set_audit())
        )

        _, output_path, _, result = _run(tmp_path)
        report = output_path.read_text(encoding="utf-8")

        assert result.exit_code == 0
        assert models_seen == ["fast/model", "heavy/model"]
        assert "| Модель извлечения | fast/model |" in report
        assert "| Модель аудита | heavy/model |" in report

    def test_passport_names_agent_version_and_commit(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("app.cli.load_settings", _settings)
        monkeypatch.setattr("app.cli.build_llm_client", lambda settings: object())
        monkeypatch.setattr("app.cli.RequirementExtractor", _stub_extractor())
        monkeypatch.setattr("app.cli.RequirementAnalyzer", _stub_analyzer())
        monkeypatch.setattr(
            "app.cli.RequirementSetAnalyzer", _stub_set_analyzer(_set_audit())
        )

        _, output_path, tmp_path, result = _run(tmp_path)
        report = output_path.read_text(encoding="utf-8")

        assert result.exit_code == 0
        assert "| Агент | AI Requirements Reviewer, версия 0.2.0 |" in report
        assert "| Коммит агента | abc1234 |" in report
        assert "| Модель извлечения | test/model |" in report
        assert "| Модель аудита | test/model |" in report
        payload = json.loads(
            (tmp_path / "report.requirements.json").read_text(encoding="utf-8")
        )
        assert payload["passport"]["agent_version"] == "0.2.0"


class TestCliPipeline:
    """The real services with one scripted model answering all three stages."""

    DOCUMENT = (
        "# Требования\n\n"
        "## Функциональные требования\n\n"
        "* **Ф-1.** Система должна отправлять push.\n"
        "* **Ф-2.** Порог задаётся в тенге.\n\n"
        "## Тестирование требований\n\n"
        "заполняются тестировщиками\n"
    )

    def test_report_is_built_from_all_three_stages(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        client = _ScriptedClient()
        monkeypatch.setattr("app.cli.load_settings", _settings)
        monkeypatch.setattr("app.cli.build_llm_client", lambda settings: client)
        input_path = tmp_path / "requirements.md"
        input_path.write_text(self.DOCUMENT, encoding="utf-8")
        output_path = tmp_path / "report.md"

        result = CliRunner().invoke(
            app,
            ["analyze", "--input", str(input_path), "--output", str(output_path)],
        )

        assert result.exit_code == 0, result.output
        assert "Вердикт: BLOCKED" in result.output
        stages = Counter(call.__name__ for call in client.calls)
        assert stages == {
            "LLMExtractionResponse": 1,
            "LLMAnalysisResponse": 2,
            "LLMSetAuditResponse": 1,
        }
        report = output_path.read_text(encoding="utf-8")
        assert "**Вердикт: BLOCKED.**" in report
        assert "заполняются тестировщиками" in report
        extraction = json.loads(
            (tmp_path / "report.requirements.json").read_text(encoding="utf-8")
        )
        assert [item["id"] for item in extraction["requirements"]] == [
            "REQ-001",
            "REQ-002",
        ]
        analysis = json.loads(
            (tmp_path / "report.analysis.json").read_text(encoding="utf-8")
        )
        assert analysis["summary"]["status"]["WARNING"] == 2
        assert analysis["set_audit"]["verdict"] == "BLOCKED"
        assert analysis["set_audit"]["findings"][0]["evidence"] == [
            "заполняются тестировщиками"
        ]


class _ScriptedClient:
    """Answer every stage deterministically, without any network access."""

    def __init__(self) -> None:
        self.calls: list[type] = []

    def complete_json(
        self, *, system_prompt: str, user_prompt: str, response_model: type
    ) -> Any:
        self.calls.append(response_model)
        if response_model is LLMExtractionResponse:
            return _extract_list_items(user_prompt)
        if response_model is LLMAnalysisResponse:
            return LLMAnalysisResponse(
                findings=[
                    LLMFinding(
                        severity=Severity.MEDIUM,
                        criterion=QualityCriterion.VERIFIABILITY,
                        explanation="Не указан наблюдаемый результат.",
                        evidence="Порог задаётся в тенге.",
                        recommendation="Добавить наблюдаемый результат.",
                    )
                ]
            )
        if response_model is LLMSetAuditResponse:
            return LLMSetAuditResponse(
                summary="Раздел тестирования не заполнен.",
                goal_coverage="Цель покрыта частично.",
                findings=[
                    SetFinding(
                        category=SetFindingCategory.INCOMPLETE_SECTION,
                        severity=Severity.HIGH,
                        title="Раздел тестирования не заполнен",
                        description="Вместо сценариев стоит заглушка.",
                        evidence=["заполняются тестировщиками"],
                    )
                ],
                priority_actions=["Заполнить раздел тестирования."],
            )
        raise AssertionError(f"unexpected request: {response_model}")


def _extract_list_items(user_prompt: str) -> LLMExtractionResponse:
    """Treat every Markdown list item of the chunk as a requirement."""
    requirements = []
    for line in user_prompt.splitlines():
        if line.startswith("* "):
            text = line[2:].replace("*", "")
            requirements.append(
                LLMExtractedRequirement(
                    original_id=text.split(".")[0],
                    text=text,
                    category=RequirementCategory.FUNCTIONAL,
                )
            )
    return LLMExtractionResponse(requirements=requirements)


def _require_git() -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")


def _git(cwd: Path, *args: str) -> str:
    """Run git in ``cwd`` and return its stripped output."""
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fresh_repo(cwd: Path) -> None:
    """Create a repository with one empty commit at ``cwd``."""
    _git(cwd, "init", "-q")
    _git(
        cwd,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.test",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "init",
    )


class TestGetGitCommit:
    def test_marks_changes_in_the_agent_code_as_dirty(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        _require_git()
        agent_dir = tmp_path / "app"
        agent_dir.mkdir()
        _fresh_repo(tmp_path)
        monkeypatch.setattr("app.cli._AGENT_DIR", agent_dir)
        commit = _git(tmp_path, "rev-parse", "--short", "HEAD")

        assert get_git_commit() == commit

        (agent_dir / "new_module.py").write_text("", encoding="utf-8")
        assert get_git_commit() == f"{commit}-dirty"

    def test_unrelated_files_do_not_make_the_commit_dirty(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        _require_git()
        agent_dir = tmp_path / "app"
        agent_dir.mkdir()
        _fresh_repo(tmp_path)
        monkeypatch.setattr("app.cli._AGENT_DIR", agent_dir)
        (tmp_path / "report.md").write_text("generated", encoding="utf-8")

        assert get_git_commit() == _git(tmp_path, "rev-parse", "--short", "HEAD")

    def test_unknown_outside_a_git_repository(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("app.cli._AGENT_DIR", tmp_path)

        assert get_git_commit() == "unknown"
