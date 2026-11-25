"""CLI integration tests with the LLM services stubbed out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from app.cli import app
from app.config import Settings
from app.llm.client import LLMRequestError
from app.models.analysis import (
    RequirementAnalysis,
    RequirementAnalysisCollection,
    RequirementStatus,
)
from app.models.requirement import (
    DocumentFormat,
    Requirement,
    RequirementCollection,
    SourceDocument,
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
