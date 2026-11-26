"""Unit tests for the deterministic pre-LLM linter."""

from __future__ import annotations

from app.models.analysis import QualityCriterion, Severity
from app.models.lint import LintCategory
from app.models.requirement import Requirement, RequirementCollection
from app.services.static_linter import StaticLinter


def _collection(*texts: str) -> RequirementCollection:
    return RequirementCollection(
        requirements=[
            Requirement(id=f"REQ-{number:03d}", text=text)
            for number, text in enumerate(texts, start=1)
        ]
    )


class TestVagueWording:
    def test_vague_words_are_bound_to_requirements(self) -> None:
        collection = _collection(
            "Система должна работать быстро.",
            "Интерфейс должен быть удобным и современным.",
            "При необходимости отправлять push.",
            "Порог задаётся в тенге.",
        )
        document = "\n".join(item.text for item in collection.requirements)

        result = StaticLinter().analyze(document, collection)

        by_requirement = {
            flag.requirement_id: flag for flag in result.requirement_flags
        }
        assert by_requirement["REQ-001"].category is LintCategory.VAGUE_WORDING
        assert by_requirement["REQ-001"].severity is Severity.HIGH
        assert (
            by_requirement["REQ-001"].criterion
            is QualityCriterion.VERIFIABILITY
        )
        assert by_requirement["REQ-002"].severity is Severity.MEDIUM
        assert by_requirement["REQ-003"].criterion is QualityCriterion.COMPLETENESS
        assert "REQ-004" not in by_requirement
        assert result.suspicious_ids == {"REQ-001", "REQ-002", "REQ-003"}

    def test_clear_requirement_has_no_flag(self) -> None:
        collection = _collection(
            "Система возвращает список операций за последние 30 дней."
        )

        result = StaticLinter().analyze(
            collection.requirements[0].text, collection
        )

        assert result.all_flags == []


class TestDocumentTemplates:
    DOCUMENT = """# Требования

## Проверка

| Проверка | Создано | Проверено |
| :--- | :--- | :--- |
| Требование зафиксировано | Да / Нет | Да / Нет |
|  |  |  |

## Тестирование

заполняются тестировщиками
"""

    def test_empty_template_empty_row_and_stub_are_reported(self) -> None:
        result = StaticLinter().analyze(self.DOCUMENT, RequirementCollection())

        categories = {flag.category for flag in result.document_flags}
        assert LintCategory.EMPTY_TEMPLATE in categories
        assert LintCategory.EMPTY_TABLE_CELL in categories
        assert LintCategory.STUB_TEXT in categories
        assert all(flag.evidence for flag in result.document_flags)
        assert len({flag.flag_id for flag in result.all_flags}) == len(
            result.all_flags
        )

    def test_vertical_key_value_table_reports_missing_url(self) -> None:
        document = """# API

| Поле | Значение |
| :--- | :--- |
| **Request URL** | Dev: `/alerts/saveLimit` . Prod: — |
| **Swagger** | — |
"""

        result = StaticLinter().analyze(document, RequirementCollection())

        messages = " ".join(flag.message for flag in result.document_flags)
        assert "Request URL" in messages
        assert "Swagger" in messages


class TestHttpConsistency:
    def test_status_in_body_contradicts_the_section_heading(self) -> None:
        document = """# API

#### 403. Forbidden

**404** Для чужой карты возвращается 404, как и для отсутствующей карты.
"""

        result = StaticLinter().analyze(document, RequirementCollection())

        mismatches = [
            flag
            for flag in result.document_flags
            if flag.category is LintCategory.HTTP_STATUS_MISMATCH
        ]
        assert mismatches
        assert "404" in mismatches[0].message
        assert mismatches[0].severity is Severity.HIGH

    def test_heading_reason_must_match_the_code(self) -> None:
        document = "#### 200. Successful — Created\n"

        result = StaticLinter().analyze(document, RequirementCollection())

        assert any(
            flag.category is LintCategory.HTTP_STATUS_MISMATCH
            for flag in result.document_flags
        )

    def test_get_is_rejected_for_a_save_endpoint(self) -> None:
        document = """# API

| Поле | Значение |
| :--- | :--- |
| **Description** | Сохранение порога |
| **REST type** | GET |
| **Request URL** | Dev: `/alerts/saveLimit` |
"""

        result = StaticLinter().analyze(document, RequirementCollection())

        assert any(
            flag.category is LintCategory.HTTP_METHOD_MISMATCH
            for flag in result.document_flags
        )
