"""Unit tests for the placeholder requirement extractor."""

from pathlib import Path

from app.models.requirement import (
    DocumentFormat,
    RequirementCategory,
    SourceDocument,
)
from app.services.requirement_extractor import RequirementExtractor

SAMPLE_MARKDOWN = """# Раздел 1

Вводный абзац без маркеров не является требованием.

## Функциональные требования

* **Ф-1.** Система должна отправлять push при достижении порога.
* **Ф-2.** Порог задаётся в тенге.

1. Первый шаг интеграции.
2. Второй шаг.

Таблица тоже не попадает в выборку:

| Колонка | Значение |
| :--- | :--- |
| а | б |
"""


def _markdown_document(text: str) -> SourceDocument:
    return SourceDocument(
        path=Path("sample.md"),
        format=DocumentFormat.MARKDOWN,
        text=text,
    )


class TestRequirementExtractor:
    def test_extracts_list_items_with_sequential_ids(self) -> None:
        collection = RequirementExtractor().extract(_markdown_document(SAMPLE_MARKDOWN))

        assert [requirement.id for requirement in collection.requirements] == [
            "REQ-001",
            "REQ-002",
            "REQ-003",
            "REQ-004",
        ]

    def test_bullet_markers_and_bold_are_cleaned(self) -> None:
        collection = RequirementExtractor().extract(_markdown_document(SAMPLE_MARKDOWN))

        assert collection.requirements[0].text == (
            "Ф-1. Система должна отправлять push при достижении порога."
        )
        assert collection.requirements[2].text == "Первый шаг интеграции."

    def test_plain_paragraphs_and_tables_are_skipped(self) -> None:
        collection = RequirementExtractor().extract(_markdown_document(SAMPLE_MARKDOWN))

        texts = [requirement.text for requirement in collection.requirements]
        assert all("Вводный абзац" not in text for text in texts)
        assert all("Колонка" not in text for text in texts)

    def test_placeholder_does_not_classify_or_invent_ids(self) -> None:
        collection = RequirementExtractor().extract(_markdown_document(SAMPLE_MARKDOWN))

        for requirement in collection.requirements:
            assert requirement.category is RequirementCategory.OTHER
            assert requirement.original_id is None

    def test_markdown_locations_track_sections(self) -> None:
        collection = RequirementExtractor().extract(_markdown_document(SAMPLE_MARKDOWN))

        sections = {requirement.location.section for requirement in collection.requirements}
        assert sections == {"Функциональные требования"}
        assert all(requirement.location.page is None for requirement in collection.requirements)

    def test_empty_document_yields_empty_collection(self) -> None:
        collection = RequirementExtractor().extract(_markdown_document(""))

        assert len(collection) == 0

    def test_pdf_locations_track_pages(self) -> None:
        document = SourceDocument(
            path=Path("sample.pdf"),
            format=DocumentFormat.PDF,
            text="1. Один\n2. Два",
            pages=["1. Один\n2. Два"],
        )

        collection = RequirementExtractor().extract(document)

        assert len(collection) == 2
        assert [requirement.location.page for requirement in collection.requirements] == [1, 1]
        assert all(requirement.location.section is None for requirement in collection.requirements)
