"""Unit tests for the Pydantic data models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.requirement import (
    DocumentFormat,
    Requirement,
    RequirementCategory,
    RequirementCollection,
    SourceDocument,
    SourceLocation,
)


class TestRequirement:
    def test_minimal_requirement_uses_defaults(self) -> None:
        requirement = Requirement(id="REQ-001", text="Система должна отправлять push.")

        assert requirement.original_id is None
        assert requirement.category is RequirementCategory.OTHER
        assert requirement.location == SourceLocation()

    def test_full_requirement(self) -> None:
        requirement = Requirement(
            id="REQ-002",
            original_id="Ф-2",
            text="Порог задаётся в тенге.",
            category=RequirementCategory.FUNCTIONAL,
            location=SourceLocation(page=3, section="Требования"),
        )

        assert requirement.original_id == "Ф-2"
        assert requirement.category is RequirementCategory.FUNCTIONAL
        assert requirement.location.page == 3

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Requirement(id="REQ-001", text="")

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Requirement(id="", text="Система должна работать.")

    def test_page_number_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            SourceLocation(page=0)


class TestRequirementCollection:
    def test_empty_collection(self) -> None:
        collection = RequirementCollection()

        assert len(collection) == 0
        assert collection.by_category() == {}

    def test_duplicate_ids_rejected(self) -> None:
        first = Requirement(id="REQ-001", text="Первое требование.")
        second = Requirement(id="REQ-001", text="Второе требование.")

        with pytest.raises(ValidationError, match="REQ-001"):
            RequirementCollection(requirements=[first, second])

    def test_by_category_groups_requirements(self) -> None:
        functional = Requirement(
            id="REQ-001", text="Функциональное.", category=RequirementCategory.FUNCTIONAL
        )
        other = Requirement(id="REQ-002", text="Прочее.")
        collection = RequirementCollection(requirements=[functional, other])

        grouped = collection.by_category()

        assert set(grouped) == {RequirementCategory.FUNCTIONAL, RequirementCategory.OTHER}
        assert grouped[RequirementCategory.FUNCTIONAL] == [functional]
        assert grouped[RequirementCategory.OTHER] == [other]


class TestSourceDocument:
    def test_markdown_document(self) -> None:
        document = SourceDocument(
            path=Path("requirements.md"),
            format=DocumentFormat.MARKDOWN,
            text="# Требования",
        )

        assert document.page_count == 0
        assert document.pages == []

    def test_pdf_document_keeps_pages(self) -> None:
        document = SourceDocument(
            path=Path("requirements.pdf"),
            format=DocumentFormat.PDF,
            text="первая страница\n\nвторая страница",
            pages=["первая страница", "вторая страница"],
        )

        assert document.page_count == 2
        assert document.text.count("\n\n") == 1
