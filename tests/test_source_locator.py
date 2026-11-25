"""Unit tests for deterministic source location lookup."""

from pathlib import Path

from app.models.requirement import DocumentFormat, SourceDocument, SourceLocation
from app.services.source_locator import SourceLocator

MARKDOWN = """# Часть A

Вводный текст.

## Функциональные требования

* **Ф-1.** Система должна отправлять push.

## Требования к данным

Хранить идентификатор карты.
"""


class TestMarkdownLocator:
    def test_finds_section_and_nearest_heading(self) -> None:
        locator = SourceLocator(
            SourceDocument(
                path=Path("requirements.md"),
                format=DocumentFormat.MARKDOWN,
                text=MARKDOWN,
            )
        )

        location = locator.locate("Ф-1. Система должна отправлять push.")

        assert location is not None
        assert location.section == "Часть A > Функциональные требования"
        assert location.heading == "Функциональные требования"

    def test_finds_second_section(self) -> None:
        locator = SourceLocator(
            SourceDocument(
                path=Path("requirements.md"),
                format=DocumentFormat.MARKDOWN,
                text=MARKDOWN,
            )
        )

        location = locator.locate("Хранить идентификатор карты.")

        assert location is not None
        assert location.section == "Часть A > Требования к данным"

    def test_intro_before_first_heading_has_no_section(self) -> None:
        locator = SourceLocator(
            SourceDocument(
                path=Path("requirements.md"),
                format=DocumentFormat.MARKDOWN,
                text="Вводный текст.\n\n" + MARKDOWN,
            )
        )

        location = locator.locate("Вводный текст.")

        assert location == SourceLocation()

    def test_normalized_whitespace_and_case_are_tolerated(self) -> None:
        locator = SourceLocator(
            SourceDocument(
                path=Path("requirements.md"),
                format=DocumentFormat.MARKDOWN,
                text=MARKDOWN,
            )
        )

        location = locator.locate("ф-1.  система должна отправлять push.")

        assert location is not None
        assert location.section == "Часть A > Функциональные требования"

    def test_unknown_text_returns_none(self) -> None:
        locator = SourceLocator(
            SourceDocument(
                path=Path("requirements.md"),
                format=DocumentFormat.MARKDOWN,
                text=MARKDOWN,
            )
        )

        assert locator.locate("Такого текста нет.") is None


class TestPdfLocator:
    def test_finds_page_number(self) -> None:
        locator = SourceLocator(
            SourceDocument(
                path=Path("requirements.pdf"),
                format=DocumentFormat.PDF,
                text="Первое требование.\n\nВторое требование.",
                pages=["Первое требование.", "Второе требование."],
            )
        )

        location = locator.locate("Второе требование.")

        assert location is not None
        assert location.page == 2

    def test_unknown_text_returns_none(self) -> None:
        locator = SourceLocator(
            SourceDocument(
                path=Path("requirements.pdf"),
                format=DocumentFormat.PDF,
                text="Первое требование.",
                pages=["Первое требование."],
            )
        )

        assert locator.locate("Нет такого текста.") is None
