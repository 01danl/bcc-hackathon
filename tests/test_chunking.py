"""Unit tests for heading-aware document chunking."""

from pathlib import Path

from app.models.requirement import DocumentFormat, SourceDocument
from app.services.chunking import DocumentChunker


def _markdown(text: str) -> SourceDocument:
    return SourceDocument(
        path=Path("requirements.md"),
        format=DocumentFormat.MARKDOWN,
        text=text,
    )


class TestMarkdownChunking:
    def test_heading_lines_are_kept_in_chunk_text(self) -> None:
        document = _markdown(
            "# Часть A\n\n"
            "Вводный текст.\n\n"
            "## Функциональные требования\n\n"
            "* Ф-1. Система должна отправлять push.\n"
            "* Ф-2. Порог задаётся в тенге.\n\n"
            "### Детали\n\n"
            "Уточняющий текст.\n"
        )

        chunks = DocumentChunker(max_chars=1000, overlap_chars=100).split(document)

        assert len(chunks) == 1
        assert "# Часть A" in chunks[0].text
        assert "## Функциональные требования" in chunks[0].text
        assert "### Детали" in chunks[0].text
        assert "Ф-1" in chunks[0].text
        assert chunks[0].location.section == "Часть A"

    def test_chunks_respect_max_size(self) -> None:
        paragraphs = [f"Абзац номер {number} с текстом требования." for number in range(10)]
        document = _markdown("# Требования\n\n" + "\n\n".join(paragraphs))
        chunker = DocumentChunker(max_chars=80, overlap_chars=20)

        chunks = chunker.split(document)

        assert len(chunks) > 1
        assert all(len(chunk.text) <= 80 for chunk in chunks)
        assert [chunk.index for chunk in chunks] == list(
            range(1, len(chunks) + 1)
        )

    def test_overlap_repeats_tail_of_previous_chunk(self) -> None:
        paragraphs = [
            "Первый короткий блок.",
            "Второй короткий блок.",
            "Третий короткий блок.",
            "Четвёртый короткий блок.",
        ]
        document = _markdown("\n\n".join(paragraphs))
        chunks = DocumentChunker(max_chars=60, overlap_chars=25).split(document)

        assert len(chunks) >= 2
        first_paragraphs = set(chunks[0].text.split("\n\n"))
        second_paragraphs = set(chunks[1].text.split("\n\n"))
        assert first_paragraphs & second_paragraphs

    def test_oversized_line_is_split_within_limit(self) -> None:
        long_line = "Очень длинное требование. " * 10
        document = _markdown(long_line)
        chunker = DocumentChunker(max_chars=50, overlap_chars=10)

        chunks = chunker.split(document)

        assert chunks
        assert all(len(chunk.text) <= 50 for chunk in chunks)

    def test_empty_markdown_yields_no_chunks(self) -> None:
        assert DocumentChunker().split(_markdown("")) == []


class TestPdfChunking:
    def test_each_page_keeps_its_number(self) -> None:
        document = SourceDocument(
            path=Path("requirements.pdf"),
            format=DocumentFormat.PDF,
            text="Страница 1\n\nСтраница 2",
            pages=[
                "1. Первое требование.",
                "2. Второе требование.",
            ],
        )

        chunks = DocumentChunker(max_chars=200, overlap_chars=20).split(document)

        assert len(chunks) == 2
        assert [chunk.location.page for chunk in chunks] == [1, 2]
        assert all(chunk.location.section is None for chunk in chunks)

    def test_long_pdf_page_is_split_but_stays_on_page(self) -> None:
        page = "\n\n".join(
            f"Требование номер {number} с описанием." for number in range(8)
        )
        document = SourceDocument(
            path=Path("requirements.pdf"),
            format=DocumentFormat.PDF,
            text=page,
            pages=[page],
        )

        chunks = DocumentChunker(max_chars=80, overlap_chars=20).split(document)

        assert len(chunks) > 1
        assert all(chunk.location.page == 1 for chunk in chunks)
        assert all(len(chunk.text) <= 80 for chunk in chunks)
