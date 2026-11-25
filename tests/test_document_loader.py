"""Unit tests for the document loader."""

from pathlib import Path

import pymupdf
import pytest

from app.ingestion.document_loader import (
    DocumentLoader,
    DocumentLoaderError,
    DocumentNotFoundError,
    DocumentReadError,
    UnsupportedFormatError,
)
from app.models.requirement import DocumentFormat


@pytest.fixture()
def loader() -> DocumentLoader:
    return DocumentLoader()


class TestMarkdownLoading:
    def test_loads_markdown_text(self, loader: DocumentLoader, tmp_path: Path) -> None:
        path = tmp_path / "requirements.md"
        path.write_text(
            "# Требования\n\n* **Ф-1.** Система должна отправлять push.\n",
            encoding="utf-8",
        )

        document = loader.load(path)

        assert document.format is DocumentFormat.MARKDOWN
        assert document.path == path
        assert "Ф-1" in document.text
        assert document.pages == []
        assert document.page_count == 0

    def test_missing_file_raises(self, loader: DocumentLoader, tmp_path: Path) -> None:
        with pytest.raises(DocumentNotFoundError):
            loader.load(tmp_path / "missing.md")

    def test_directory_raises(self, loader: DocumentLoader, tmp_path: Path) -> None:
        with pytest.raises(DocumentLoaderError):
            loader.load(tmp_path)

    def test_unsupported_suffix_raises(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("текст", encoding="utf-8")

        with pytest.raises(UnsupportedFormatError) as exc_info:
            loader.load(path)

        assert ".md" in str(exc_info.value)

    def test_non_utf8_markdown_raises(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        path = tmp_path / "broken.md"
        path.write_bytes(b"\xff\xfe invalid utf-8")

        with pytest.raises(DocumentReadError):
            loader.load(path)


class TestPdfLoading:
    def test_loads_pdf_text_and_preserves_pages(
        self, loader: DocumentLoader, tmp_path: Path
    ) -> None:
        path = tmp_path / "requirements.pdf"
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((72, 72), "1. The system shall notify the user.")
            pdf.new_page().insert_text((72, 72), "2. The system shall store the limit.")
            pdf.save(str(path))

        document = loader.load(path)

        assert document.format is DocumentFormat.PDF
        assert document.page_count == 2
        assert "notify the user" in document.pages[0]
        assert "store the limit" in document.pages[1]
        assert "notify the user" in document.text

    def test_corrupt_pdf_raises(self, loader: DocumentLoader, tmp_path: Path) -> None:
        path = tmp_path / "broken.pdf"
        path.write_bytes(b"this is definitely not a pdf")

        with pytest.raises(DocumentReadError):
            loader.load(path)
