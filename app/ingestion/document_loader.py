"""Loading of requirement documents from disk.

Supports PDF (via PyMuPDF) and Markdown. The loader is intentionally
dumb about requirements: it only extracts text and, for PDFs, preserves
per-page text. Requirement detection lives in the extraction service.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

from app.models.requirement import DocumentFormat, SourceDocument

_SUFFIX_TO_FORMAT: dict[str, DocumentFormat] = {
    ".pdf": DocumentFormat.PDF,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
}


class DocumentLoaderError(Exception):
    """Base class for document loading errors."""


class DocumentNotFoundError(DocumentLoaderError):
    """The requested file does not exist."""


class UnsupportedFormatError(DocumentLoaderError):
    """The file format is not supported."""


class DocumentReadError(DocumentLoaderError):
    """The file exists but could not be read or parsed."""


class DocumentLoader:
    """Load requirement documents (PDF or Markdown) as :class:`SourceDocument`."""

    def load(self, path: str | Path) -> SourceDocument:
        """Load a document from ``path``.

        Args:
            path: path to a PDF or Markdown file.

        Returns:
            Loaded document with full text (and per-page text for PDFs).

        Raises:
            DocumentNotFoundError: if the file does not exist.
            DocumentLoaderError: if the path points to a directory.
            UnsupportedFormatError: if the file type is neither PDF nor Markdown.
            DocumentReadError: if the file cannot be read or parsed.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise DocumentNotFoundError(f"Файл не найден: {file_path}")
        if file_path.is_dir():
            raise DocumentLoaderError(f"Указан каталог, а не файл: {file_path}")

        document_format = _SUFFIX_TO_FORMAT.get(file_path.suffix.lower())
        if document_format is None:
            supported = ", ".join(sorted(_SUFFIX_TO_FORMAT))
            raise UnsupportedFormatError(
                f"Неподдерживаемый формат файла: {file_path.name}. "
                f"Поддерживаются: {supported}"
            )

        if document_format is DocumentFormat.PDF:
            return self._load_pdf(file_path)
        return self._load_markdown(file_path)

    def _load_pdf(self, path: Path) -> SourceDocument:
        """Extract text page by page from a PDF file."""
        try:
            with pymupdf.open(str(path)) as pdf:
                pages = [page.get_text() for page in pdf]
        except pymupdf.FileDataError as exc:
            raise DocumentReadError(f"Не удалось прочитать PDF: {path}") from exc
        return SourceDocument(
            path=path,
            format=DocumentFormat.PDF,
            text="\n\n".join(pages),
            pages=pages,
        )

    def _load_markdown(self, path: Path) -> SourceDocument:
        """Read a Markdown file as UTF-8 text."""
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise DocumentReadError(f"Не удалось прочитать файл: {path}") from exc
        return SourceDocument(
            path=path,
            format=DocumentFormat.MARKDOWN,
            text=text,
        )
