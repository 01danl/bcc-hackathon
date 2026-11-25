"""Unit tests for the LLM extraction path (fully mocked model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.models.analysis import LLMExtractedRequirement, LLMExtractionResponse
from app.models.requirement import (
    DocumentFormat,
    RequirementCategory,
    SourceDocument,
    SourceLocation,
)
from app.services.chunking import DocumentChunker
from app.services.requirement_extractor import RequirementExtractor


class FakeClient:
    def __init__(self, responses: list[LLMExtractionResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def complete_json(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _document(text: str) -> SourceDocument:
    return SourceDocument(
        path=Path("requirements.md"),
        format=DocumentFormat.MARKDOWN,
        text=text,
    )


class TestLlmExtraction:
    def test_assigns_ids_and_preserves_model_fields(self) -> None:
        document = _document(
            "# Требования\n\n"
            "## Функциональные требования\n\n"
            "* Ф-1. Система должна отправлять push.\n"
            "* Ф-2. Порог задаётся в тенге.\n"
        )
        client = FakeClient(
            [
                LLMExtractionResponse(
                    requirements=[
                        LLMExtractedRequirement(
                            original_id="Ф-1",
                            text="Ф-1. Система должна отправлять push.",
                            category=RequirementCategory.FUNCTIONAL,
                            source_location=SourceLocation(
                                section="Требования > Функциональные требования"
                            ),
                        ),
                        LLMExtractedRequirement(
                            original_id="Ф-2",
                            text="Ф-2. Порог задаётся в тенге.",
                            category=RequirementCategory.BUSINESS,
                        ),
                    ]
                )
            ]
        )
        extractor = RequirementExtractor(
            client,  # type: ignore[arg-type]
            DocumentChunker(max_chars=1000, overlap_chars=100),
        )

        collection = extractor.extract(document)

        assert [item.id for item in collection.requirements] == ["REQ-001", "REQ-002"]
        assert [item.original_id for item in collection.requirements] == ["Ф-1", "Ф-2"]
        assert collection.requirements[0].text == "Ф-1. Система должна отправлять push."
        assert (
            collection.requirements[0].category is RequirementCategory.FUNCTIONAL
        )
        assert collection.requirements[1].category is RequirementCategory.BUSINESS
        # Missing location falls back to the chunk's section.
        assert collection.requirements[1].location.section == (
            "Требования > Функциональные требования"
        )

    def test_overlapping_chunks_are_deduplicated(self) -> None:
        paragraphs = [
            "Первое требование с достаточно длинным текстом.",
            "Второе требование с достаточно длинным текстом.",
            "Третье требование с достаточно длинным текстом.",
        ]
        document = _document("\n\n".join(paragraphs))
        duplicate = LLMExtractionResponse(
            requirements=[
                LLMExtractedRequirement(
                    text="Первое требование с достаточно длинным текстом.",
                    category=RequirementCategory.FUNCTIONAL,
                )
            ]
        )
        client = FakeClient([duplicate, duplicate, duplicate, duplicate])
        extractor = RequirementExtractor(
            client,  # type: ignore[arg-type]
            DocumentChunker(max_chars=70, overlap_chars=25),
        )

        collection = extractor.extract(document)

        assert len(client.calls) > 1
        assert len(collection.requirements) == 1
        assert collection.requirements[0].id == "REQ-001"

    def test_no_requirements_yields_empty_collection(self) -> None:
        client = FakeClient([LLMExtractionResponse(requirements=[])])
        extractor = RequirementExtractor(
            client,  # type: ignore[arg-type]
            DocumentChunker(max_chars=1000, overlap_chars=100),
        )

        collection = extractor.extract(_document("# Пустой документ\n\nТекст."))

        assert len(collection) == 0

    def test_prompt_contains_section_metadata_and_fragment(self) -> None:
        document = _document("# Раздел\n\n* Требование.")
        client = FakeClient([LLMExtractionResponse(requirements=[])])
        extractor = RequirementExtractor(
            client,  # type: ignore[arg-type]
            DocumentChunker(max_chars=1000, overlap_chars=100),
        )

        extractor.extract(document)

        prompt = client.calls[0]["user_prompt"]
        assert "section: Раздел" in prompt
        assert "* Требование." in prompt
        assert "extract_requirements" not in prompt  # system prompt is separate
        assert "JSON" in client.calls[0]["system_prompt"]

    def test_validate_warns_about_missing_location(self) -> None:
        client = FakeClient(
            [
                LLMExtractionResponse(
                    requirements=[
                        LLMExtractedRequirement(
                            text="Требование без места.",
                            category=RequirementCategory.OTHER,
                        )
                    ]
                )
            ]
        )
        extractor = RequirementExtractor(
            client,  # type: ignore[arg-type]
            DocumentChunker(max_chars=1000, overlap_chars=100),
        )
        collection = extractor.extract(_document("Требование без места."))

        warnings = extractor.validate(collection)

        assert any("раздел или страницу" in warning for warning in warnings)
