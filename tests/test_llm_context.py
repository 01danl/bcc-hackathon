"""Unit tests for the shared prefix used by prefix caching."""

from __future__ import annotations

from app.prompts import load_prompt
from app.services.llm_context import build_document_system_prompt


class TestDocumentSystemPrompt:
    def test_prefix_is_byte_identical_for_repeated_calls(self) -> None:
        base = load_prompt("auditor_system")
        document = "# ТЗ\n\n* Ф-1. Система отправляет push.\n"

        first = build_document_system_prompt(base, document)
        second = build_document_system_prompt(base, document)

        assert first == second
        assert first.startswith(base.rstrip())
        assert document in first
        assert first.rstrip().endswith("<<<END_DOCUMENT>>>")

    def test_prefix_contains_no_task_specific_text(self) -> None:
        prefix = build_document_system_prompt("system", "document")

        assert "BATCH REQUIREMENTS" not in prefix
        assert "EXTRACTED REQUIREMENTS" not in prefix
        assert "TASK" not in prefix
