"""Unit tests for turning lint flags into set-finding titles."""

from __future__ import annotations

from app.models.analysis import QualityCriterion, Severity
from app.models.lint import LintCategory, LintFlag
from app.services.lint_merge import merge_document_flags


def _flag(message: str) -> LintFlag:
    return LintFlag(
        flag_id="LINT-001",
        category=LintCategory.HTTP_STATUS_MISMATCH,
        severity=Severity.MEDIUM,
        criterion=QualityCriterion.CONSISTENCY,
        message=message,
        evidence=message,
        recommendation="Привести код и описание в соответствие.",
    )


def _title(message: str) -> str:
    findings, added = merge_document_flags([], [_flag(message)], set())
    assert added == 1
    return findings[0].title


class TestFlagTitle:
    def test_dot_inside_quotes_is_not_a_sentence_boundary(self) -> None:
        message = (
            "Заголовок «#### 200. Successful — Created» "
            "использует название статуса 201 (created)."
        )

        title = _title(message)

        assert title == message
        assert "#### 200. Successful — Created" in title

    def test_colon_outside_quotes_ends_the_title(self) -> None:
        message = (
            "В разделе 403 указан другой статус 404: "
            "«Для чужой карты возвращается 404»."
        )

        assert _title(message) == "В разделе 403 указан другой статус 404"

    def test_long_title_is_not_cut_inside_quotes(self) -> None:
        message = "Заголовок «" + "очень длинная цитата " * 10 + "» и пояснение."

        title = _title(message)

        assert title.endswith("...")
        assert len(title) <= 90 + len("...")
        assert title.count("«") == title.count("»")
