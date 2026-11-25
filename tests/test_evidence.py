"""Unit tests for evidence verification against a source document."""

from app.services.evidence import find_missing_quotes, normalize_for_matching


class TestNormalizeForMatching:
    def test_ignores_case_whitespace_and_markdown_markers(self) -> None:
        left = normalize_for_matching("* **Ф-1.**  Система   должна\nотправлять `push`.")
        right = normalize_for_matching("Ф-1. Система должна отправлять push.")

        assert left == right

    def test_text_is_trimmed(self) -> None:
        assert normalize_for_matching("  цитата\n") == "цитата"


class TestFindMissingQuotes:
    SOURCE = "## Цели\n\n* **Ф-1.** Держатель задаёт порог от 1 000 до 1 000 000 ₸."

    def test_quote_without_markdown_markers_is_found(self) -> None:
        quotes = ["Ф-1. Держатель задаёт порог", "от 1 000 до 1 000 000 ₸"]

        assert find_missing_quotes(quotes, self.SOURCE) == []

    def test_quote_with_different_case_and_line_breaks_is_found(self) -> None:
        quote = "ЗАДАЁТ ПОРОГ от  1 000"

        assert find_missing_quotes([quote], self.SOURCE) == []

    def test_only_the_missing_quotes_are_returned_in_order(self) -> None:
        quotes = ["этой фразы нет", "Держатель задаёт порог", "и этой нет"]

        assert find_missing_quotes(quotes, self.SOURCE) == [
            "этой фразы нет",
            "и этой нет",
        ]

    def test_empty_or_marker_only_quote_is_never_found(self) -> None:
        assert find_missing_quotes(["", "   ", "**", "_"], self.SOURCE) == [
            "",
            "   ",
            "**",
            "_",
        ]

    def test_word_boundaries_are_not_required(self) -> None:
        # Matching is a plain substring search: a mid-word slice is found too.
        assert find_missing_quotes(["ЖАТЕЛЬ зад"], self.SOURCE) == []
