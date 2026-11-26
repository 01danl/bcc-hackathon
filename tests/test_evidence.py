"""Unit tests for evidence verification against a source document."""

from app.services.evidence import (
    align_quote,
    canonicalize_quotes,
    find_missing_quotes,
    normalize_for_matching,
)


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


class TestFuzzyAlignment:
    SOURCE = (
        "* **Ф-1.** Держатель активной карты KZT задаёт целый порог "
        "от 1 000 до 1 000 000 ₸ включительно."
    )

    def test_one_letter_difference_returns_the_source_slice(self) -> None:
        quote = (
            "Держатель активной карты KZT задает целый порог "
            "от 1 000 до 1 000 000 ₸"
        )

        match = align_quote(quote, self.SOURCE)

        assert match is not None
        assert match.score >= 0.92
        assert match.matched_text == (
            "Держатель активной карты KZT задаёт целый порог "
            "от 1 000 до 1 000 000 ₸"
        )
        assert "**" not in match.matched_text

    def test_changed_amount_is_rejected(self) -> None:
        quote = (
            "Держатель активной карты KZT задаёт целый порог "
            "от 1 000 до 2 000 000 ₸"
        )

        assert align_quote(quote, self.SOURCE) is None

    def test_changed_endpoint_is_rejected(self) -> None:
        source = "Сервис сохраняет порог через /alerts/saveLimit."
        quote = "Сервис сохраняет порог через /alerts/deleteLimit."

        assert align_quote(quote, source) is None

    def test_canonicalize_returns_unique_source_slices(self) -> None:
        quotes = [
            "Держатель активной карты KZT задает целый порог",
            "Держатель активной карты KZT задаёт целый порог",
            "этой фразы нет",
        ]

        aligned, missing = canonicalize_quotes(quotes, self.SOURCE)

        assert len(aligned) == 1
        assert aligned[0].startswith("Держатель активной карты KZT задаёт")
        assert missing == ["этой фразы нет"]
