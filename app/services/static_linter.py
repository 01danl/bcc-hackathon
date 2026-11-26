"""Cheap deterministic checks that run before any LLM request.

The linter catches defects that do not need a reasoning model: unfilled
templates, stub sections, empty table rows, vague wording and internal API
contradictions. It costs milliseconds and no tokens. Its findings are passed
to the auditor as ``pre_detected_flags``; confirmed flags are kept even when
the model forgets to repeat them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.models.analysis import QualityCriterion, Severity
from app.models.lint import LintCategory, LintFlag, LintResult
from app.models.requirement import RequirementCollection
from app.services.evidence import normalize_for_matching

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_HEADING_CODE_RE = re.compile(r"^\s*#{1,6}\s*(\d{3})\b(.*)$")
_HTTP_CODE_RE = re.compile(r"\b([1-5]\d{2})\b")
_STUB_RE = re.compile(
    r"заполня[а-яё]*\s+(?:тестировщик|аналитик|разработчик|команд|вручную)"
    r"|\b(?:TBD|TODO|FIXME)\b"
    r"|документ\s+\(версия/дата\)\s+в\s+источнике\s+не\s+заполнен"
    r"|отметки\s+не\s+проставлены",
    re.IGNORECASE,
)
_TEMPLATE_CELL_RE = re.compile(r"\bДа\s*/\s*Нет\b", re.IGNORECASE)
_EMPTY_CELLS = frozenset({"", "—", "-", "–", "n/a", "н/д", "нет данных"})
_PLACEHOLDER_HEADERS = frozenset(
    {"request url", "swagger", "prod", "url", "gitlab project", "framework"}
)
_REQUIRED_HEADERS = (
    "фио",
    "подпись",
    "значение",
    "обязательна",
    "mandatory",
    "проверено",
    "создано",
)
_MUTATING_WORDS_RE = re.compile(
    r"\b(save|create|update|delete|patch|сохран\w*|созда\w*|обнов\w*|удал\w*)",
    re.IGNORECASE,
)
_READING_WORDS_RE = re.compile(
    r"\b(get|read|list|fetch|получ\w*|чит\w*|список|загруз\w*)",
    re.IGNORECASE,
)
_HTTP_REASONS = {
    200: {"ok", "successful", "success"},
    201: {"created"},
    400: {"bad request", "bad"},
    401: {"unauthorized"},
    403: {"forbidden"},
    404: {"not found"},
    500: {"internal server error", "server error"},
}
_BODY_STATUS_WORDS_RE = re.compile(
    r"возвраща|ответ|статус|код|response|return|status|error", re.IGNORECASE
)
_STATUS_LINE_RE = re.compile(r"^\s*\*\*\s*[1-5]\d{2}\.\s*\*\*")


@dataclass(frozen=True)
class _VagueRule:
    key: str
    pattern: re.Pattern[str]
    criterion: QualityCriterion
    severity: Severity
    message: str
    recommendation: str


_VAGUE_RULES = (
    _VagueRule(
        "speed",
        re.compile(
            r"\b(?:быстро|быстрый|быстрая|быстрое|быстрые|быстрее|быстрей"
            r"|fast|quickly|quick|rapid)\b",
            re.IGNORECASE,
        ),
        QualityCriterion.VERIFIABILITY,
        Severity.HIGH,
        "Нечёткая скорость: нет измеримого предела времени.",
        "Укажите измеримый предел, например время отклика в секундах.",
    ),
    _VagueRule(
        "convenience",
        re.compile(
            r"\b(?:удобн\w*|эффективн\w*|качественн\w*|надёжн\w*|надежн\w*"
            r"|интуитивн\w*|современн\w*|гибк\w*|прост\w*|легк\w*"
            r"|user[- ]friendly|convenient|efficient|reliable|intuitive"
            r"|flexible|easy|modern)\b",
            re.IGNORECASE,
        ),
        QualityCriterion.CLARITY,
        Severity.MEDIUM,
        "Субъективная оценка без измеримого критерия.",
        "Замените оценку проверяемым условием или метрикой.",
    ),
    _VagueRule(
        "timing",
        re.compile(
            r"\b(?:своевременн\w*|оперативн\w*|timely|asap)\b",
            re.IGNORECASE,
        ),
        QualityCriterion.VERIFIABILITY,
        Severity.HIGH,
        "Срок не задан: «своевременно» нельзя проверить.",
        "Укажите конкретный срок или максимальную задержку.",
    ),
    _VagueRule(
        "bounded",
        re.compile(
            r"\b(?:минимальн\w*|максимальн\w*|оптимальн\w*|достаточн\w*"
            r"|приемлем\w*|приемлем\w*|адекватн\w*|корректн\w*"
            r"|minimal|maximal|optimal|sufficient|acceptable|adequate|correct)\b",
            re.IGNORECASE,
        ),
        QualityCriterion.VERIFIABILITY,
        Severity.MEDIUM,
        "Граница не задана: «минимально/достаточно» нельзя измерить.",
        "Укажите конкретное значение, диапазон или правило выбора.",
    ),
    _VagueRule(
        "optional",
        re.compile(
            r"\b(?:при необходимости|по возможности|как правило|если нужно"
            r"|if necessary|if possible|as needed|as required)\b",
            re.IGNORECASE,
        ),
        QualityCriterion.COMPLETENESS,
        Severity.MEDIUM,
        "Необязательное поведение не определено.",
        "Опишите условие, при котором действие обязательно, и поведение иначе.",
    ),
    _VagueRule(
        "open_list",
        re.compile(
            r"\b(?:и\s*т\.?\s*д\.?|и\s*др\.?|etc\.?|and/or)\b",
            re.IGNORECASE,
        ),
        QualityCriterion.COMPLETENESS,
        Severity.MEDIUM,
        "Открытый перечень допускает произвольное расширение.",
        "Перечислите полный набор значений или задайте правило расширения.",
    ),
    _VagueRule(
        "approximate",
        re.compile(
            r"\b(?:примерно|приблизительно|около|approximately|about|around)\b",
            re.IGNORECASE,
        ),
        QualityCriterion.VERIFIABILITY,
        Severity.MEDIUM,
        "Приблизительное значение не позволяет проверить результат.",
        "Укажите точное значение или допустимый диапазон.",
    ),
)


@dataclass
class _DraftFlag:
    category: LintCategory
    severity: Severity
    criterion: QualityCriterion
    message: str
    evidence: str
    recommendation: str
    requirement_id: str | None = None
    related_req_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _TableRow:
    line_number: int
    line: str
    cells: list[str]


class StaticLinter:
    """Run all deterministic checks over one document and its requirements."""

    def analyze(
        self, document_text: str, collection: RequirementCollection
    ) -> LintResult:
        """Return requirement-level and document-level flags."""
        requirement_drafts = _lint_requirements(collection)
        document_drafts = _lint_document(document_text)
        _attach_related_ids(document_drafts, collection)
        document_flags = _materialize(document_drafts, start=1)
        requirement_flags = _materialize(
            requirement_drafts, start=len(document_flags) + 1
        )
        return LintResult(
            requirement_flags=requirement_flags,
            document_flags=document_flags,
        )


def _lint_requirements(collection: RequirementCollection) -> list[_DraftFlag]:
    drafts: list[_DraftFlag] = []
    for requirement in collection.requirements:
        text = requirement.text
        for rule in _VAGUE_RULES:
            if rule.pattern.search(text):
                drafts.append(
                    _DraftFlag(
                        category=LintCategory.VAGUE_WORDING,
                        severity=rule.severity,
                        criterion=rule.criterion,
                        message=f"{rule.message} Фрагмент: «{_match_text(rule, text)}».",
                        evidence=text,
                        recommendation=rule.recommendation,
                        requirement_id=requirement.id,
                    )
                )
        if _STUB_RE.search(text):
            drafts.append(
                _DraftFlag(
                    category=LintCategory.STUB_TEXT,
                    severity=Severity.HIGH,
                    criterion=QualityCriterion.COMPLETENESS,
                    message="Требование содержит заглушку вместо проверяемого условия.",
                    evidence=text,
                    recommendation="Замените заглушку конкретным требованием.",
                    requirement_id=requirement.id,
                )
            )
        if text.strip().casefold() in _EMPTY_CELLS:
            drafts.append(
                _DraftFlag(
                    category=LintCategory.EMPTY_TEMPLATE,
                    severity=Severity.HIGH,
                    criterion=QualityCriterion.COMPLETENESS,
                    message="Требование состоит только из пустого значения.",
                    evidence=text,
                    recommendation="Удалите пустую строку или заполните требование.",
                    requirement_id=requirement.id,
                )
            )
    return drafts


def _lint_document(document_text: str) -> list[_DraftFlag]:
    lines = document_text.splitlines()
    drafts: list[_DraftFlag] = []
    drafts.extend(_lint_tables(lines))
    drafts.extend(_lint_stubs(lines))
    drafts.extend(_lint_http(lines))
    drafts.extend(_lint_methods(lines))
    return _dedupe(drafts)


def _lint_tables(lines: list[str]) -> list[_DraftFlag]:
    drafts: list[_DraftFlag] = []
    for header, rows in _iter_table_blocks(lines):
        if not rows:
            continue
        header_cells = header.cells
        handled_lines: set[int] = set()

        template_rows = [
            row
            for row in rows
            if any(_TEMPLATE_CELL_RE.search(cell) for cell in row.cells)
        ]
        if template_rows:
            handled_lines.update(row.line_number for row in template_rows)
            drafts.append(
                _DraftFlag(
                    category=LintCategory.EMPTY_TEMPLATE,
                    severity=Severity.HIGH,
                    criterion=QualityCriterion.COMPLETENESS,
                    message=(
                        "В таблице "
                        f"{len(template_rows)} строк(и) с шаблоном «Да / Нет»: "
                        "отметки не проставлены."
                    ),
                    evidence=template_rows[0].line,
                    recommendation="Заполнить отметки или удалить пустой шаблон.",
                )
            )

        empty_rows = [
            row
            for row in rows
            if all(_is_empty_cell(cell) for cell in row.cells)
        ]
        if empty_rows:
            handled_lines.update(row.line_number for row in empty_rows)
            drafts.append(
                _DraftFlag(
                    category=LintCategory.EMPTY_TABLE_CELL,
                    severity=Severity.HIGH,
                    criterion=QualityCriterion.COMPLETENESS,
                    message=(
                        "В таблице есть полностью пустая строка: "
                        "значения не заполнены."
                    ),
                    evidence=empty_rows[0].line,
                    recommendation="Заполнить строку или удалить её из шаблона.",
                )
            )

        # Vertical key/value tables (``Поле | Значение``) keep the field name
        # in one column and its value in the next. Catch an unfilled
        # ``Request URL`` or ``Swagger`` before the generic required-column
        # check, which would only see the word "Значение".
        if len(header_cells) >= 2:
            for row in rows:
                if row.line_number in handled_lines or len(row.cells) < 2:
                    continue
                field_name = _normalize_header(row.cells[0])
                if (
                    field_name in _PLACEHOLDER_HEADERS
                    and _has_placeholder_value(row.cells[1])
                ):
                    handled_lines.add(row.line_number)
                    drafts.append(
                        _DraftFlag(
                            category=LintCategory.UNFILLED_PLACEHOLDER,
                            severity=Severity.HIGH,
                            criterion=QualityCriterion.COMPLETENESS,
                            message=(
                                f"Поле «{row.cells[0].strip('* ')}» не заполнено: "
                                "вместо значения стоит прочерк или заглушка."
                            ),
                            evidence=row.line,
                            recommendation=(
                                "Указать значение поля или объяснить, "
                                "почему оно не применимо."
                            ),
                        )
                    )

        required_columns = [
            index
            for index, cell in enumerate(header_cells)
            if _is_required_header(cell)
        ]
        if required_columns:
            offending = [
                row
                for row in rows
                if row.line_number not in handled_lines
                and any(
                    _is_empty_cell(row.cells[index])
                    for index in required_columns
                    if index < len(row.cells)
                )
            ]
            if offending:
                names = ", ".join(
                    header_cells[index].strip("* ")
                    for index in required_columns
                )
                drafts.append(
                    _DraftFlag(
                        category=LintCategory.UNFILLED_PLACEHOLDER,
                        severity=Severity.HIGH,
                        criterion=QualityCriterion.COMPLETENESS,
                        message=f"В таблице не заполнены обязательные поля: {names}.",
                        evidence=offending[0].line,
                        recommendation="Заполнить обязательные поля таблицы.",
                    )
                )
                handled_lines.update(row.line_number for row in offending)

        for index, cell in enumerate(header_cells):
            normalized_header = _normalize_header(cell)
            if normalized_header not in _PLACEHOLDER_HEADERS:
                continue
            offending = [
                row
                for row in rows
                if row.line_number not in handled_lines
                and index < len(row.cells)
                and _has_placeholder_value(row.cells[index])
            ]
            if not offending:
                continue
            drafts.append(
                _DraftFlag(
                    category=LintCategory.UNFILLED_PLACEHOLDER,
                    severity=Severity.HIGH,
                    criterion=QualityCriterion.COMPLETENESS,
                    message=(
                        f"Поле «{cell.strip('* ')}» не заполнено: "
                        "вместо значения стоит прочерк или заглушка."
                    ),
                    evidence=offending[0].line,
                    recommendation="Указать значение поля или объяснить, почему оно не применимо.",
                )
            )
            handled_lines.update(row.line_number for row in offending)

    return drafts


def _lint_stubs(lines: list[str]) -> list[_DraftFlag]:
    drafts: list[_DraftFlag] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or _TABLE_ROW_RE.match(stripped):
            continue
        if not _STUB_RE.search(stripped):
            continue
        drafts.append(
            _DraftFlag(
                category=LintCategory.STUB_TEXT,
                severity=Severity.HIGH,
                criterion=QualityCriterion.COMPLETENESS,
                message="Незаполненная заглушка вместо требования или сценария.",
                evidence=stripped,
                recommendation="Заменить заглушку конкретными требованиями или сценариями.",
            )
        )
    return drafts


def _lint_http(lines: list[str]) -> list[_DraftFlag]:
    drafts: list[_DraftFlag] = []
    for index, line in enumerate(lines):
        heading = _HEADING_CODE_RE.match(line)
        if heading is None:
            continue
        code = int(heading.group(1))
        title = heading.group(2).strip()
        reason = _reason_mismatch(code, title)
        if reason is not None:
            drafts.append(
                _DraftFlag(
                    category=LintCategory.HTTP_STATUS_MISMATCH,
                    severity=Severity.MEDIUM,
                    criterion=QualityCriterion.CONSISTENCY,
                    message=(
                        f"Заголовок «{line.strip()}» использует название "
                        f"статуса {reason[0]} ({reason[1]})."
                    ),
                    evidence=line.strip(),
                    recommendation="Привести HTTP-код и название статуса в соответствие.",
                )
            )

        for body_line in _section_body(lines, index + 1):
            for other in _HTTP_CODE_RE.findall(body_line):
                other_code = int(other)
                if other_code == code:
                    continue
                if not _BODY_STATUS_WORDS_RE.search(body_line):
                    continue
                drafts.append(
                    _DraftFlag(
                        category=LintCategory.HTTP_STATUS_MISMATCH,
                        severity=Severity.HIGH,
                        criterion=QualityCriterion.CONSISTENCY,
                        message=(
                            f"В разделе {code} указан другой статус {other_code}: "
                            f"«{body_line.strip()}»."
                        ),
                        evidence=body_line.strip(),
                        recommendation=(
                            "Согласовать статус в заголовке раздела и в описании поведения."
                        ),
                    )
                )
    return drafts


def _lint_methods(lines: list[str]) -> list[_DraftFlag]:
    document = "\n".join(lines)
    method_match = re.search(
        r"\|\s*\**\s*REST\s*type\s*\**\s*\|\s*([A-Za-z]+)\s*\|",
        document,
        re.IGNORECASE,
    )
    if method_match is None:
        return []
    method = method_match.group(1).upper()
    method_line = next(
        (line for line in lines if "rest type" in line.casefold()), ""
    )
    if not method_line:
        return []

    url_match = re.search(
        r"\|\s*\**\s*Request\s*URL\s*\**\s*\|([^\n]+)",
        document,
        re.IGNORECASE,
    )
    action_text = url_match.group(1) if url_match else ""
    if method in {"GET", "HEAD"} and _MUTATING_WORDS_RE.search(action_text):
        return [
            _DraftFlag(
                category=LintCategory.HTTP_METHOD_MISMATCH,
                severity=Severity.HIGH,
                criterion=QualityCriterion.CONSISTENCY,
                message=(
                    f"Метод {method} указан для изменяющей операции: "
                    f"«{action_text.strip()}». GET не должен менять состояние."
                ),
                evidence=method_line.strip(),
                recommendation=(
                    "Использовать POST/PUT/PATCH для изменения состояния "
                    "или переименовать операцию в чтение."
                ),
            )
        ]
    if method in {"POST", "PUT", "PATCH", "DELETE"} and _READING_WORDS_RE.search(
        action_text
    ):
        return [
            _DraftFlag(
                category=LintCategory.HTTP_METHOD_MISMATCH,
                severity=Severity.MEDIUM,
                criterion=QualityCriterion.CONSISTENCY,
                message=(
                    f"Метод {method} указан для операции чтения: "
                    f"«{action_text.strip()}»."
                ),
                evidence=method_line.strip(),
                recommendation="Использовать GET для чтения или уточнить назначение метода.",
            )
        ]
    return []


def _iter_table_blocks(
    lines: list[str],
) -> list[tuple[_TableRow, list[_TableRow]]]:
    blocks: list[tuple[_TableRow, list[_TableRow]]] = []
    block: list[_TableRow] = []
    for index, line in enumerate(lines):
        if _TABLE_ROW_RE.match(line):
            cells = _split_row(line)
            if cells and all(
                cell and _TABLE_SEPARATOR_RE.match(cell) for cell in cells
            ):
                continue
            block.append(_TableRow(index, line, cells))
            continue
        if block:
            blocks.append(_finish_block(block))
            block = []
    if block:
        blocks.append(_finish_block(block))
    return [item for item in blocks if item[0].cells and item[1]]


def _finish_block(block: list[_TableRow]) -> tuple[_TableRow, list[_TableRow]]:
    return block[0], block[1:]


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _section_body(lines: list[str], start: int) -> list[str]:
    body: list[str] = []
    for line in lines[start:]:
        if re.match(r"^\s*#{1,6}\s+", line):
            break
        # A new bold status line starts the next status block. The ``**404**``
        # line inside the 403 section has no trailing dot, so it is still
        # scanned and the 403/404 mismatch is reported.
        if _STATUS_LINE_RE.match(line):
            break
        body.append(line)
    return body


def _reason_mismatch(code: int, title: str) -> tuple[int, str] | None:
    normalized = title.casefold()
    for other_code, reasons in _HTTP_REASONS.items():
        if other_code == code:
            continue
        for reason in reasons:
            if _contains_reason(normalized, reason):
                return other_code, reason
    return None


def _contains_reason(text: str, reason: str) -> bool:
    """Match a reason phrase on word boundaries, not inside another word."""
    return re.search(rf"(?<!\w){re.escape(reason)}(?!\w)", text) is not None


def _attach_related_ids(
    drafts: list[_DraftFlag], collection: RequirementCollection
) -> None:
    normalized_requirements = [
        (item.id, normalize_for_matching(item.text))
        for item in collection.requirements
    ]
    for draft in drafts:
        if draft.requirement_id is not None:
            continue
        needle = normalize_for_matching(draft.evidence)
        if not needle:
            continue
        related = [
            requirement_id
            for requirement_id, text in normalized_requirements
            if needle in text
        ]
        draft.related_req_ids = related[:5]


def _materialize(drafts: list[_DraftFlag], *, start: int) -> list[LintFlag]:
    flags: list[LintFlag] = []
    for offset, draft in enumerate(drafts):
        flags.append(
            LintFlag(
                flag_id=f"LINT-{start + offset:03d}",
                category=draft.category,
                severity=draft.severity,
                criterion=draft.criterion,
                message=draft.message,
                evidence=draft.evidence,
                recommendation=draft.recommendation,
                requirement_id=draft.requirement_id,
                related_req_ids=draft.related_req_ids,
            )
        )
    return flags


def _dedupe(drafts: list[_DraftFlag]) -> list[_DraftFlag]:
    unique: list[_DraftFlag] = []
    seen: set[tuple[LintCategory, str, str | None]] = set()
    for draft in drafts:
        key = (draft.category, draft.evidence, draft.requirement_id)
        if key in seen:
            continue
        seen.add(key)
        unique.append(draft)
    return unique


def _match_text(rule: _VagueRule, text: str) -> str:
    match = rule.pattern.search(text)
    return match.group(0) if match is not None else ""


def _normalize_header(value: str) -> str:
    return " ".join(value.strip("* ").casefold().split())


def _is_required_header(value: str) -> bool:
    normalized = _normalize_header(value)
    return any(marker in normalized for marker in _REQUIRED_HEADERS)


def _is_empty_cell(value: str) -> bool:
    return value.strip().casefold() in _EMPTY_CELLS


def _has_placeholder_value(value: str) -> bool:
    stripped = value.strip()
    if _is_empty_cell(stripped):
        return True
    return bool(re.search(r"(?:^|[\s:])—(?:$|[\s.,])|\b(?:TBD|TODO)\b", stripped))
