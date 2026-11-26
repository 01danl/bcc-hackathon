"""Merge deterministic lint flags into model findings.

The linter is not a replacement for the auditor: it catches only obvious
defects. But a model must not be able to hide those defects by omitting them.
Every flag the model did not explicitly reject is therefore converted into a
regular finding with ``origin=static_linter``. Findings that the model already
reported with overlapping evidence are not duplicated.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.models.analysis import FindingOrigin, LLMBatchFinding
from app.models.lint import LintCategory, LintFlag
from app.models.set_audit import SetFinding, SetFindingCategory
from app.services.evidence import normalize_for_matching

_SET_CATEGORY_BY_LINT = {
    LintCategory.EMPTY_TEMPLATE: SetFindingCategory.INCOMPLETE_SECTION,
    LintCategory.EMPTY_TABLE_CELL: SetFindingCategory.INCOMPLETE_SECTION,
    LintCategory.STUB_TEXT: SetFindingCategory.INCOMPLETE_SECTION,
    LintCategory.UNFILLED_PLACEHOLDER: SetFindingCategory.INCOMPLETE_SECTION,
    LintCategory.HTTP_STATUS_MISMATCH: SetFindingCategory.INTEGRATION_MISMATCH,
    LintCategory.HTTP_METHOD_MISMATCH: SetFindingCategory.INTEGRATION_MISMATCH,
    LintCategory.VAGUE_WORDING: SetFindingCategory.AMBIGUOUS_REQUIREMENT,
}

_TITLE_LIMIT = 90
_QUOTED_SPAN_RE = re.compile(r"«[^»]*»")


def merge_requirement_flags(
    findings: Sequence[LLMBatchFinding],
    flags: Sequence[LintFlag],
    rejected_flag_ids: set[str],
) -> tuple[list[LLMBatchFinding], int]:
    """Add confirmed requirement-level lint flags to model findings.

    Returns the merged findings and the number of lint findings added.
    """
    merged = list(findings)
    added = 0
    for flag in flags:
        if flag.flag_id in rejected_flag_ids or not flag.requirement_id:
            continue
        if _requirement_duplicate(merged, flag):
            continue
        merged.append(
            LLMBatchFinding(
                requirement_id=flag.requirement_id,
                evidence=flag.evidence,
                explanation=flag.message,
                criterion=flag.criterion,
                severity=flag.severity,
                recommendation=flag.recommendation,
                origin=FindingOrigin.STATIC_LINTER,
            )
        )
        added += 1
    return merged, added


def merge_document_flags(
    findings: Sequence[SetFinding],
    flags: Sequence[LintFlag],
    rejected_flag_ids: set[str],
) -> tuple[list[SetFinding], int]:
    """Add confirmed document-level lint flags to set findings.

    Returns the merged findings and the number of lint findings added.
    """
    merged = list(findings)
    added = 0
    for flag in flags:
        if flag.flag_id in rejected_flag_ids:
            continue
        category = _SET_CATEGORY_BY_LINT[flag.category]
        if _set_duplicate(merged, flag):
            continue
        merged.append(
            SetFinding(
                evidence=[flag.evidence],
                description=flag.message,
                category=category,
                severity=flag.severity,
                title=_flag_title(flag),
                related_req_ids=list(flag.related_req_ids),
                origin=FindingOrigin.STATIC_LINTER,
            )
        )
        added += 1
    return merged, added


def _requirement_duplicate(
    findings: Sequence[LLMBatchFinding], flag: LintFlag
) -> bool:
    """Return True when the model already reported this flag's issue."""
    evidence = normalize_for_matching(flag.evidence)
    if not evidence:
        return False
    for finding in findings:
        if finding.criterion is not flag.criterion:
            continue
        other = normalize_for_matching(finding.evidence)
        if other and (evidence in other or other in evidence):
            return True
    return False


def _set_duplicate(findings: Sequence[SetFinding], flag: LintFlag) -> bool:
    """Return True when a set finding already quotes the flagged fragment."""
    evidence = normalize_for_matching(flag.evidence)
    if not evidence:
        return False
    for finding in findings:
        for quote in finding.evidence:
            other = normalize_for_matching(quote)
            if other and (evidence in other or other in evidence):
                return True
    return False


def _flag_title(flag: LintFlag) -> str:
    """Build a short title without cutting a quoted fragment.

    A dot inside quotes (for example ``#### 200. Successful — Created``) is
    not a sentence boundary, so the old ``split('. ')`` produced broken
    titles. Prefer the text before the first colon outside quotes; when there
    is no such colon, shorten to 90 characters on a word boundary outside
    quotes.
    """
    text = " ".join(flag.message.split())
    title = _before_unquoted_colon(text)
    return _shorten_title(title) or flag.category.value


def _before_unquoted_colon(text: str) -> str:
    """Return the part before the first ``:`` that is not inside ``«...»``."""
    depth = 0
    for index, char in enumerate(text):
        if char == "«":
            depth += 1
        elif char == "»":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            return text[:index].strip()
    return text


def _shorten_title(text: str, limit: int = _TITLE_LIMIT) -> str:
    """Shorten ``text`` on a word boundary without breaking a quoted span."""
    text = text.strip()
    if len(text) <= limit:
        return text

    spans = [(match.start(), match.end()) for match in _QUOTED_SPAN_RE.finditer(text)]
    candidates = [
        index
        for index, char in enumerate(text)
        if index <= limit and char.isspace() and _is_safe_cut(text, index, spans)
    ]
    cut = candidates[-1] if candidates else limit
    while cut > 0 and not _is_safe_cut(text, cut, spans):
        cut -= 1
    return text[:cut].rstrip(" ,.;:—-") + "..."


def _is_safe_cut(
    text: str, position: int, spans: list[tuple[int, int]]
) -> bool:
    """Return True when cutting at ``position`` leaves quotes intact."""
    for start, end in spans:
        if start < position < end:
            return False
    return text.count("«", 0, position) <= text.count("»", 0, position)
