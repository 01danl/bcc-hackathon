"""LLM-based audit of a requirement set as a whole.

The auditor model receives the full document text in a stable system prefix and
the extracted requirements plus document-level lint flags in the user message.
It looks for defects that only show at the level of the whole set: contradictions
between sections, mismatches between business requirements and technical
specifications, missing cases, unfinished sections and uncovered goals.

The model's answer is not trusted blindly. Every evidence quote is fuzzy-aligned
with the document and replaced by the exact source slice; a quote that cannot be
aligned triggers one correction request. Findings that keep no verified quote are
excluded, confirmed lint flags are merged in, and the final verdict is computed
in Python from finding severities and the per-requirement statuses.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from app.llm.client import LLMClient, LLMResponseError
from app.models.analysis import RequirementAnalysis, RequirementStatus, Severity
from app.models.lint import LintFlag
from app.models.requirement import RequirementCollection
from app.models.set_audit import (
    LLMSetAuditResponse,
    RequirementSetAudit,
    SetFinding,
    VerdictStatus,
)
from app.prompts import load_prompt
from app.services.evidence import EvidenceAligner
from app.services.lint_merge import merge_document_flags
from app.services.llm_context import build_document_system_prompt

_BLOCKING_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})


class RequirementSetAnalyzer:
    """Audit the whole requirement set against the source document."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        max_response_retries: int = 1,
    ) -> None:
        self._llm_client = llm_client
        self._max_response_retries = max(0, max_response_retries)
        self._system_prompt = load_prompt("auditor_system")
        self._task_prompt = load_prompt("set_audit")

    def analyze(
        self,
        document_text: str,
        collection: RequirementCollection,
        analyses: Iterable[RequirementAnalysis],
        *,
        pre_detected_flags: Sequence[LintFlag] = (),
    ) -> RequirementSetAudit:
        """Audit the requirement set and derive the final verdict.

        Raises:
            LLMResponseError: if the model answer stays malformed.
            LLMRequestError: on network/provider failures.
        """
        flags = list(pre_detected_flags)
        system_prompt = build_document_system_prompt(
            self._system_prompt, document_text
        )
        base_prompt = _build_set_audit_prompt(
            collection, flags, self._task_prompt
        )
        aligner = EvidenceAligner(document_text)
        response = self._complete_verified(
            system_prompt, base_prompt, aligner
        )
        findings, excluded = _keep_supported_findings(
            response.findings, collection, aligner
        )
        known_flag_ids = {flag.flag_id for flag in flags}
        rejected = {
            flag_id
            for flag_id in response.rejected_flag_ids
            if flag_id in known_flag_ids
        }
        findings, static_findings = merge_document_flags(
            findings, flags, rejected
        )
        verdict, reason = self._compute_verdict(
            findings, analyses, excluded_findings=excluded
        )
        return RequirementSetAudit(
            summary=response.summary,
            goal_coverage=response.goal_coverage,
            findings=findings,
            priority_actions=response.priority_actions,
            rejected_flag_ids=sorted(rejected),
            static_findings=static_findings,
            verdict=verdict,
            verdict_reason=reason,
            excluded_findings=excluded,
        )

    def _complete_verified(
        self,
        system_prompt: str,
        base_prompt: str,
        aligner: EvidenceAligner,
    ) -> LLMSetAuditResponse:
        """Request the audit, repeating it once when the answer needs a correction.

        A corrected answer replaces the previous one. When the last answer
        still carries unverified quotes it is returned as is; the caller
        drops the findings that remain unsupported.
        """
        correction = ""
        response: LLMSetAuditResponse | None = None
        last_error: LLMResponseError | None = None

        for _ in range(self._max_response_retries + 1):
            try:
                response = self._llm_client.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=base_prompt + correction,
                    response_model=LLMSetAuditResponse,
                )
            except LLMResponseError as exc:
                last_error = exc
                correction = _retry_note(
                    f"Ответ не прошёл проверку схемы: {exc}. "
                    "Верни JSON строго по схеме."
                )
                continue
            correction = _retry_note(_correction_request(response, aligner))
            if not correction:
                return response

        if response is None:
            assert last_error is not None
            raise last_error
        return response

    def _compute_verdict(
        self,
        set_findings: Sequence[SetFinding],
        item_analyses: Sequence[RequirementAnalysis],
        excluded_findings: int = 0,
    ) -> tuple[VerdictStatus, str]:
        """Derive the verdict from severities and statuses, never from the model.

        HIGH/CRITICAL findings and failed requirements block the document;
        any other remark means it needs revision. Findings that could not
        be backed by a quote neither block nor clear the document.
        """
        has_blocking = any(
            finding.severity in _BLOCKING_SEVERITIES for finding in set_findings
        ) or any(
            analysis.status is RequirementStatus.FAIL for analysis in item_analyses
        )
        if has_blocking:
            return (
                VerdictStatus.BLOCKED,
                "Передавать ТЗ в разработку и тестирование нельзя: обнаружены "
                "критические противоречия или пробелы уровня HIGH/CRITICAL.",
            )

        has_remarks = bool(set_findings) or any(
            analysis.status is RequirementStatus.WARNING for analysis in item_analyses
        )
        if has_remarks:
            return (
                VerdictStatus.NEEDS_REVISION,
                "ТЗ требует уточнения неоднозначностей перед передачей в "
                "разработку (есть замечания уровня LOW/MEDIUM).",
            )

        if excluded_findings:
            return (
                VerdictStatus.NEEDS_REVISION,
                "Подтверждённых замечаний нет, но часть находок модели не "
                "подтверждена дословными цитатами и исключена из отчёта "
                f"(находок: {excluded_findings}). Перед началом разработки "
                "проверьте документ вручную.",
            )

        return (
            VerdictStatus.READY,
            "ТЗ непротиворечиво, полно и готово к передаче в разработку и тестирование.",
        )


def _build_set_audit_prompt(
    collection: RequirementCollection,
    flags: Sequence[LintFlag],
    task_prompt: str,
) -> str:
    """Build the varying suffix: requirements, document flags and the task last."""
    requirements = [
        {
            "id": item.id,
            "original_id": item.original_id,
            "text": item.text,
            "category": item.category.value,
        }
        for item in collection.requirements
    ]
    parts = [
        "### EXTRACTED REQUIREMENTS (JSON)",
        json.dumps(requirements, ensure_ascii=False, indent=2),
    ]
    if flags:
        parts.extend(
            [
                "",
                "### PRE-DETECTED DOCUMENT FLAGS (JSON)",
                json.dumps(
                    [
                        {
                            "flag_id": flag.flag_id,
                            "category": flag.category.value,
                            "severity": flag.severity.value,
                            "message": flag.message,
                            "evidence": flag.evidence,
                            "related_req_ids": flag.related_req_ids,
                        }
                        for flag in flags
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
    parts.extend(["", task_prompt.rstrip()])
    return "\n".join(parts)


def _retry_note(correction: str) -> str:
    """Wrap a correction request that is appended to the user prompt."""
    return f"\n\nОШИБКА ВАЛИДАЦИИ: {correction}" if correction else ""


def _correction_request(
    response: LLMSetAuditResponse, aligner: EvidenceAligner
) -> str:
    """Say what in the answer the document does not back; '' when all is backed."""
    missing = aligner.missing(
        quote for finding in response.findings for quote in finding.evidence
    )
    if missing:
        listed = " | ".join(missing)
        return (
            f"Следующие цитаты не найдены дословно в документе: {listed}. "
            "Замени их на точные фрагменты из исходного текста и повтори ответ."
        )

    without_quotes = [
        finding.title for finding in response.findings if not finding.evidence
    ]
    if without_quotes:
        listed = " | ".join(without_quotes)
        return (
            f"У следующих находок нет цитат в поле evidence: {listed}. "
            "Добавь дословные цитаты из исходного текста и повтори ответ."
        )
    return ""


def _keep_supported_findings(
    findings: Sequence[SetFinding],
    collection: RequirementCollection,
    aligner: EvidenceAligner,
) -> tuple[list[SetFinding], int]:
    """Keep only what the document backs.

    Quotes are aligned to the exact source slice; a near-match is accepted and
    canonicalized. A finding without a single verified quote is excluded, and
    requirement ids that the collection does not know are discarded (an
    identifier from the document is mapped to its internal id). Returns the
    supported findings and the excluded count.
    """
    internal_ids = {item.id for item in collection.requirements}
    document_ids = {
        item.original_id: item.id for item in collection.requirements if item.original_id
    }

    supported: list[SetFinding] = []
    excluded = 0
    for finding in findings:
        verified: list[str] = []
        for quote in finding.evidence:
            match = aligner.align(quote)
            if match is None:
                continue
            if match.matched_text not in verified:
                verified.append(match.matched_text)
        if not verified:
            excluded += 1
            continue
        related: list[str] = []
        for raw_id in finding.related_req_ids:
            requirement_id = document_ids.get(raw_id, raw_id)
            if requirement_id in internal_ids and requirement_id not in related:
                related.append(requirement_id)
        supported.append(
            finding.model_copy(update={"evidence": verified, "related_req_ids": related})
        )
    return supported, excluded
