"""Cascading, batched LLM analysis of extracted requirements.

The cheap extractor model first triages every requirement and marks only the
ones that deserve a deep reasoning review. Static lint flags are always routed
to the auditor as well. The auditor model then receives a small batch of
related requirements and returns all findings in one JSON object. Each finding
is bound to a concrete ``REQ-ID``; statuses are derived in Python.

The full source document lives in the stable system prefix shared by every
auditor request, so a provider can reuse its prefix cache. Batch data and the
task instruction are appended at the end of the user message.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.llm.client import LLMClient, LLMResponseError
from app.models.analysis import (
    LLMBatchAnalysisResponse,
    LLMBatchFinding,
    RequirementAnalysis,
    RequirementAnalysisCollection,
)
from app.models.lint import LintFlag
from app.models.requirement import Requirement, RequirementCollection
from app.prompts import load_prompt
from app.services.evidence import EvidenceAligner
from app.services.lint_merge import merge_requirement_flags
from app.services.llm_context import build_document_system_prompt
from app.services.requirement_clustering import build_requirement_batches
from app.services.requirement_triage import RequirementTriage, TriageResult

BATCH_SIZE = 20
ProgressCallback = Callable[[int, int, Requirement], None]


@dataclass(frozen=True)
class _BatchAudit:
    """One validated auditor answer."""

    findings: list[LLMBatchFinding]
    rejected_flag_ids: set[str]


class RequirementAnalyzer:
    """Analyze requirements in clustered batches with the auditor model."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        max_response_retries: int = 1,
        batch_size: int = BATCH_SIZE,
        triage: RequirementTriage | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._llm_client = llm_client
        self._max_response_retries = max(0, max_response_retries)
        self._batch_size = batch_size
        self._triage = triage
        self._system_prompt = load_prompt("auditor_system")
        self._task_prompt = load_prompt("analyze_batch")

    def analyze_all(
        self,
        collection: RequirementCollection,
        *,
        document_text: str | None = None,
        pre_detected_flags: Sequence[LintFlag] = (),
        progress: ProgressCallback | None = None,
    ) -> RequirementAnalysisCollection:
        """Audit requirements in related batches, skipping triaged clean ones."""
        corpus = document_text or "\n".join(
            item.text for item in collection.requirements
        )
        requirements = collection.requirements
        flags = list(pre_detected_flags)
        flags_by_requirement: dict[str, list[LintFlag]] = defaultdict(list)
        for flag in flags:
            if flag.requirement_id:
                flags_by_requirement[flag.requirement_id].append(flag)

        triage_result: TriageResult | None = None
        if self._triage is not None:
            triage_result = self._triage.triage(
                collection,
                document_text=corpus,
                pre_detected_flags=flags,
            )
            suspicious_ids = set(triage_result.suspicious_ids)
        else:
            suspicious_ids = {item.id for item in requirements}
        # A deterministic flag must never be skipped, even if triage missed it.
        suspicious_ids.update(flags_by_requirement)

        to_audit = [
            item for item in requirements if item.id in suspicious_ids
        ]
        system_prompt = build_document_system_prompt(
            self._system_prompt, corpus
        )
        aligner = EvidenceAligner(corpus)
        findings_by_requirement: dict[str, list[LLMBatchFinding]] = defaultdict(list)
        rejected_flag_ids: set[str] = set()

        audited = 0
        for batch in build_requirement_batches(to_audit, self._batch_size):
            batch_flags = [
                flag
                for requirement in batch
                for flag in flags_by_requirement.get(requirement.id, [])
            ]
            result = self._analyze_batch(
                batch,
                aligner=aligner,
                system_prompt=system_prompt,
                flags=batch_flags,
            )
            for finding in result.findings:
                findings_by_requirement[finding.requirement_id].append(finding)
            rejected_flag_ids.update(result.rejected_flag_ids)
            for requirement in batch:
                audited += 1
                if progress is not None:
                    progress(audited, len(to_audit), requirement)

        analyses: list[RequirementAnalysis] = []
        static_findings = 0
        for requirement in requirements:
            merged, added = merge_requirement_flags(
                findings_by_requirement.get(requirement.id, []),
                flags_by_requirement.get(requirement.id, []),
                rejected_flag_ids,
            )
            static_findings += added
            analyses.append(
                RequirementAnalysis.from_findings(requirement.id, merged)
            )

        return RequirementAnalysisCollection(
            analyses=analyses,
            skipped_requirements=len(requirements) - len(to_audit),
            deep_audited_requirements=len(to_audit),
            static_findings=static_findings,
            triage_summary=triage_result.summary if triage_result else "",
            triage_fallback_used=(
                triage_result.fallback_used if triage_result else False
            ),
        )

    def analyze(
        self,
        requirement: Requirement,
        collection: RequirementCollection,
        *,
        document_text: str | None = None,
    ) -> RequirementAnalysis:
        """Backward-compatible wrapper for analyzing a single requirement."""
        single = RequirementCollection(requirements=[requirement])
        return self.analyze_all(
            single,
            document_text=document_text
            or "\n".join(item.text for item in collection.requirements),
        ).analyses[0]

    def _analyze_batch(
        self,
        batch: Sequence[Requirement],
        *,
        aligner: EvidenceAligner,
        system_prompt: str,
        flags: Sequence[LintFlag],
    ) -> _BatchAudit:
        """Request and validate the findings for one batch."""
        base_prompt = _build_batch_prompt(batch, flags, self._task_prompt)
        known_ids = {item.id for item in batch}
        known_flag_ids = {flag.flag_id for flag in flags}
        correction = ""
        last_error: LLMResponseError | None = None

        for _ in range(self._max_response_retries + 1):
            try:
                response = self._llm_client.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=base_prompt + correction,
                    response_model=LLMBatchAnalysisResponse,
                )
                return _validate_batch_response(
                    response,
                    aligner=aligner,
                    known_ids=known_ids,
                    known_flag_ids=known_flag_ids,
                )
            except LLMResponseError as exc:
                last_error = exc
                correction = (
                    "\n\nVALIDATION ERROR: the previous answer did not pass "
                    "validation. Return one JSON object, copy evidence verbatim "
                    "from the source document and use only known IDs from the "
                    "batch."
                )

        assert last_error is not None
        raise last_error


def _build_batch_prompt(
    batch: Sequence[Requirement],
    flags: Sequence[LintFlag],
    task_prompt: str,
) -> str:
    """Build the varying suffix: batch data, flags and the task last."""
    requirements = [
        {
            "id": item.id,
            "original_id": item.original_id,
            "text": item.text,
            "category": item.category.value,
            "location": item.location.model_dump(mode="json"),
        }
        for item in batch
    ]
    parts = [
        "### BATCH REQUIREMENTS (JSON)",
        json.dumps(requirements, ensure_ascii=False, indent=2),
    ]
    if flags:
        parts.extend(
            [
                "",
                "### PRE-DETECTED FLAGS (JSON)",
                json.dumps(
                    [
                        {
                            "flag_id": flag.flag_id,
                            "requirement_id": flag.requirement_id,
                            "category": flag.category.value,
                            "severity": flag.severity.value,
                            "message": flag.message,
                            "evidence": flag.evidence,
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


def _validate_batch_response(
    response: LLMBatchAnalysisResponse,
    *,
    aligner: EvidenceAligner,
    known_ids: set[str],
    known_flag_ids: set[str],
) -> _BatchAudit:
    """Align every quote with the source and reject unknown ids."""
    aligned: list[LLMBatchFinding] = []
    missing: list[str] = []
    unknown: list[str] = []

    for finding in response.findings:
        if finding.requirement_id not in known_ids:
            unknown.append(finding.requirement_id)
            continue
        match = aligner.align(finding.evidence)
        if match is None:
            missing.append(finding.evidence)
            continue
        aligned.append(
            finding.model_copy(update={"evidence": match.matched_text})
        )

    problems: list[str] = []
    if missing:
        problems.append(
            "evidence is not an exact fragment of the source document: "
            + " | ".join(missing)
        )
    if unknown:
        problems.append(
            "findings reference unknown requirement ids: "
            + ", ".join(sorted(set(unknown)))
        )
    if problems:
        raise LLMResponseError("; ".join(problems))

    return _BatchAudit(
        findings=aligned,
        rejected_flag_ids={
            flag_id
            for flag_id in response.rejected_flag_ids
            if flag_id in known_flag_ids
        },
    )
