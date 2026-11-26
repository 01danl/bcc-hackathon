"""Cheap triage that routes requirements to the expensive auditor.

Most requirements in a real specification are fine. Sending every one of
them through a reasoning model at ``effort=max`` burns budget without
improving the result. The extractor model performs a fast first pass and
marks only requirements with a concrete, plausible defect. Static lint flags
are always routed to the auditor as well.

Triage is a routing hint, not a verdict. A failed triage answer falls back to
auditing the whole batch, so the pipeline degrades safely instead of silently
skipping requirements.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.llm.client import LLMClient, LLMError, LLMResponseError
from app.models.lint import LintFlag
from app.models.requirement import Requirement, RequirementCollection
from app.models.triage import LLMTriageResponse
from app.prompts import load_prompt
from app.services.llm_context import build_document_system_prompt
from app.services.requirement_clustering import build_requirement_batches

BATCH_SIZE = 20


@dataclass(frozen=True)
class TriageResult:
    """Which requirements should receive a deep auditor review."""

    suspicious_ids: frozenset[str]
    reasons: dict[str, str] = field(default_factory=dict)
    summary: str = ""
    fallback_used: bool = False
    triaged_requirements: int = 0

    @classmethod
    def all_suspicious(
        cls, collection: RequirementCollection, *, reason: str
    ) -> TriageResult:
        """Route every requirement to the auditor (safe fallback)."""
        ids = frozenset(item.id for item in collection.requirements)
        return cls(
            suspicious_ids=ids,
            reasons={requirement_id: reason for requirement_id in ids},
            summary=reason,
            fallback_used=True,
            triaged_requirements=len(collection),
        )


class RequirementTriage:
    """Fast first pass over every requirement with the extractor model."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        batch_size: int = BATCH_SIZE,
        max_response_retries: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._llm_client = llm_client
        self._batch_size = batch_size
        self._max_response_retries = max(0, max_response_retries)
        self._system_prompt = load_prompt("triage_requirements")

    def triage(
        self,
        collection: RequirementCollection,
        *,
        document_text: str | None = None,
        pre_detected_flags: Sequence[LintFlag] = (),
    ) -> TriageResult:
        """Return the ids that deserve a deep auditor review."""
        requirements = collection.requirements
        if not requirements:
            return TriageResult(suspicious_ids=frozenset(), triaged_requirements=0)

        corpus = document_text or "\n".join(item.text for item in requirements)
        system_prompt = build_document_system_prompt(self._system_prompt, corpus)
        batches = build_requirement_batches(requirements, self._batch_size)

        suspicious: set[str] = set()
        reasons: dict[str, str] = {}
        summaries: list[str] = []
        fallback_used = False

        for batch in batches:
            known_ids = {item.id for item in batch}
            flags = [
                flag
                for flag in pre_detected_flags
                if flag.requirement_id in known_ids
            ]
            try:
                response = self._triage_batch(
                    system_prompt, _build_triage_prompt(batch, flags), known_ids
                )
            except LLMError:
                # Safe degradation: a failed or malformed triage answer must
                # never silently turn into "everything is fine".
                fallback_used = True
                suspicious.update(known_ids)
                for requirement_id in known_ids:
                    reasons[requirement_id] = (
                        "Triage не смог разобрать ответ; требование отправлено "
                        "на глубокую проверку."
                    )
                summaries.append(
                    "Triage недоступен или вернул некорректный ответ; "
                    "пакет проверен целиком."
                )
                continue

            if response.summary:
                summaries.append(response.summary)
            for item in response.suspicious:
                if item.requirement_id not in known_ids:
                    continue
                suspicious.add(item.requirement_id)
                reasons[item.requirement_id] = item.reason

        # Linter flags are deterministic and always go to the deep auditor.
        for flag in pre_detected_flags:
            if not flag.requirement_id:
                continue
            suspicious.add(flag.requirement_id)
            reasons.setdefault(
                flag.requirement_id, f"Линтер: {flag.message}"
            )

        valid_ids = {item.id for item in requirements}
        suspicious &= valid_ids
        summary = " ".join(dict.fromkeys(item for item in summaries if item))
        if fallback_used:
            prefix = "Triage деградировал до полной проверки."
            summary = f"{prefix} {summary}".strip()

        return TriageResult(
            suspicious_ids=frozenset(suspicious),
            reasons={key: value for key, value in reasons.items() if key in valid_ids},
            summary=summary,
            fallback_used=fallback_used,
            triaged_requirements=len(requirements),
        )

    def _triage_batch(
        self,
        system_prompt: str,
        user_prompt: str,
        known_ids: set[str],
    ) -> LLMTriageResponse:
        """Request one batch, retrying only a malformed or unknown-id answer."""
        correction = ""
        last_error: LLMResponseError | None = None

        for _ in range(self._max_response_retries + 1):
            try:
                response = self._llm_client.complete_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt + correction,
                    response_model=LLMTriageResponse,
                )
            except LLMResponseError as exc:
                last_error = exc
                correction = (
                    "\n\nОШИБКА ВАЛИДАЦИИ: предыдущий ответ не прошёл проверку. "
                    "Верни JSON строго по схеме и используй только ID из пакета."
                )
                continue

            known = [
                item for item in response.suspicious if item.requirement_id in known_ids
            ]
            if response.suspicious and not known:
                last_error = LLMResponseError(
                    "triage response references unknown requirement ids"
                )
                correction = (
                    "\n\nОШИБКА ВАЛИДАЦИИ: в ответе нет ни одного ID из пакета. "
                    "Используй только переданные REQ-ID."
                )
                continue
            return response.model_copy(update={"suspicious": known})

        assert last_error is not None
        raise last_error


def _build_triage_prompt(
    batch: Sequence[Requirement], flags: Sequence[LintFlag]
) -> str:
    """Build the varying part of the triage request."""
    requirements = [
        {
            "id": item.id,
            "original_id": item.original_id,
            "text": item.text,
            "category": item.category.value,
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
    parts.extend(
        [
            "",
            "### TASK",
            "Check the batch and return a JSON object that follows the schema "
            "from the system message.",
        ]
    )
    return "\n".join(parts)
