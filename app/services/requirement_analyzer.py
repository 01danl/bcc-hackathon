"""LLM-based quality analysis of individual requirements.

Each requirement is sent to the model together with the full requirement
set (needed for consistency checks). The response is validated with
Pydantic and every finding must quote an exact fragment from the source,
otherwise the analysis is rejected and retried once.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from app.llm.client import LLMClient, LLMResponseError
from app.models.analysis import (
    LLMAnalysisResponse,
    RequirementAnalysis,
    RequirementAnalysisCollection,
)
from app.models.requirement import Requirement, RequirementCollection
from app.prompts import load_prompt

_WHITESPACE_RE = re.compile(r"\s+")
_MARKDOWN_MARKERS_RE = re.compile(r"[*_`]+")

ProgressCallback = Callable[[int, int, Requirement], None]


class RequirementAnalyzer:
    """Analyze requirements against ISO/IEC/IEEE 29148 quality criteria."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        max_response_retries: int = 1,
    ) -> None:
        self._llm_client = llm_client
        self._max_response_retries = max(0, max_response_retries)
        self._system_prompt = load_prompt("analyze_requirement")

    def analyze(
        self,
        requirement: Requirement,
        collection: RequirementCollection,
        *,
        document_text: str | None = None,
    ) -> RequirementAnalysis:
        """Analyze one requirement.

        Raises:
            LLMResponseError: if the model answer stays malformed or its
                evidence cannot be found in the source.
            LLMRequestError: on network/provider failures.
        """
        corpus = document_text or "\n".join(
            item.text for item in collection.requirements
        )
        base_prompt = _build_analysis_prompt(requirement, collection)
        last_error: LLMResponseError | None = None

        for attempt in range(self._max_response_retries + 1):
            user_prompt = base_prompt
            if attempt > 0:
                user_prompt += (
                    "\n\nIMPORTANT: the previous answer failed validation. "
                    "Copy the evidence EXACTLY from the provided requirement "
                    "text and return only the requested JSON object."
                )
            try:
                response = self._llm_client.complete_json(
                    system_prompt=self._system_prompt,
                    user_prompt=user_prompt,
                    response_model=LLMAnalysisResponse,
                )
                _ensure_evidence_is_verbatim(response, corpus, requirement.id)
                return RequirementAnalysis.from_findings(
                    requirement.id, response.findings
                )
            except LLMResponseError as exc:
                last_error = exc

        assert last_error is not None
        raise last_error

    def analyze_all(
        self,
        collection: RequirementCollection,
        *,
        document_text: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> RequirementAnalysisCollection:
        """Analyze every requirement and return all structured results."""
        analyses: list[RequirementAnalysis] = []
        total = len(collection)
        for number, requirement in enumerate(collection.requirements, start=1):
            if progress is not None:
                progress(number, total, requirement)
            analyses.append(
                self.analyze(
                    requirement,
                    collection,
                    document_text=document_text,
                )
            )
        return RequirementAnalysisCollection(analyses=analyses)


def _build_analysis_prompt(
    requirement: Requirement, collection: RequirementCollection
) -> str:
    """Build the user message for one requirement analysis request."""
    target = {
        "id": requirement.id,
        "original_id": requirement.original_id,
        "text": requirement.text,
        "category": requirement.category.value,
        "location": requirement.location.model_dump(mode="json"),
    }
    full_set = [
        {
            "id": item.id,
            "original_id": item.original_id,
            "text": item.text,
            "category": item.category.value,
        }
        for item in collection.requirements
    ]
    return (
        "TARGET REQUIREMENT\n"
        + json.dumps(target, ensure_ascii=False, indent=2)
        + "\n\nALL REQUIREMENTS (for consistency checks)\n"
        + json.dumps(full_set, ensure_ascii=False, indent=2)
        + "\n\nEvaluate only the target requirement and return the JSON object."
    )


def _ensure_evidence_is_verbatim(
    response: LLMAnalysisResponse,
    document_text: str,
    requirement_id: str,
) -> None:
    """Reject findings whose evidence is not present in the source text."""
    haystack = _normalize_for_matching(document_text)
    for finding in response.findings:
        needle = _normalize_for_matching(finding.evidence)
        if not needle:
            raise LLMResponseError(
                f"{requirement_id}: {finding.criterion.value} finding has empty evidence."
            )
        if needle not in haystack:
            raise LLMResponseError(
                f"{requirement_id}: evidence for the {finding.criterion.value} "
                "finding is not an exact fragment of the source document."
            )


def _normalize_for_matching(text: str) -> str:
    """Normalize case, whitespace and Markdown markers for evidence checks."""
    without_markers = _MARKDOWN_MARKERS_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", without_markers).strip().casefold()
