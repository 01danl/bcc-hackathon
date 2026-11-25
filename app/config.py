"""Application configuration.

Settings are read from environment variables; a local ``.env`` file is
loaded first when present. LLM settings are required for the real
extraction/analysis pipeline and are validated when the client is built.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_AGENT_VERSION = "0.2.0"
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_TIMEOUT = 60.0
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_CHUNK_MAX_CHARS = 6000
DEFAULT_CHUNK_OVERLAP_CHARS = 600
DEFAULT_EXTRACTOR_EFFORT = "low"
DEFAULT_AUDITOR_EFFORT = "max"
DEFAULT_EXTRACTOR_MAX_TOKENS = 4000
DEFAULT_AUDITOR_MAX_TOKENS = 16000

# OpenRouter normalizes reasoning across providers. The two stages use
# different models, and their allowed effort levels are not identical.
_ALL_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
_MODEL_EFFORTS: dict[str, frozenset[str]] = {
    # Claude Haiku 5.5 has no ``minimal``; ``none`` disables thinking.
    "anthropic/claude-haiku-5.5": frozenset(
        {"none", "low", "medium", "high", "xhigh", "max"}
    ),
    # Muse Ember always thinks; ``none`` returns HTTP 400.
    "meta/muse-ember": frozenset(
        {"minimal", "low", "medium", "high", "xhigh", "max"}
    ),
}


class SettingsError(ValueError):
    """Raised when an environment setting has an invalid value."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for a single pipeline run.

    Each pipeline stage uses its own model: ``extractor_model`` extracts
    the requirements and ``auditor_model`` checks each requirement and the
    whole set. A stage without a model of its own falls back to
    ``llm_model``, the single-model setting.

    ``llm_api_key`` is excluded from the dataclass ``repr`` so that a
    settings object can be logged safely.
    """

    agent_name: str = "AI Requirements Reviewer"
    agent_version: str = DEFAULT_AGENT_VERSION
    llm_api_key: str | None = field(default=None, repr=False)
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str | None = None
    extractor_model: str | None = None
    auditor_model: str | None = None
    extractor_effort: str | None = None
    auditor_effort: str | None = None
    extractor_max_tokens: int = DEFAULT_EXTRACTOR_MAX_TOKENS
    auditor_max_tokens: int = DEFAULT_AUDITOR_MAX_TOKENS
    llm_timeout: float = DEFAULT_LLM_TIMEOUT
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS
    llm_max_tokens: int | None = field(default=None, repr=False)
    request_extra_body: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Fall back to ``llm_model`` for stages without their own model."""
        if not self.extractor_model:
            object.__setattr__(self, "extractor_model", self.llm_model)
        if not self.auditor_model:
            object.__setattr__(self, "auditor_model", self.llm_model)

    def with_model(
        self,
        model: str | None,
        *,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Settings:
        """Return a copy pinned to ``model`` for a stage's LLM client.

        ``extra_body`` carries OpenRouter reasoning options and ``max_tokens``
        leaves room for thinking tokens. Only the copy changes, so one
        settings object can pin different options for the two stages.
        """
        return replace(
            self,
            llm_model=model,
            request_extra_body=extra_body,
            llm_max_tokens=max_tokens,
        )


def reasoning_extra_body(effort: str | None) -> dict[str, Any] | None:
    """Build the unified OpenRouter reasoning body for a stage.

    The ``openai`` SDK has no top-level ``reasoning`` field, so the option is
    sent through ``extra_body``. ``exclude`` hides the thinking trace from
    the message content; the tokens are still billed as output tokens.
    """
    if not effort:
        return None
    return {"reasoning": {"effort": effort, "exclude": True}}


def allowed_efforts(model: str | None) -> frozenset[str]:
    """Return the effort levels accepted by ``model`` on OpenRouter."""
    if not model:
        return _ALL_EFFORTS
    return _MODEL_EFFORTS.get(model, _ALL_EFFORTS)


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Build settings from the environment, loading ``.env`` when available.

    Args:
        env_file: optional explicit path to a dotenv file; by default the
            ``.env`` file in the current directory is used if present.

    Returns:
        Freshly built settings; no global state is kept.

    Raises:
        SettingsError: if a numeric setting is malformed or an effort level
            is not supported by the stage's model.
    """
    load_dotenv(dotenv_path=env_file)

    max_chars = _read_int("LLM_CHUNK_MAX_CHARS", DEFAULT_CHUNK_MAX_CHARS, minimum=200)
    overlap = _read_int("LLM_CHUNK_OVERLAP_CHARS", DEFAULT_CHUNK_OVERLAP_CHARS, minimum=0)
    if overlap >= max_chars:
        raise SettingsError(
            "LLM_CHUNK_OVERLAP_CHARS must be smaller than LLM_CHUNK_MAX_CHARS"
        )

    timeout = _read_float("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT, minimum=1.0)
    retries = _read_int("LLM_MAX_RETRIES", DEFAULT_LLM_MAX_RETRIES, minimum=0)
    extractor_effort = _read_effort("EXTRACTOR_EFFORT", DEFAULT_EXTRACTOR_EFFORT)
    auditor_effort = _read_effort("AUDITOR_EFFORT", DEFAULT_AUDITOR_EFFORT)
    extractor_max_tokens = _read_int(
        "EXTRACTOR_MAX_TOKENS", DEFAULT_EXTRACTOR_MAX_TOKENS, minimum=1000
    )
    auditor_max_tokens = _read_int(
        "AUDITOR_MAX_TOKENS", DEFAULT_AUDITOR_MAX_TOKENS, minimum=1000
    )

    settings = Settings(
        agent_version=_read_text("AGENT_VERSION") or DEFAULT_AGENT_VERSION,
        llm_api_key=_read_text("LLM_API_KEY"),
        llm_base_url=_read_text("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL,
        llm_model=_read_text("LLM_MODEL"),
        extractor_model=_read_text("EXTRACTOR_MODEL"),
        auditor_model=_read_text("AUDITOR_MODEL"),
        extractor_effort=extractor_effort,
        auditor_effort=auditor_effort,
        extractor_max_tokens=extractor_max_tokens,
        auditor_max_tokens=auditor_max_tokens,
        llm_timeout=timeout,
        llm_max_retries=retries,
        chunk_max_chars=max_chars,
        chunk_overlap_chars=overlap,
    )
    _validate_stage_effort("extractor", settings.extractor_model, extractor_effort)
    _validate_stage_effort("auditor", settings.auditor_model, auditor_effort)
    return settings


def _validate_stage_effort(stage: str, model: str | None, effort: str | None) -> None:
    """Reject an effort level that the stage's model does not accept."""
    if not model or not effort:
        return
    accepted = allowed_efforts(model)
    if effort not in accepted:
        raise SettingsError(
            f"{stage} model {model!r} does not support effort={effort!r}. "
            f"Allowed: {', '.join(sorted(accepted))}"
        )


def _read_text(name: str) -> str | None:
    """Return a stripped environment value, or ``None`` when empty."""
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _read_int(name: str, default: int, *, minimum: int) -> int:
    raw = _read_text(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise SettingsError(f"{name} must be >= {minimum}, got {value}")
    return value


def _read_float(name: str, default: float, *, minimum: float) -> float:
    raw = _read_text(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise SettingsError(f"{name} must be >= {minimum}, got {value}")
    return value


def _read_effort(name: str, default: str) -> str:
    """Read an effort level and reject values unknown to OpenRouter."""
    raw = _read_text(name)
    if raw is None:
        return default
    value = raw.lower()
    if value not in _ALL_EFFORTS:
        allowed = ", ".join(sorted(_ALL_EFFORTS))
        raise SettingsError(f"{name} must be one of: {allowed}; got {raw!r}")
    return value
