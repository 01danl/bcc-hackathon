"""Application configuration.

Settings are read from environment variables; a local ``.env`` file is
loaded first when present. LLM settings are required for the real
extraction/analysis pipeline and are validated when the client is built.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_AGENT_VERSION = "0.2.0"
DEFAULT_LLM_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_TIMEOUT = 60.0
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_CHUNK_MAX_CHARS = 6000
DEFAULT_CHUNK_OVERLAP_CHARS = 600


class SettingsError(ValueError):
    """Raised when an environment setting has an invalid value."""


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for a single pipeline run.

    ``llm_api_key`` is excluded from the dataclass ``repr`` so that a
    settings object can be logged safely.
    """

    agent_name: str = "AI Requirements Reviewer"
    agent_version: str = DEFAULT_AGENT_VERSION
    llm_api_key: str | None = field(default=None, repr=False)
    llm_base_url: str = DEFAULT_LLM_BASE_URL
    llm_model: str | None = None
    llm_timeout: float = DEFAULT_LLM_TIMEOUT
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Build settings from the environment, loading ``.env`` when available.

    Args:
        env_file: optional explicit path to a dotenv file; by default the
            ``.env`` file in the current directory is used if present.

    Returns:
        Freshly built settings; no global state is kept.

    Raises:
        SettingsError: if a numeric setting is malformed or inconsistent.
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

    return Settings(
        agent_version=_read_text("AGENT_VERSION") or DEFAULT_AGENT_VERSION,
        llm_api_key=_read_text("LLM_API_KEY"),
        llm_base_url=_read_text("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL,
        llm_model=_read_text("LLM_MODEL"),
        llm_timeout=timeout,
        llm_max_retries=retries,
        chunk_max_chars=max_chars,
        chunk_overlap_chars=overlap,
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
