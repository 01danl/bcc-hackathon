"""Application configuration.

Settings are read from environment variables; a local ``.env`` file is
loaded first when present. LLM-related variables are reserved for later
milestones and are not used yet.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_AGENT_VERSION = "0.1.0"


@dataclass(frozen=True)
class Settings:
    """Immutable runtime settings for a single pipeline run."""

    agent_name: str = "AI Requirements Reviewer"
    agent_version: str = DEFAULT_AGENT_VERSION
    llm_api_key: str | None = None
    llm_model: str | None = None


def load_settings(env_file: str | Path | None = None) -> Settings:
    """Build settings from the environment, loading ``.env`` when available.

    Args:
        env_file: optional explicit path to a dotenv file; by default the
            ``.env`` file in the current directory is used if present.

    Returns:
        Freshly built settings; no global state is kept.
    """
    load_dotenv(dotenv_path=env_file)
    return Settings(
        agent_version=os.getenv("AGENT_VERSION", DEFAULT_AGENT_VERSION),
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_model=os.getenv("LLM_MODEL") or None,
    )
