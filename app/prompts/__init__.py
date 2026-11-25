"""Loading of prompt templates kept outside the Python business logic."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


class PromptNotFoundError(FileNotFoundError):
    """A requested prompt template does not exist."""


def load_prompt(name: str) -> str:
    """Load a prompt template by file stem, e.g. ``extract_requirements``.

    Raises:
        PromptNotFoundError: when the template is missing or unreadable.
    """
    if not name or not all(character.isalnum() or character in "._" for character in name):
        raise ValueError(f"Invalid prompt name: {name!r}")
    path = PROMPTS_DIR / f"{name}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptNotFoundError(f"Prompt template not found: {path.name}") from exc
