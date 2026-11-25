"""Unit tests for environment-based configuration."""

from pathlib import Path

import pytest

from app import config

_ENV_NAMES = (
    "AGENT_VERSION",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TIMEOUT",
    "LLM_MAX_RETRIES",
    "LLM_CHUNK_MAX_CHARS",
    "LLM_CHUNK_OVERLAP_CHARS",
)


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def _load_without_dotenv(tmp_path: Path) -> config.Settings:
    return config.load_settings(env_file=tmp_path / "missing.env")


class TestSettings:
    def test_defaults(self, tmp_path: Path) -> None:
        settings = _load_without_dotenv(tmp_path)

        assert settings.llm_api_key is None
        assert settings.llm_model is None
        assert settings.llm_base_url == config.DEFAULT_LLM_BASE_URL
        assert settings.chunk_max_chars == config.DEFAULT_CHUNK_MAX_CHARS
        assert settings.chunk_overlap_chars == config.DEFAULT_CHUNK_OVERLAP_CHARS

    def test_reads_llm_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_API_KEY", "  sk-or-v1-test  ")
        monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
        monkeypatch.setenv("LLM_MODEL", "test/model")
        monkeypatch.setenv("LLM_TIMEOUT", "12.5")
        monkeypatch.setenv("LLM_MAX_RETRIES", "0")
        monkeypatch.setenv("LLM_CHUNK_MAX_CHARS", "800")
        monkeypatch.setenv("LLM_CHUNK_OVERLAP_CHARS", "100")

        settings = _load_without_dotenv(tmp_path)

        assert settings.llm_api_key == "sk-or-v1-test"
        assert settings.llm_model == "test/model"
        assert settings.llm_timeout == 12.5
        assert settings.llm_max_retries == 0
        assert settings.chunk_max_chars == 800
        assert settings.chunk_overlap_chars == 100

    def test_api_key_is_not_in_repr(self) -> None:
        settings = config.Settings(llm_api_key="sk-or-v1-secret-value")

        assert "sk-or-v1-secret-value" not in repr(settings)

    def test_invalid_integer_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_TIMEOUT", "not-a-number")

        with pytest.raises(config.SettingsError, match="LLM_TIMEOUT"):
            _load_without_dotenv(tmp_path)

    def test_overlap_must_be_smaller_than_chunk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_CHUNK_MAX_CHARS", "500")
        monkeypatch.setenv("LLM_CHUNK_OVERLAP_CHARS", "500")

        with pytest.raises(config.SettingsError, match="smaller"):
            _load_without_dotenv(tmp_path)
