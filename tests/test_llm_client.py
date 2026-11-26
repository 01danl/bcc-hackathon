"""Unit tests for the OpenAI-compatible LLM client (no real API calls)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from app.llm.client import (
    LLMClient,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMRequestError,
    LLMResponseError,
    parse_json_response,
)
from app.models.analysis import LLMExtractionResponse


def _completion(content: str | None) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class _FakeCompletions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeOpenAI:
    def __init__(self, outcomes: list[Any]) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(outcomes))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chat.completions.calls


def _client(outcomes: list[Any]) -> tuple[LLMClient, _FakeOpenAI]:
    fake = _FakeOpenAI(outcomes)
    client = LLMClient(
        api_key="sk-or-v1-test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        client=fake,
    )
    return client, fake


def _http_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://example.test/chat/completions")
    return httpx.Response(status_code, request=request)


def _api_error(
    error_class: type[openai.APIStatusError], message: str, status_code: int
) -> openai.APIStatusError:
    return error_class(message, response=_http_response(status_code), body=None)


class TestConfiguration:
    @pytest.mark.parametrize(
        ("api_key", "base_url", "model", "expected"),
        [
            (None, "https://example.test", "test/model", "LLM_API_KEY"),
            ("key", None, "test/model", "LLM_BASE_URL"),
            ("key", "https://example.test", None, "LLM_MODEL"),
        ],
    )
    def test_missing_settings_rejected(
        self,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        expected: str,
    ) -> None:
        with pytest.raises(LLMConfigurationError, match=expected):
            LLMClient(api_key=api_key, base_url=base_url, model=model)

    def test_key_not_leaked_in_error(self) -> None:
        with pytest.raises(LLMConfigurationError) as exc_info:
            LLMClient(
                api_key="sk-or-v1-super-secret",
                base_url="https://example.test",
                model=None,
            )

        assert "sk-or-v1-super-secret" not in str(exc_info.value)


class TestCompleteJson:
    def test_valid_json_is_validated(self) -> None:
        payload = {
            "requirements": [
                {
                    "original_id": "Ф-1",
                    "text": "Система должна отправлять push.",
                    "category": "non-functional",
                    "source_location": {"section": "Функциональные требования"},
                }
            ]
        }
        client, fake = _client([_completion(json.dumps(payload, ensure_ascii=False))])

        result = client.complete_json(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractionResponse,
        )

        assert len(result.requirements) == 1
        assert result.requirements[0].text == "Система должна отправлять push."
        assert result.requirements[0].category.value == "non_functional"
        assert fake.calls[0]["response_format"] == {"type": "json_object"}
        assert fake.calls[0]["model"] == "test/model"

    def test_markdown_fenced_json_is_accepted(self) -> None:
        content = '```json\n{"findings": []}\n```'
        client, _ = _client([_completion(content)])

        result = client.complete_json(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractionResponse,
        )

        assert result.requirements == []

    def test_malformed_json_raises(self) -> None:
        client, _ = _client([_completion("not json at all")])

        with pytest.raises(LLMResponseError, match="not valid JSON"):
            client.complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=LLMExtractionResponse,
            )

    def test_schema_mismatch_raises(self) -> None:
        client, _ = _client([_completion('{"requirements": [{"text": ""}]}')])

        with pytest.raises(LLMResponseError, match="schema validation"):
            client.complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=LLMExtractionResponse,
            )

    def test_empty_answer_raises(self) -> None:
        client, _ = _client([_completion(None)])

        with pytest.raises(LLMResponseError, match="empty"):
            client.complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=LLMExtractionResponse,
            )

    def test_json_mode_is_retried_without_response_format(self) -> None:
        payload = {"requirements": []}
        unsupported = _api_error(
            openai.BadRequestError,
            "response_format json_object is not supported",
            400,
        )
        client, fake = _client(
            [unsupported, _completion(json.dumps(payload))]
        )

        result = client.complete_json(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractionResponse,
        )

        assert result.requirements == []
        assert len(fake.calls) == 2
        assert "response_format" in fake.calls[0]
        assert "response_format" not in fake.calls[1]

    def test_other_bad_request_is_not_retried(self) -> None:
        bad_request = _api_error(
            openai.BadRequestError,
            "context length exceeded",
            400,
        )
        client, fake = _client([bad_request])

        with pytest.raises(LLMRequestError, match="HTTP 400"):
            client.complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=LLMExtractionResponse,
            )

        assert len(fake.calls) == 1

    def test_extra_body_is_forwarded(self) -> None:
        payload = {"requirements": []}
        fake = _FakeOpenAI([_completion(json.dumps(payload))])
        client = LLMClient(
            api_key="sk-or-v1-test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            client=fake,
            extra_body={"reasoning": {"effort": "low", "exclude": True}},
        )

        result = client.complete_json(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractionResponse,
        )

        assert result.requirements == []
        assert fake.calls[0]["extra_body"] == {
            "reasoning": {"effort": "low", "exclude": True}
        }
        assert fake.calls[0]["model"] == "test/model"
        assert "temperature" not in fake.calls[0]

    def test_no_token_limit_is_sent(self) -> None:
        payload = {"requirements": []}
        fake = _FakeOpenAI([_completion(json.dumps(payload))])
        client = LLMClient(
            api_key="sk-or-v1-test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            client=fake,
        )

        client.complete_json(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractionResponse,
        )

        assert "max_tokens" not in fake.calls[0]
        assert "temperature" not in fake.calls[0]

    def test_extra_body_survives_json_mode_retry(self) -> None:
        payload = {"requirements": []}
        unsupported = _api_error(
            openai.BadRequestError,
            "response_format json_object is not supported",
            400,
        )
        fake = _FakeOpenAI([unsupported, _completion(json.dumps(payload))])
        client = LLMClient(
            api_key="sk-or-v1-test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            client=fake,
            extra_body={"reasoning": {"effort": "max", "exclude": True}},
        )

        result = client.complete_json(
            system_prompt="system",
            user_prompt="user",
            response_model=LLMExtractionResponse,
        )

        assert result.requirements == []
        assert len(fake.calls) == 2
        assert all(
            call["extra_body"] == {"reasoning": {"effort": "max", "exclude": True}}
            for call in fake.calls
        )
        assert "response_format" not in fake.calls[1]


class TestApiErrors:
    def test_rate_limit(self) -> None:
        error = _api_error(
            openai.RateLimitError, "rate limited for sk-or-v1-test-key", 429
        )
        client, _ = _client([error])

        with pytest.raises(LLMRateLimitError) as exc_info:
            client.complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=LLMExtractionResponse,
            )

        assert "sk-or-v1-test-key" not in str(exc_info.value)

    def test_authentication(self) -> None:
        error = _api_error(openai.AuthenticationError, "invalid key", 401)
        client, _ = _client([error])

        with pytest.raises(LLMConfigurationError, match="401"):
            client.complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=LLMExtractionResponse,
            )

    def test_timeout(self) -> None:
        request = httpx.Request("POST", "https://example.test/chat/completions")
        client, _ = _client([openai.APITimeoutError(request=request)])

        with pytest.raises(LLMRequestError, match="timed out"):
            client.complete_json(
                system_prompt="system",
                user_prompt="user",
                response_model=LLMExtractionResponse,
            )


class TestParseJsonResponse:
    def test_prose_before_json_is_tolerated(self) -> None:
        parsed = parse_json_response('Here is the answer:\n{"requirements": []}')

        assert parsed == {"requirements": []}

    def test_bare_array_is_tolerated(self) -> None:
        parsed = parse_json_response('[{"text": "x"}]')

        assert parsed == [{"text": "x"}]

    def test_double_encoded_json_is_unwrapped(self) -> None:
        parsed = parse_json_response('"{\\"requirements\\": []}"')

        assert parsed == {"requirements": []}
