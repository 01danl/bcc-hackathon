"""Small OpenAI-compatible LLM client used by the requirements agent.

The client is intentionally thin: it sends a system prompt and a user
prompt, requests JSON output, parses the response and validates it with
Pydantic. Business prompts live in :mod:`app.prompts`.

The module never logs or embeds the API key in error messages. Callers
receive short, actionable exceptions instead of raw provider payloads.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_TRAILING_FENCE_RE = re.compile(r"\s*```$")
T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base class for all LLM integration errors."""


class LLMConfigurationError(LLMError):
    """The client is missing configuration or the API rejected credentials."""


class LLMRequestError(LLMError):
    """The API request failed (network, timeout, server error, etc.)."""


class LLMRateLimitError(LLMRequestError):
    """The provider rate limit was exceeded."""


class LLMResponseError(LLMError):
    """The provider answered, but the response was empty, malformed or invalid."""


class LLMClient:
    """OpenAI-compatible chat client with JSON/Pydantic response validation."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str | None,
        model: str | None,
        timeout: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMConfigurationError(
                "LLM_API_KEY is not configured. Set it in .env or the environment."
            )
        if not model or not model.strip():
            raise LLMConfigurationError(
                "LLM_MODEL is not configured. Set a model id in .env or the environment."
            )
        if not base_url or not base_url.strip():
            raise LLMConfigurationError(
                "LLM_BASE_URL is not configured. "
                "For OpenRouter use https://openrouter.ai/api/v1."
            )

        self._model = model.strip()
        self._timeout = timeout
        # Provider options such as OpenRouter's reasoning effort are sent as
        # extra body fields on every request made by this client.
        self._extra_body = dict(extra_body) if extra_body else None
        # Reasoning tokens share this budget with the visible answer.
        self._max_tokens = max_tokens
        # The API key is intentionally not stored on the instance. It is
        # passed straight to the SDK client, which owns it.
        self._client = client or OpenAI(
            api_key=api_key.strip(),
            base_url=base_url.strip(),
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def model(self) -> str:
        """Model id used for every request."""
        return self._model

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> T:
        """Send a chat request and validate the JSON answer with ``response_model``.

        Args:
            system_prompt: instructions loaded from ``app/prompts``.
            user_prompt: document fragment or requirement context.
            response_model: Pydantic model describing the expected JSON.

        Returns:
            A validated instance of ``response_model``.

        Raises:
            LLMRateLimitError: provider rate limit.
            LLMConfigurationError: missing or rejected credentials.
            LLMRequestError: network, timeout or provider error.
            LLMResponseError: empty, malformed JSON or schema mismatch.
        """
        content = self._request_content(system_prompt, user_prompt)
        data = parse_json_response(content)
        try:
            return response_model.model_validate(data)
        except ValidationError as exc:
            raise LLMResponseError(
                "LLM response failed schema validation: "
                + _summarize_validation_error(exc)
            ) from exc

    def _request_content(self, system_prompt: str, user_prompt: str) -> str:
        """Call the API and return the text content of the first choice."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        # Claude Haiku 5.5 rejects any temperature other than 1, so no
        # sampling parameter is sent; prompts stay the source of determinism.
        if self._max_tokens is not None:
            request["max_tokens"] = self._max_tokens
        if self._extra_body:
            request["extra_body"] = self._extra_body
        try:
            completion = self._client.chat.completions.create(
                **request,
                response_format={"type": "json_object"},
            )
        except openai.BadRequestError as exc:
            # Some OpenAI-compatible providers do not implement JSON mode.
            # Retry once without response_format, then let the parser and
            # Pydantic validation handle the answer. Provider options stay.
            if not _looks_like_json_mode_error(exc):
                raise _map_api_error(exc) from exc
            try:
                completion = self._client.chat.completions.create(**request)
            except openai.OpenAIError as retry_exc:
                raise _map_api_error(retry_exc) from retry_exc
        except openai.OpenAIError as exc:
            raise _map_api_error(exc) from exc

        return _extract_content(completion, self._timeout)


def build_llm_client(settings: Settings) -> LLMClient:
    """Create a client from application settings.

    Raises:
        LLMConfigurationError: when required settings are missing.
    """
    return LLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        extra_body=settings.request_extra_body,
        max_tokens=settings.llm_max_tokens,
    )


def parse_json_response(raw: str) -> Any:
    """Parse JSON from a model answer, tolerating Markdown fences and prose.

    The function never includes the raw answer in its exception message,
    because the answer can contain the source document.

    Raises:
        LLMResponseError: when no JSON object or array can be decoded.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise LLMResponseError("LLM returned an empty response.")

    text = _JSON_FENCE_RE.sub("", raw.strip())
    text = _TRAILING_FENCE_RE.sub("", text).strip()

    try:
        value = json.loads(text)
        return _unwrap_json_string(value)
    except json.JSONDecodeError:
        pass

    # Some models prepend a short explanation before the JSON object.
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return _unwrap_json_string(value)

    raise LLMResponseError("LLM returned a response that is not valid JSON.")


def _unwrap_json_string(value: Any) -> Any:
    """Handle the occasional double-encoded JSON string."""
    if isinstance(value, str) and value.strip().startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _extract_content(completion: Any, timeout: float) -> str:
    """Pull plain text out of an OpenAI-compatible completion object."""
    choices = getattr(completion, "choices", None)
    if not choices:
        raise LLMResponseError("LLM returned no choices.")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            if text:
                parts.append(str(text))
        content = "".join(parts)

    if not content or not str(content).strip():
        raise LLMResponseError(
            f"LLM returned an empty response (timeout setting: {timeout:g}s)."
        )
    return str(content)


def _looks_like_json_mode_error(exc: openai.BadRequestError) -> bool:
    """Return True when a 400 error is likely caused by JSON mode."""
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("response_format", "json mode", "structured output", "json_object")
    )


def _map_api_error(exc: openai.OpenAIError) -> LLMError:
    """Translate an SDK exception into a short, key-free application error."""
    if isinstance(exc, openai.AuthenticationError):
        return LLMConfigurationError(
            "LLM API rejected the credentials (HTTP 401). Check LLM_API_KEY."
        )
    if isinstance(exc, openai.PermissionDeniedError):
        return LLMConfigurationError(
            "LLM API denied access for this key (HTTP 403). "
            "Check the key permissions and model access."
        )
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError(
            "LLM API rate limit or quota exceeded. Wait and retry, "
            "or use another model/key."
        )
    if isinstance(exc, openai.APITimeoutError):
        return LLMRequestError(
            "LLM API request timed out. Increase LLM_TIMEOUT or retry."
        )
    if isinstance(exc, openai.APIConnectionError):
        return LLMRequestError(
            "Could not connect to the LLM API. Check network access and LLM_BASE_URL."
        )
    if isinstance(exc, openai.APIStatusError):
        return LLMRequestError(
            f"LLM API returned HTTP {exc.status_code}. "
            "Check the model id, request size and provider status."
        )
    return LLMRequestError("LLM API request failed.")


def _summarize_validation_error(exc: ValidationError) -> str:
    """Return field names and messages without echoing document text."""
    parts: list[str] = []
    for error in exc.errors()[:5]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
        parts.append(f"{location}: {error.get('msg', 'invalid value')}")
    if exc.error_count() > len(parts):
        parts.append(f"... and {exc.error_count() - len(parts)} more")
    return "; ".join(parts)
