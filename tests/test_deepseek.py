"""Tests for the DeepSeek OpenAI-compatible provider."""

from collections.abc import AsyncIterator
from io import StringIO
from types import SimpleNamespace
from typing import Any, cast

import httpx
import openai
import pytest

from codewright.config import ConfigError, ProviderConfig
from codewright.llm import (
    LLMAuthenticationError,
    LLMError,
    LLMModelNotFoundError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    Message,
    MessageRole,
    Provider,
)
from codewright.llm.deepseek import DeepSeekProvider
from codewright.llm.factory import create_provider
from codewright.utils.logging import configure_logging

SYNTHETIC_SECRET = "test-key-not-a-real-secret"


class FakeStream:
    """Asynchronous stream with optional terminal failure."""

    def __init__(self, chunks: list[object], error: Exception | None = None) -> None:
        self._chunks = iter(chunks)
        self._error = error

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._chunks)
        except StopIteration:
            if self._error is not None:
                error, self._error = self._error, None
                raise error from None
            raise StopAsyncIteration from None


class FakeCompletions:
    """Record SDK requests and return or raise a configured result."""

    def __init__(self, result: object = None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.requests: list[dict[str, object]] = []

    async def create(self, **parameters: object) -> object:
        self.requests.append(parameters)
        if self.error is not None:
            raise self.error
        return self.result


class FakeClient:
    """Minimal async client matching the provider's internal SDK boundary."""

    def __init__(self, completions: FakeCompletions) -> None:
        self.chat = SimpleNamespace(completions=completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def provider_config(**overrides: Any) -> ProviderConfig:
    data: dict[str, object] = {
        "name": "deepseek",
        "protocol": "openai-compatible",
        "api_key": SYNTHETIC_SECRET,
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "stream": True,
        "timeout_seconds": 30,
        "temperature": 0.5,
        "max_tokens": 1024,
        "extra_params": {"frequency_penalty": 0.1},
    }
    data.update(overrides)
    return ProviderConfig.model_validate(data)


def messages() -> tuple[Message, ...]:
    return (
        Message(MessageRole.SYSTEM, "You are Codewright."),
        Message(MessageRole.USER, "Hello"),
    )


def response(content: object = "Hello back") -> object:
    return SimpleNamespace(
        model="deepseek-chat",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )


def chunk(content: object, *, include_choice: bool = True) -> object:
    choices = [SimpleNamespace(delta=SimpleNamespace(content=content))] if include_choice else []
    return SimpleNamespace(choices=choices)


@pytest.mark.asyncio
async def test_chat_builds_openai_compatible_request_and_parses_result() -> None:
    completions = FakeCompletions(result=response())
    provider: Provider = DeepSeekProvider(provider_config(), client=FakeClient(completions))

    result = await provider.chat(messages(), parameters={"temperature": 0.2})

    assert result.message == Message(MessageRole.ASSISTANT, "Hello back")
    assert result.model == "deepseek-chat"
    assert result.finish_reason == "stop"
    assert result.usage is not None
    assert result.usage.total_tokens == 7
    assert completions.requests == [
        {
            "frequency_penalty": 0.1,
            "temperature": 0.2,
            "max_tokens": 1024,
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are Codewright."},
                {"role": "user", "content": "Hello"},
            ],
            "stream": False,
            "timeout": 30.0,
        }
    ]


@pytest.mark.asyncio
async def test_stream_skips_empty_chunks_and_completes_once() -> None:
    stream = FakeStream(
        [
            chunk(None),
            chunk("", include_choice=True),
            chunk(None, include_choice=False),
            chunk("A"),
            chunk("B"),
        ]
    )
    completions = FakeCompletions(result=stream)
    provider: Provider = DeepSeekProvider(provider_config(), client=FakeClient(completions))

    events = [event async for event in provider.stream_chat(messages())]

    assert [event.text for event in events] == ["A", "B", ""]
    assert [event.done for event in events] == [False, False, True]
    assert sum(event.done for event in events) == 1
    assert all(event.error is None for event in events)
    assert completions.requests[0]["stream"] is True


@pytest.mark.asyncio
async def test_stream_maps_midstream_failure_without_completion() -> None:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    stream = FakeStream(
        [chunk("partial")],
        error=openai.APIConnectionError(request=request),
    )
    provider = DeepSeekProvider(
        provider_config(), client=FakeClient(FakeCompletions(result=stream))
    )

    events = [event async for event in provider.stream_chat(messages())]

    assert [event.text for event in events] == ["partial", ""]
    assert events[-1].done is False
    assert isinstance(events[-1].error, LLMNetworkError)


@pytest.mark.asyncio
async def test_chat_rejects_missing_or_invalid_content() -> None:
    provider = DeepSeekProvider(
        provider_config(), client=FakeClient(FakeCompletions(result=response(None)))
    )

    with pytest.raises(LLMResponseError):
        await provider.chat(messages())


@pytest.mark.asyncio
async def test_chat_rejects_protected_parameter_override() -> None:
    provider = DeepSeekProvider(
        provider_config(), client=FakeClient(FakeCompletions(result=response()))
    )

    with pytest.raises(LLMResponseError, match="protected fields: model"):
        await provider.chat(messages(), parameters={"model": "other-model"})


def sdk_error(error_type: type[Exception]) -> Exception:
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    if error_type is openai.APIConnectionError:
        return openai.APIConnectionError(request=request)
    if error_type is openai.APITimeoutError:
        return openai.APITimeoutError(request)

    status_by_error_name = {
        "AuthenticationError": 401,
        "PermissionDeniedError": 403,
        "NotFoundError": 404,
        "RateLimitError": 429,
        "InternalServerError": 500,
    }
    status = status_by_error_name[error_type.__name__]
    sdk_response = httpx.Response(status, request=request)
    status_error_type = cast(Any, error_type)
    return cast(
        Exception,
        status_error_type("unsafe sdk detail", response=sdk_response, body=None),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sdk_error_type", "expected_error_type"),
    [
        (openai.AuthenticationError, LLMAuthenticationError),
        (openai.PermissionDeniedError, LLMAuthenticationError),
        (openai.APIConnectionError, LLMNetworkError),
        (openai.APITimeoutError, LLMTimeoutError),
        (openai.NotFoundError, LLMModelNotFoundError),
        (openai.RateLimitError, LLMRateLimitError),
        (openai.InternalServerError, LLMServiceError),
    ],
)
async def test_chat_maps_sdk_errors(
    sdk_error_type: type[Exception], expected_error_type: type[LLMError]
) -> None:
    provider = DeepSeekProvider(
        provider_config(),
        client=FakeClient(FakeCompletions(error=sdk_error(sdk_error_type))),
    )

    with pytest.raises(expected_error_type) as captured:
        await provider.chat(messages())

    assert "unsafe sdk detail" not in str(captured.value)
    assert SYNTHETIC_SECRET not in str(captured.value)


@pytest.mark.asyncio
async def test_close_releases_client() -> None:
    client = FakeClient(FakeCompletions(result=response()))
    provider = DeepSeekProvider(provider_config(), client=client)

    await provider.close()

    assert client.closed is True


@pytest.mark.asyncio
async def test_provider_logs_safe_metadata_without_message_content() -> None:
    log_stream = StringIO()
    configure_logging("DEBUG", stream=log_stream, sensitive_values=(SYNTHETIC_SECRET,))
    completions = FakeCompletions(result=response("unique-private-model-reply"))
    provider = DeepSeekProvider(provider_config(), client=FakeClient(completions))

    await provider.chat(
        (
            Message(MessageRole.SYSTEM, "You are Codewright."),
            Message(MessageRole.USER, "unique-private-user-message"),
        )
    )

    output = log_stream.getvalue()
    assert "provider=deepseek" in output
    assert "model=deepseek-chat" in output
    assert "request_id=" in output
    assert "elapsed_seconds=" in output
    assert SYNTHETIC_SECRET not in output
    assert "unique-private-user-message" not in output
    assert "unique-private-model-reply" not in output


@pytest.mark.asyncio
async def test_factory_creates_deepseek_provider() -> None:
    provider = create_provider(provider_config())

    assert isinstance(provider, DeepSeekProvider)
    assert provider.provider_name == "deepseek"
    assert provider.model_name == "deepseek-chat"
    assert SYNTHETIC_SECRET not in repr(provider)
    await provider.close()


def test_factory_rejects_unknown_protocol() -> None:
    config = provider_config().model_copy(update={"protocol": "unknown"})

    with pytest.raises(ConfigError, match="Unsupported provider protocol"):
        create_provider(config)
