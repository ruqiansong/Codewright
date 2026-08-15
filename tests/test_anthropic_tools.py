"""Offline Anthropic Messages tool-call protocol tests."""

from collections.abc import AsyncIterator
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from codewright.config import ProviderConfig
from codewright.llm import (
    Message,
    MessageRole,
    PromptTooLongError,
    RequestContext,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from codewright.llm.anthropic_provider import AnthropicProvider
from codewright.llm.factory import create_provider


class FakeStream:
    def __init__(self, events: list[object], final_message: object) -> None:
        self._events = iter(events)
        self._final_message = final_message

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None

    async def get_final_message(self) -> object:
        return self._final_message


class FakeManager:
    def __init__(self, stream: FakeStream) -> None:
        self.stream = stream

    async def __aenter__(self) -> FakeStream:
        return self.stream

    async def __aexit__(self, *args: object) -> None:
        return None


class FakeMessages:
    def __init__(self, stream: FakeStream | Exception, create_result: object | None = None) -> None:
        self._stream = stream
        self._create_result = create_result
        self.stream_requests: list[dict[str, object]] = []
        self.create_requests: list[dict[str, object]] = []

    def stream(self, **parameters: object) -> FakeManager:
        self.stream_requests.append(parameters)
        if isinstance(self._stream, Exception):
            raise self._stream
        return FakeManager(self._stream)

    async def create(self, **parameters: object) -> object:
        self.create_requests.append(parameters)
        assert self._create_result is not None
        return self._create_result


class FakeClient:
    def __init__(self, messages: FakeMessages) -> None:
        self.messages = messages
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def config() -> ProviderConfig:
    return ProviderConfig.model_validate(
        {
            "name": "anthropic",
            "protocol": "anthropic",
            "api_key": "synthetic-key",
            "base_url": "https://api.anthropic.com",
            "model": "claude-sonnet-4-5",
            "max_tokens": 500,
        }
    )


def definition() -> ToolDefinition:
    return ToolDefinition(
        "read_file",
        "Read a file.",
        {"type": "object", "properties": {"path": {"type": "string"}}},
    )


def final_message(
    *blocks: object,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> object:
    return SimpleNamespace(
        model="claude-sonnet-4-5",
        content=list(blocks),
        stop_reason="tool_use",
        usage=SimpleNamespace(
            input_tokens=5,
            output_tokens=3,
            cache_creation_input_tokens=cache_write_tokens,
            cache_read_input_tokens=cache_read_tokens,
        ),
    )


def delta(delta_type: str, **values: object) -> object:
    return SimpleNamespace(delta=SimpleNamespace(type=delta_type, **values))


@pytest.mark.asyncio
async def test_anthropic_stream_emits_text_discards_thinking_and_extracts_tool_use() -> None:
    final = final_message(
        SimpleNamespace(type="text", text="I will inspect it."),
        SimpleNamespace(
            type="tool_use",
            id="tool-1",
            name="read_file",
            input={"path": "README.md"},
        ),
    )
    resource = FakeMessages(
        FakeStream(
            [
                delta("thinking_delta", thinking="private reasoning"),
                delta("text_delta", text="I will inspect it."),
                delta("input_json_delta", partial_json='{"path":"README.md"}'),
            ],
            final,
        )
    )
    provider = AnthropicProvider(config(), client=FakeClient(resource))

    events = [
        event
        async for event in provider.stream_chat(
            (
                Message(MessageRole.SYSTEM, "You are Codewright."),
                Message(MessageRole.USER, "read"),
            ),
            tools=(definition(),),
            request_context=RequestContext(
                environment="Environment:\nModel: claude-sonnet-4-5",
                reminder="<system-reminder>inspect only</system-reminder>",
            ),
        )
    ]

    assert [event.text for event in events] == ["I will inspect it.", "", "", ""]
    assert events[1].tool_calls == (ToolCall("tool-1", "read_file", '{"path":"README.md"}'),)
    assert events[2].usage == TokenUsage(input_tokens=5, output_tokens=3, total_tokens=8)
    assert events[3].done is True
    assert "private reasoning" not in repr(events)
    request = resource.stream_requests[0]
    assert request["system"] == [
        {
            "type": "text",
            "text": "You are Codewright.",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Environment:\nModel: claude-sonnet-4-5"},
    ]
    assert request["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "read"},
                {
                    "type": "text",
                    "text": "<system-reminder>inspect only</system-reminder>",
                },
            ],
        }
    ]
    assert request["tools"] == [
        {
            "name": "read_file",
            "description": "Read a file.",
            "input_schema": dict(definition().input_schema),
        }
    ]


@pytest.mark.asyncio
async def test_anthropic_converts_tool_history_and_disables_thinking() -> None:
    final = final_message(SimpleNamespace(type="text", text="done"))
    resource = FakeMessages(FakeStream([delta("text_delta", text="done")], final))
    provider_config = config().model_copy(
        update={"extra_params": {"thinking": {"type": "enabled"}}}
    )
    provider = AnthropicProvider(provider_config, client=FakeClient(resource))
    call = ToolCall("tool-1", "read_file", '{"path":"README.md"}')
    result = ToolResult("tool-1", "read_file", "missing", is_error=True, error_code="not_found")

    events = [
        event
        async for event in provider.stream_chat(
            (
                Message(MessageRole.USER, "read"),
                Message(MessageRole.ASSISTANT, "", tool_calls=(call,)),
                Message(MessageRole.TOOL, "", tool_results=(result,)),
            ),
            request_context=RequestContext(reminder="<system-reminder>read only</system-reminder>"),
        )
    ]

    assert events[-1].done is True
    request = resource.stream_requests[0]
    assert "thinking" not in request
    assert request["messages"] == [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool-1",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool-1",
                    "content": "missing",
                    "is_error": True,
                },
                {
                    "type": "text",
                    "text": "<system-reminder>read only</system-reminder>",
                },
            ],
        },
    ]


@pytest.mark.asyncio
async def test_anthropic_chat_parses_text_calls_and_usage() -> None:
    response = final_message(
        SimpleNamespace(type="text", text="checking"),
        SimpleNamespace(type="tool_use", id="tool-1", name="read_file", input={}),
        cache_write_tokens=7,
        cache_read_tokens=4,
    )
    resource = FakeMessages(FakeStream([], response), create_result=response)
    provider = AnthropicProvider(config(), client=FakeClient(resource))

    result = await provider.chat(
        (
            Message(MessageRole.SYSTEM, "You are Codewright."),
            Message(MessageRole.USER, "read"),
        ),
        request_context=RequestContext(
            environment="Environment:\nModel: claude-sonnet-4-5",
            reminder="<system-reminder>use tools</system-reminder>",
        ),
    )

    assert result.message.content == "checking"
    assert result.message.tool_calls == (ToolCall("tool-1", "read_file", "{}"),)
    assert result.usage is not None
    assert result.usage.total_tokens == 8
    assert result.usage.cache_write_tokens == 7
    assert result.usage.cache_read_tokens == 4
    assert resource.create_requests[0]["system"] == [
        {
            "type": "text",
            "text": "You are Codewright.",
            "cache_control": {"type": "ephemeral"},
        },
        {"type": "text", "text": "Environment:\nModel: claude-sonnet-4-5"},
    ]


@pytest.mark.asyncio
async def test_factory_creates_anthropic_provider() -> None:
    provider = create_provider(config())

    assert isinstance(provider, AnthropicProvider)
    assert provider.provider_name == "anthropic"
    await provider.close()


def anthropic_bad_request(message: str, body: object) -> anthropic.BadRequestError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(400, request=request)
    return anthropic.BadRequestError(message, response=response, body=body)


@pytest.mark.asyncio
async def test_anthropic_stream_maps_structured_context_error_to_ptl() -> None:
    original = anthropic_bad_request(
        "generic bad request",
        {"error": {"type": "request_too_large"}},
    )
    resource = FakeMessages(original)
    provider = AnthropicProvider(config(), client=FakeClient(resource))

    events = [event async for event in provider.stream_chat((Message(MessageRole.USER, "x"),))]

    assert len(events) == 1
    assert isinstance(events[0].error, PromptTooLongError)
    assert events[0].error.__cause__ is original


@pytest.mark.asyncio
async def test_anthropic_ptl_text_fallback_does_not_match_unrelated_400() -> None:
    ptl = anthropic_bad_request("Prompt is too long for this model", {"error": {}})
    unrelated = anthropic_bad_request(
        "invalid temperature",
        {"error": {"type": "invalid_request_error"}},
    )

    ptl_events = [
        event
        async for event in AnthropicProvider(
            config(), client=FakeClient(FakeMessages(ptl))
        ).stream_chat((Message(MessageRole.USER, "x"),))
    ]
    unrelated_events = [
        event
        async for event in AnthropicProvider(
            config(), client=FakeClient(FakeMessages(unrelated))
        ).stream_chat((Message(MessageRole.USER, "x"),))
    ]

    assert isinstance(ptl_events[0].error, PromptTooLongError)
    assert not isinstance(unrelated_events[0].error, PromptTooLongError)
