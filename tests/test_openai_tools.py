"""Offline OpenAI-compatible tool-call protocol tests."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import openai
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
from codewright.llm.openai_provider import OpenAICompatibleProvider


class FakeStream:
    def __init__(self, chunks: list[object]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeCompletions:
    def __init__(self, result: object) -> None:
        self.result = result
        self.requests: list[dict[str, object]] = []

    async def create(self, **parameters: object) -> object:
        self.requests.append(parameters)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result: object) -> None:
        self.completions = FakeCompletions(result)
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def config() -> ProviderConfig:
    return ProviderConfig.model_validate(
        {
            "name": "compatible",
            "protocol": "openai-compatible",
            "api_key": "synthetic-key",
            "base_url": "https://example.invalid/v1",
            "model": "test-model",
            "max_tokens": 100,
        }
    )


def definition() -> ToolDefinition:
    return ToolDefinition(
        "read_file",
        "Read a file.",
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )


def chunk(*, text: str | None = None, calls: list[object] | None = None) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=calls))],
        usage=None,
    )


def usage_chunk(input_tokens: int, output_tokens: int, *, cached_tokens: int = 0) -> object:
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def call_part(
    index: int,
    *,
    call_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> object:
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


@pytest.mark.asyncio
async def test_openai_stream_injects_tools_and_assembles_interleaved_calls() -> None:
    stream = FakeStream(
        [
            chunk(text="Checking "),
            chunk(
                calls=[
                    call_part(0, call_id="call-0", name="read_", arguments='{"pa'),
                    call_part(1, call_id="call-1", name="gl", arguments='{"pattern":"'),
                ]
            ),
            chunk(
                calls=[
                    call_part(1, name="ob", arguments='**/*.py"}'),
                    call_part(0, name="file", arguments='th":"README.md"}'),
                ]
            ),
            usage_chunk(12, 4, cached_tokens=9),
        ]
    )
    client = FakeClient(stream)
    provider = OpenAICompatibleProvider(config(), client=client)

    events = [
        event
        async for event in provider.stream_chat(
            (
                Message(MessageRole.SYSTEM, "You are Codewright."),
                Message(MessageRole.USER, "inspect"),
            ),
            tools=(definition(),),
            request_context=RequestContext(
                environment="Environment:\nModel: test-model",
                reminder="<system-reminder>inspect only</system-reminder>",
            ),
        )
    ]

    assert events[0].text == "Checking "
    assert events[1].tool_calls == (
        ToolCall("call-0", "read_file", '{"path":"README.md"}'),
        ToolCall("call-1", "glob", '{"pattern":"**/*.py"}'),
    )
    assert events[2].usage == TokenUsage(
        input_tokens=12,
        output_tokens=4,
        total_tokens=16,
        cache_read_tokens=9,
    )
    assert events[3].done is True
    request = client.completions.requests[0]
    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file.",
                "parameters": dict(definition().input_schema),
            },
        }
    ]
    assert request["stream_options"] == {"include_usage": True}
    assert request["messages"] == [
        {
            "role": "system",
            "content": "You are Codewright.\n\nEnvironment:\nModel: test-model",
        },
        {"role": "user", "content": "inspect"},
        {"role": "user", "content": "<system-reminder>inspect only</system-reminder>"},
    ]


@pytest.mark.asyncio
async def test_openai_stream_normalizes_empty_arguments() -> None:
    client = FakeClient(
        FakeStream([chunk(calls=[call_part(0, call_id="call-0", name="read_file")])])
    )
    provider = OpenAICompatibleProvider(config(), client=client)

    events = [event async for event in provider.stream_chat((Message(MessageRole.USER, "x"),))]

    assert events[0].tool_calls == (ToolCall("call-0", "read_file", "{}"),)


@pytest.mark.asyncio
async def test_openai_converts_assistant_calls_and_tool_results_to_history() -> None:
    client = FakeClient(FakeStream([chunk(text="done")]))
    provider = OpenAICompatibleProvider(config(), client=client)
    call = ToolCall("call-1", "read_file", '{"path":"README.md"}')
    result = ToolResult("call-1", "read_file", "contents")
    messages = (
        Message(MessageRole.USER, "read"),
        Message(MessageRole.ASSISTANT, "", tool_calls=(call,)),
        Message(MessageRole.TOOL, "", tool_results=(result,)),
    )

    events = [event async for event in provider.stream_chat(messages)]

    assert events[-1].done is True
    request_messages = client.completions.requests[0]["messages"]
    assert request_messages == [
        {"role": "user", "content": "read"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
    ]


@pytest.mark.asyncio
async def test_openai_chat_parses_tool_only_response() -> None:
    raw_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="read_file", arguments='{"path":"README.md"}'),
    )
    response: Any = SimpleNamespace(
        model="test-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[raw_call]),
                finish_reason="tool_calls",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
            prompt_tokens_details={"cached_tokens": 6},
        ),
    )
    provider = OpenAICompatibleProvider(config(), client=FakeClient(response))

    result = await provider.chat(
        (
            Message(MessageRole.SYSTEM, "You are Codewright."),
            Message(MessageRole.USER, "read"),
        ),
        request_context=RequestContext(
            environment="Environment:\nModel: test-model",
            reminder="<system-reminder>use tools</system-reminder>",
        ),
    )

    assert result.message.content == ""
    assert result.message.tool_calls == (ToolCall("call-1", "read_file", '{"path":"README.md"}'),)
    assert result.usage == TokenUsage(
        input_tokens=10,
        output_tokens=2,
        total_tokens=12,
        cache_read_tokens=6,
    )


@pytest.mark.asyncio
async def test_openai_chat_parses_deepseek_cache_hit_tokens() -> None:
    response: Any = SimpleNamespace(
        model="deepseek-chat",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="OK", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=80,
            completion_tokens=1,
            total_tokens=81,
            prompt_cache_hit_tokens=64,
            prompt_cache_miss_tokens=16,
        ),
    )
    provider = OpenAICompatibleProvider(config(), client=FakeClient(response))

    result = await provider.chat((Message(MessageRole.USER, "Reply only OK."),))

    assert result.usage == TokenUsage(
        input_tokens=80,
        output_tokens=1,
        total_tokens=81,
        cache_read_tokens=64,
    )


def openai_bad_request(message: str, body: object) -> openai.BadRequestError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return openai.BadRequestError(message, response=response, body=body)


@pytest.mark.asyncio
async def test_openai_stream_maps_structured_context_error_to_ptl() -> None:
    original = openai_bad_request(
        "generic bad request",
        {"error": {"type": "invalid_request_error", "code": "context_length_exceeded"}},
    )
    provider = OpenAICompatibleProvider(config(), client=FakeClient(original))

    events = [event async for event in provider.stream_chat((Message(MessageRole.USER, "x"),))]

    assert len(events) == 1
    assert isinstance(events[0].error, PromptTooLongError)
    assert events[0].error.__cause__ is original


@pytest.mark.asyncio
async def test_openai_ptl_text_fallback_does_not_match_unrelated_400() -> None:
    ptl = openai_bad_request("Prompt is too long for this model", {"error": {}})
    unrelated = openai_bad_request(
        "temperature must be between zero and two",
        {"error": {"code": "invalid_parameter"}},
    )

    ptl_events = [
        event
        async for event in OpenAICompatibleProvider(config(), client=FakeClient(ptl)).stream_chat(
            (Message(MessageRole.USER, "x"),)
        )
    ]
    unrelated_events = [
        event
        async for event in OpenAICompatibleProvider(
            config(), client=FakeClient(unrelated)
        ).stream_chat((Message(MessageRole.USER, "x"),))
    ]

    assert isinstance(ptl_events[0].error, PromptTooLongError)
    assert not isinstance(unrelated_events[0].error, PromptTooLongError)
