"""Contract tests for vendor-neutral LLM providers and models."""

from collections.abc import AsyncIterator, Sequence

import pytest

from codewright.llm import (
    ChatResult,
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
    RequestContext,
    RequestParameters,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class FakeProvider:
    """Small provider implementation used to verify the public contract."""

    def __init__(self) -> None:
        self.requests: list[tuple[Message, ...]] = []
        self.tools: list[tuple[ToolDefinition, ...]] = []
        self.contexts: list[RequestContext | None] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del parameters
        self.requests.append(tuple(messages))
        self.tools.append(tuple(tools))
        self.contexts.append(request_context)
        return ChatResult(
            message=Message(MessageRole.ASSISTANT, "complete response"),
            model=self.model_name,
            usage=TokenUsage(input_tokens=4, output_tokens=2, total_tokens=6),
            finish_reason="stop",
        )

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters
        self.requests.append(tuple(messages))
        self.tools.append(tuple(tools))
        self.contexts.append(request_context)
        yield StreamEvent.delta("streamed ")
        yield StreamEvent.delta("response")
        yield StreamEvent.completed()


def conversation_messages() -> tuple[Message, ...]:
    return (
        Message(MessageRole.SYSTEM, "You are Codewright."),
        Message(MessageRole.USER, "Hello"),
    )


def test_message_supports_only_defined_roles() -> None:
    message = Message(MessageRole.USER, "  preserve whitespace  ")

    assert message.role is MessageRole.USER
    assert message.content == "  preserve whitespace  "
    assert {role.value for role in MessageRole} == {"system", "user", "assistant", "tool"}


def test_message_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="content must not be empty"):
        Message(MessageRole.USER, "   ")


def test_message_rejects_untyped_role() -> None:
    with pytest.raises(TypeError, match="role must be a MessageRole"):
        Message("user", "hello")  # type: ignore[arg-type]


def test_token_usage_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        TokenUsage(input_tokens=-1, output_tokens=0, total_tokens=0)


def test_token_usage_rejects_non_integer_values() -> None:
    with pytest.raises(ValueError, match="input_tokens"):
        TokenUsage(input_tokens=1.5, output_tokens=0, total_tokens=0)  # type: ignore[arg-type]


def test_request_context_is_immutable_and_validates_text_fields() -> None:
    context = RequestContext(environment="Environment: test", reminder="remember")

    assert context.environment == "Environment: test"
    assert context.reminder == "remember"
    with pytest.raises(TypeError, match="environment must be a string"):
        RequestContext(environment=None)  # type: ignore[arg-type]


def test_token_usage_supports_cache_accounting() -> None:
    usage = TokenUsage(
        input_tokens=10,
        output_tokens=2,
        total_tokens=12,
        cache_write_tokens=8,
        cache_read_tokens=5,
    )

    assert usage.cache_write_tokens == 8
    assert usage.cache_read_tokens == 5
    with pytest.raises(ValueError, match="cache_read_tokens"):
        TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, cache_read_tokens=-1)


def test_chat_result_requires_assistant_message() -> None:
    with pytest.raises(ValueError, match="assistant role"):
        ChatResult(message=Message(MessageRole.USER, "hello"), model="fake-model")


def test_chat_result_normalizes_model_and_finish_reason() -> None:
    result = ChatResult(
        message=Message(MessageRole.ASSISTANT, "hello"),
        model="  fake-model  ",
        finish_reason="   ",
    )

    assert result.model == "fake-model"
    assert result.finish_reason is None


def test_stream_event_factories_create_mutually_exclusive_events() -> None:
    error = LLMResponseError()
    usage = TokenUsage(input_tokens=3, output_tokens=2, total_tokens=5)

    assert StreamEvent.delta("text") == StreamEvent(text="text")
    assert StreamEvent.usage_report(usage) == StreamEvent(usage=usage)
    assert StreamEvent.completed() == StreamEvent(done=True)
    assert StreamEvent.failed(error) == StreamEvent(error=error)


def test_tool_models_and_messages_enforce_role_invariants() -> None:
    call = ToolCall(" call-1 ", " read_file ", "")
    result = ToolResult("call-1", "read_file", "file content", metadata={"lines": 1})
    definition = ToolDefinition(
        "read_file",
        "Read a UTF-8 text file.",
        {"type": "object", "properties": {"path": {"type": "string"}}},
    )

    assert call == ToolCall("call-1", "read_file", "{}")
    assert definition.input_schema["type"] == "object"
    assert Message(MessageRole.ASSISTANT, "", tool_calls=(call,)).tool_calls == (call,)
    assert Message(MessageRole.TOOL, "", tool_results=(result,)).tool_results == (result,)
    assert StreamEvent.tool_calls_ready((call,)).tool_calls == (call,)

    with pytest.raises(ValueError, match="content must not be empty"):
        Message(MessageRole.ASSISTANT, "")
    with pytest.raises(ValueError, match="cannot contain tool data"):
        Message(MessageRole.USER, "run it", tool_calls=(call,))
    with pytest.raises(ValueError, match="must contain tool results"):
        Message(MessageRole.TOOL, "")


def test_tool_model_metadata_and_schema_are_defensively_copied() -> None:
    metadata: dict[str, object] = {"count": 1}
    schema: dict[str, object] = {"type": "object", "properties": {}}
    result = ToolResult("call-1", "read_file", "ok", metadata=metadata)
    definition = ToolDefinition("read_file", "Read a file", schema)

    metadata["count"] = 2
    schema["type"] = "string"

    assert result.metadata == {"count": 1}
    assert definition.input_schema["type"] == "object"


@pytest.mark.parametrize(
    "event_values",
    [
        {},
        {"text": "text", "done": True},
        {"text": "text", "error": LLMResponseError()},
        {
            "text": "text",
            "usage": TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        },
        {"done": True, "error": LLMResponseError()},
        {"tool_calls": (ToolCall("call-1", "read_file"),), "done": True},
    ],
)
def test_stream_event_rejects_ambiguous_states(event_values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        StreamEvent(**event_values)  # type: ignore[arg-type]


def test_stream_event_rejects_non_llm_error() -> None:
    with pytest.raises(TypeError, match="error must be an LLMError"):
        StreamEvent(error=ValueError("unsafe provider error"))  # type: ignore[arg-type]


def test_stream_event_rejects_non_token_usage() -> None:
    with pytest.raises(TypeError, match="usage must be a TokenUsage"):
        StreamEvent(usage={"input_tokens": 1})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("error_type", "retryable"),
    [
        (LLMAuthenticationError, False),
        (LLMNetworkError, True),
        (LLMTimeoutError, True),
        (LLMModelNotFoundError, False),
        (LLMRateLimitError, True),
        (LLMServiceError, True),
        (LLMResponseError, False),
    ],
)
def test_error_categories_have_safe_defaults(error_type: type[LLMError], retryable: bool) -> None:
    error = error_type(code="test-code")

    assert str(error) == error.safe_message
    assert error.safe_message
    assert error.code == "test-code"
    assert error.retryable is retryable


@pytest.mark.asyncio
async def test_fake_provider_satisfies_non_streaming_contract() -> None:
    fake_provider = FakeProvider()
    provider: Provider = fake_provider
    messages = conversation_messages()

    tools = (ToolDefinition("read_file", "Read a file", {"type": "object"}),)
    context = RequestContext(environment="Environment: test")
    result = await provider.chat(
        messages,
        parameters={"temperature": 0.5},
        tools=tools,
        request_context=context,
    )

    assert isinstance(provider, Provider)
    assert provider.provider_name == "fake"
    assert provider.model_name == "fake-model"
    assert result.message.content == "complete response"
    assert fake_provider.requests == [messages]
    assert fake_provider.tools == [tools]
    assert fake_provider.contexts == [context]


@pytest.mark.asyncio
async def test_fake_provider_satisfies_streaming_contract() -> None:
    fake_provider = FakeProvider()
    provider: Provider = fake_provider
    messages = conversation_messages()

    context = RequestContext(reminder="remember")
    events = [event async for event in provider.stream_chat(messages, request_context=context)]

    assert [event.text for event in events] == ["streamed ", "response", ""]
    assert [event.done for event in events] == [False, False, True]
    assert sum(event.done for event in events) == 1
    assert all(event.error is None for event in events)
    assert fake_provider.requests == [messages]
    assert fake_provider.contexts == [context]
