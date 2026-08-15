"""Vendor-neutral messages, tool calls, responses, and stream events."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Self

from codewright.llm.errors import LLMError


class MessageRole(StrEnum):
    """Roles supported by the vendor-neutral conversation model."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


def _immutable_mapping(value: Mapping[str, object], *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One complete, vendor-neutral model request to invoke a tool."""

    id: str
    name: str
    arguments_json: str = "{}"

    def __post_init__(self) -> None:
        for field_name in ("id", "name", "arguments_json"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")

        normalized_id = self.id.strip()
        normalized_name = self.name.strip()
        if not normalized_id:
            raise ValueError("tool call id must not be empty")
        if not normalized_name:
            raise ValueError("tool call name must not be empty")

        object.__setattr__(self, "id", normalized_id)
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "arguments_json", self.arguments_json or "{}")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """One bounded tool result associated with a model tool call."""

    tool_call_id: str
    tool_name: str
    content: str
    is_error: bool = False
    error_code: str | None = None
    truncated: bool = False
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        for field_name in ("tool_call_id", "tool_name", "content"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")
        if not self.tool_call_id.strip():
            raise ValueError("tool_call_id must not be empty")
        if not self.tool_name.strip():
            raise ValueError("tool_name must not be empty")
        if not isinstance(self.is_error, bool):
            raise TypeError("is_error must be a boolean")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a boolean")
        if self.error_code is not None and not isinstance(self.error_code, str):
            raise TypeError("error_code must be a string or None")

        normalized_error_code = self.error_code.strip() if self.error_code is not None else None
        if self.is_error and not normalized_error_code:
            raise ValueError("an error tool result must have an error_code")
        if not self.is_error and normalized_error_code:
            raise ValueError("a successful tool result cannot have an error_code")

        object.__setattr__(self, "tool_call_id", self.tool_call_id.strip())
        object.__setattr__(self, "tool_name", self.tool_name.strip())
        object.__setattr__(self, "error_code", normalized_error_code)
        object.__setattr__(
            self,
            "metadata",
            _immutable_mapping(self.metadata, field_name="metadata"),
        )


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A provider-independent tool definition exposed to a model."""

    name: str
    description: str
    input_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        for field_name in ("name", "description"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")
        if not self.name.strip():
            raise ValueError("tool definition name must not be empty")
        if not self.description.strip():
            raise ValueError("tool definition description must not be empty")

        schema = _immutable_mapping(self.input_schema, field_name="input_schema")
        if schema.get("type") != "object":
            raise ValueError("tool definition input_schema type must be object")

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "input_schema", schema)


@dataclass(frozen=True, slots=True)
class Message:
    """A text or tool message independent of any provider SDK."""

    role: MessageRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("role must be a MessageRole")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")

        calls = tuple(self.tool_calls)
        results = tuple(self.tool_results)
        if not all(isinstance(call, ToolCall) for call in calls):
            raise TypeError("tool_calls must contain only ToolCall values")
        if not all(isinstance(result, ToolResult) for result in results):
            raise TypeError("tool_results must contain only ToolResult values")
        object.__setattr__(self, "tool_calls", calls)
        object.__setattr__(self, "tool_results", results)

        if self.role in {MessageRole.SYSTEM, MessageRole.USER}:
            if not self.content.strip():
                raise ValueError("content must not be empty")
            if calls or results:
                raise ValueError("system and user messages cannot contain tool data")
        elif self.role is MessageRole.ASSISTANT:
            if results:
                raise ValueError("assistant messages cannot contain tool results")
            if not self.content.strip() and not calls:
                raise ValueError("content must not be empty")
        else:
            if self.content:
                raise ValueError("tool messages cannot contain text content")
            if calls:
                raise ValueError("tool messages cannot contain tool calls")
            if not results:
                raise ValueError("tool messages must contain tool results")


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Dynamic request-only context excluded from persistent conversation history."""

    environment: str = ""
    reminder: str = ""

    def __post_init__(self) -> None:
        for field_name in ("environment", "reminder"):
            if not isinstance(getattr(self, field_name), str):
                raise TypeError(f"{field_name} must be a string")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Optional token accounting returned by a provider."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cache_write_tokens",
            "cache_read_tokens",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ChatResult:
    """A complete response returned by a non-streaming provider call."""

    message: Message
    model: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, Message):
            raise TypeError("message must be a Message")
        if self.message.role is not MessageRole.ASSISTANT:
            raise ValueError("chat result message must have the assistant role")

        normalized_model = self.model.strip()
        if not normalized_model:
            raise ValueError("model must not be empty")
        object.__setattr__(self, "model", normalized_model)

        if self.finish_reason is not None:
            normalized_reason = self.finish_reason.strip()
            object.__setattr__(self, "finish_reason", normalized_reason or None)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """One text, tool-call, usage, completion, or error streaming event."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None
    done: bool = False
    error: LLMError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        calls = tuple(self.tool_calls)
        if not all(isinstance(call, ToolCall) for call in calls):
            raise TypeError("tool_calls must contain only ToolCall values")
        object.__setattr__(self, "tool_calls", calls)
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be a TokenUsage")
        if not isinstance(self.done, bool):
            raise TypeError("done must be a boolean")
        if self.error is not None and not isinstance(self.error, LLMError):
            raise TypeError("error must be an LLMError")

        states = sum(
            (
                bool(self.text),
                bool(calls),
                self.usage is not None,
                self.done,
                self.error is not None,
            )
        )
        if states != 1:
            raise ValueError("a stream event must have exactly one event state")

    @classmethod
    def delta(cls, text: str) -> Self:
        """Create a text delta event."""
        return cls(text=text)

    @classmethod
    def completed(cls) -> Self:
        """Create a successful completion event."""
        return cls(done=True)

    @classmethod
    def tool_calls_ready(cls, calls: tuple[ToolCall, ...]) -> Self:
        """Create an event containing complete ordered tool calls."""
        return cls(tool_calls=calls)

    @classmethod
    def usage_report(cls, usage: TokenUsage) -> Self:
        """Create an event containing token accounting for one request."""
        return cls(usage=usage)

    @classmethod
    def failed(cls, error: LLMError) -> Self:
        """Create an error event."""
        return cls(error=error)
