"""Safe construction and detection of forked Agent conversations."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from codewright.conversation import Conversation
from codewright.llm import Message, MessageRole, ToolCall, ToolResult

FORK_BOILERPLATE = """<codewright-fork-context>
You are running in an isolated fork of the parent conversation. Complete only the task below.
</codewright-fork-context>

"""
_MISSING_RESULT_CONTENT = "The parent tool call had no result before this conversation was forked."


def build_fork_conversation(parent: Conversation, prompt: str) -> Conversation:
    """Deep-copy parent history, repair dangling calls, and append one fork task."""
    if not isinstance(parent, Conversation):
        raise TypeError("parent must be a Conversation")
    messages = parent.messages()
    return build_fork_conversation_from_messages(messages[0].content, messages[1:], prompt)


def build_fork_conversation_from_messages(
    system_prompt: str,
    messages: Sequence[Message],
    prompt: str,
) -> Conversation:
    """Build a fork from an explicit system prompt and non-system history."""
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    history = tuple(messages)
    if not all(isinstance(message, Message) for message in history):
        raise TypeError("messages must contain only Message values")
    if any(message.role is MessageRole.SYSTEM for message in history):
        raise ValueError("messages must not contain a system message")

    copied = [_clone_message(message) for message in history]
    missing = _missing_tool_results(copied)
    if missing:
        copied.append(Message(MessageRole.TOOL, "", tool_results=missing))
    copied.append(Message(MessageRole.USER, FORK_BOILERPLATE + prompt))
    return Conversation.from_messages(system_prompt, copied)


def is_fork_context(value: Conversation | Sequence[Message]) -> bool:
    """Return whether any message contains the stable fork marker."""
    messages = value.messages() if isinstance(value, Conversation) else tuple(value)
    return any(
        isinstance(message, Message) and FORK_BOILERPLATE.strip() in message.content
        for message in messages
    )


def _clone_message(message: Message) -> Message:
    calls = tuple(ToolCall(call.id, call.name, call.arguments_json) for call in message.tool_calls)
    results = tuple(
        ToolResult(
            result.tool_call_id,
            result.tool_name,
            result.content,
            is_error=result.is_error,
            error_code=result.error_code,
            truncated=result.truncated,
            metadata=deepcopy(dict(result.metadata)),
        )
        for result in message.tool_results
    )
    return Message(message.role, message.content, tool_calls=calls, tool_results=results)


def _missing_tool_results(messages: Sequence[Message]) -> tuple[ToolResult, ...]:
    calls: dict[str, ToolCall] = {}
    completed: set[str] = set()
    for message in messages:
        for call in message.tool_calls:
            calls[call.id] = call
        completed.update(result.tool_call_id for result in message.tool_results)
    return tuple(
        ToolResult(
            call.id,
            call.name,
            _MISSING_RESULT_CONTENT,
            is_error=True,
            error_code="fork_missing_tool_result",
        )
        for call_id, call in calls.items()
        if call_id not in completed
    )


__all__ = [
    "FORK_BOILERPLATE",
    "build_fork_conversation",
    "build_fork_conversation_from_messages",
    "is_fork_context",
]
