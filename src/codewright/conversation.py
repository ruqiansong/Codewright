"""In-memory conversation history for a Codewright session."""

import threading
from collections.abc import Callable, Sequence

from codewright.llm import Message, MessageRole, ToolCall, ToolResult


class Conversation:
    """Maintain ordered messages for one non-persistent process session."""

    def __init__(
        self,
        system_prompt: str,
        on_append: Callable[[Message], None] | None = None,
        on_replace: Callable[[list[Message]], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._messages = [Message(MessageRole.SYSTEM, system_prompt)]
        self._on_append = on_append
        self._on_replace = on_replace

    @classmethod
    def from_messages(
        cls,
        system_prompt: str,
        messages: Sequence[Message],
        on_append: Callable[[Message], None] | None = None,
        on_replace: Callable[[list[Message]], None] | None = None,
    ) -> "Conversation":
        """Build a conversation from persisted non-system messages."""
        restored = list(messages)
        if not all(isinstance(message, Message) for message in restored):
            raise ValueError("messages must contain only Message values")
        if any(message.role is MessageRole.SYSTEM for message in restored):
            raise ValueError("restored messages must not contain a system message")
        conversation = cls(system_prompt, on_append=on_append, on_replace=on_replace)
        with conversation._lock:
            conversation._messages.extend(restored)
        return conversation

    def add_user(self, content: str) -> Message:
        """Append and return a user message."""
        with self._lock:
            message = Message(MessageRole.USER, content)
            self._messages.append(message)
        if self._on_append is not None:
            self._on_append(message)
        return message

    def add_assistant(self, content: str) -> Message:
        """Append and return a complete assistant message."""
        with self._lock:
            message = Message(MessageRole.ASSISTANT, content)
            self._messages.append(message)
        if self._on_append is not None:
            self._on_append(message)
        return message

    def add_assistant_with_tool_calls(
        self,
        content: str,
        calls: Sequence[ToolCall],
    ) -> Message:
        """Append an assistant message containing one or more tool calls."""
        with self._lock:
            tool_calls = tuple(calls)
            if not tool_calls:
                raise ValueError("calls must not be empty")
            message = Message(MessageRole.ASSISTANT, content, tool_calls=tool_calls)
            self._messages.append(message)
        if self._on_append is not None:
            self._on_append(message)
        return message

    def add_tool_results(self, results: Sequence[ToolResult]) -> Message:
        """Append one protocol-neutral batch of tool results."""
        with self._lock:
            tool_results = tuple(results)
            if not tool_results:
                raise ValueError("results must not be empty")
            message = Message(MessageRole.TOOL, "", tool_results=tool_results)
            self._messages.append(message)
        if self._on_append is not None:
            self._on_append(message)
        return message

    def messages(self) -> tuple[Message, ...]:
        """Return an immutable snapshot of the current ordered history."""
        with self._lock:
            return tuple(self._messages)

    def last_role(self) -> MessageRole:
        """Return the role of the final message in the conversation."""
        with self._lock:
            return self._messages[-1].role

    def clear(self, system_prompt: str | None = None) -> None:
        """Reset the history while retaining or replacing the system prompt."""
        with self._lock:
            prompt = self._messages[0].content if system_prompt is None else system_prompt
            self._messages = [Message(MessageRole.SYSTEM, prompt)]

    def replace_history(self, messages: Sequence[Message] | None) -> None:
        """Atomically replace history while preserving one leading system message."""
        with self._lock:
            if messages is None:
                raise ValueError("messages must not be None")
            replacement = list(messages)
            if not replacement:
                raise ValueError("messages must not be empty")
            if not all(isinstance(message, Message) for message in replacement):
                raise ValueError("messages must contain only Message values")
            system_indexes = [
                index
                for index, message in enumerate(replacement)
                if message.role is MessageRole.SYSTEM
            ]
            if system_indexes != [0]:
                raise ValueError("history must contain exactly one leading system message")
            self._messages = replacement
            snapshot = list(self._messages)
        if self._on_replace is not None:
            self._on_replace(snapshot)

    def replace_system_prompt(self, system_prompt: str) -> None:
        """Replace only the leading system message without persistence callbacks."""
        message = Message(MessageRole.SYSTEM, system_prompt)
        with self._lock:
            self._messages[0] = message
