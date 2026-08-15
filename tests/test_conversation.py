"""Tests for in-memory conversation history."""

import threading

import pytest

from codewright.conversation import Conversation
from codewright.llm import Message, MessageRole, ToolCall, ToolResult
from codewright.prompt import PLAN_MODE_REMINDER
from codewright.prompt import SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT

SYSTEM_PROMPT = "You are Codewright."


def test_conversation_starts_with_one_system_message() -> None:
    conversation = Conversation(SYSTEM_PROMPT)

    assert conversation.messages() == (Message(MessageRole.SYSTEM, SYSTEM_PROMPT),)


def test_default_system_prompt_initializes_conversation() -> None:
    conversation = Conversation(DEFAULT_SYSTEM_PROMPT)

    assert conversation.messages() == (Message(MessageRole.SYSTEM, DEFAULT_SYSTEM_PROMPT),)
    assert "Codewright" in DEFAULT_SYSTEM_PROMPT
    assert "read, write, and uniquely edit files" in DEFAULT_SYSTEM_PROMPT
    assert "execute shell commands" in DEFAULT_SYSTEM_PROMPT
    assert "across multiple steps" in DEFAULT_SYSTEM_PROMPT
    assert "permission decisions and sandbox boundaries" in DEFAULT_SYSTEM_PROMPT
    assert "no permission confirmation" not in DEFAULT_SYSTEM_PROMPT
    assert "Never invent tool" in DEFAULT_SYSTEM_PROMPT
    assert "PLAN MODE" in PLAN_MODE_REMINDER


def test_conversation_rejects_empty_system_prompt() -> None:
    with pytest.raises(ValueError, match="content must not be empty"):
        Conversation("   ")


def test_add_user_appends_and_returns_message() -> None:
    conversation = Conversation(SYSTEM_PROMPT)

    message = conversation.add_user("Hello")

    assert message == Message(MessageRole.USER, "Hello")
    assert conversation.messages()[-1] is message
    assert conversation.last_role() is MessageRole.USER


def test_add_assistant_appends_and_returns_complete_message() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_user("Hello")

    message = conversation.add_assistant("Hello! How can I help?")

    assert message == Message(MessageRole.ASSISTANT, "Hello! How can I help?")
    assert conversation.messages()[-1] is message


@pytest.mark.parametrize("method_name", ["add_user", "add_assistant"])
def test_conversation_rejects_empty_messages(method_name: str) -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    add_message = getattr(conversation, method_name)

    with pytest.raises(ValueError, match="content must not be empty"):
        add_message("  ")


def test_messages_preserve_role_and_insertion_order_across_turns() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_user("My name is Zhang San.")
    conversation.add_assistant("Nice to meet you, Zhang San.")
    conversation.add_user("What is my name?")

    assert conversation.messages() == (
        Message(MessageRole.SYSTEM, SYSTEM_PROMPT),
        Message(MessageRole.USER, "My name is Zhang San."),
        Message(MessageRole.ASSISTANT, "Nice to meet you, Zhang San."),
        Message(MessageRole.USER, "What is my name?"),
    )


def test_messages_returns_an_immutable_snapshot() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    snapshot = conversation.messages()

    conversation.add_user("A later message")

    assert isinstance(snapshot, tuple)
    assert snapshot == (Message(MessageRole.SYSTEM, SYSTEM_PROMPT),)
    assert len(conversation.messages()) == 2


def test_failed_or_cancelled_reply_is_not_implicitly_saved() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_user("This request will be cancelled")

    assert [message.role for message in conversation.messages()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]


def test_clear_retains_current_system_prompt() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_user("Hello")
    conversation.add_assistant("Hello")

    conversation.clear()

    assert conversation.messages() == (Message(MessageRole.SYSTEM, SYSTEM_PROMPT),)


def test_clear_can_replace_system_prompt() -> None:
    conversation = Conversation(SYSTEM_PROMPT)

    conversation.clear("A replacement system prompt.")

    assert conversation.messages() == (Message(MessageRole.SYSTEM, "A replacement system prompt."),)


def test_conversation_preserves_complete_tool_history() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    call = ToolCall("call-1", "read_file", '{"path":"README.md"}')
    result = ToolResult("call-1", "read_file", "file content")

    conversation.add_user("Read the README")
    assistant_message = conversation.add_assistant_with_tool_calls("", (call,))
    tool_message = conversation.add_tool_results((result,))
    conversation.add_assistant("The README describes Codewright.")

    assert assistant_message.tool_calls == (call,)
    assert tool_message.role is MessageRole.TOOL
    assert tool_message.tool_results == (result,)
    assert [message.role for message in conversation.messages()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert conversation.last_role() is MessageRole.ASSISTANT


def test_last_role_tracks_system_and_tool_messages() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    assert conversation.last_role() is MessageRole.SYSTEM

    call = ToolCall("call-1", "read_file")
    conversation.add_assistant_with_tool_calls("", (call,))
    conversation.add_tool_results((ToolResult("call-1", "read_file", "ok"),))

    assert conversation.last_role() is MessageRole.TOOL


def test_conversation_rejects_empty_tool_batches_and_clear_removes_them() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    call = ToolCall("call-1", "read_file")
    result = ToolResult("call-1", "read_file", "ok")

    with pytest.raises(ValueError, match="calls must not be empty"):
        conversation.add_assistant_with_tool_calls("", ())
    with pytest.raises(ValueError, match="results must not be empty"):
        conversation.add_tool_results(())

    conversation.add_assistant_with_tool_calls("", (call,))
    conversation.add_tool_results((result,))
    snapshot = conversation.messages()
    conversation.clear()

    assert snapshot[1].tool_calls == (call,)
    assert conversation.messages() == (Message(MessageRole.SYSTEM, SYSTEM_PROMPT),)


def test_conversations_do_not_share_history() -> None:
    first = Conversation(SYSTEM_PROMPT)
    first.add_user("Private to the first process session")

    second = Conversation(SYSTEM_PROMPT)

    assert second.messages() == (Message(MessageRole.SYSTEM, SYSTEM_PROMPT),)


def test_replace_history_copies_outer_sequence() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    replacement = [
        Message(MessageRole.SYSTEM, "Replacement system."),
        Message(MessageRole.USER, "Retained user message"),
    ]

    conversation.replace_history(replacement)
    replacement.append(Message(MessageRole.ASSISTANT, "Later mutation"))

    assert conversation.messages() == tuple(replacement[:2])


def test_replace_history_rejects_invalid_history_without_mutation() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_user("Keep this")
    original = conversation.messages()
    invalid_histories = (
        None,
        [],
        [Message(MessageRole.USER, "No system")],
        [Message(MessageRole.SYSTEM, "One"), Message(MessageRole.SYSTEM, "Two")],
        [Message(MessageRole.USER, "Wrong first"), Message(MessageRole.SYSTEM, "Late")],
    )

    for invalid in invalid_histories:
        with pytest.raises(ValueError):
            conversation.replace_history(invalid)
        assert conversation.messages() == original


def test_callbacks_receive_appended_and_replaced_messages_outside_lock() -> None:
    appended: list[Message] = []
    replaced: list[list[Message]] = []
    callbacks_observed_unlocked_state: list[bool] = []
    conversation: Conversation

    def observe_lock() -> None:
        completed = threading.Event()
        thread = threading.Thread(target=lambda: (conversation.messages(), completed.set()))
        thread.start()
        callbacks_observed_unlocked_state.append(completed.wait(timeout=1))
        thread.join(timeout=1)

    def on_append(message: Message) -> None:
        observe_lock()
        appended.append(message)

    def on_replace(messages: list[Message]) -> None:
        observe_lock()
        replaced.append(messages)

    conversation = Conversation(SYSTEM_PROMPT, on_append=on_append, on_replace=on_replace)
    user = conversation.add_user("hello")
    assistant = conversation.add_assistant("hi")
    replacement = [Message(MessageRole.SYSTEM, "new system"), user, assistant]
    conversation.replace_history(replacement)

    assert appended == [user, assistant]
    assert replaced == [replacement]
    assert all(callbacks_observed_unlocked_state)


def test_all_append_methods_trigger_callback_once() -> None:
    appended: list[Message] = []
    conversation = Conversation(SYSTEM_PROMPT, on_append=appended.append)
    call = ToolCall("call-1", "read_file")
    result = ToolResult("call-1", "read_file", "ok")

    conversation.add_user("read")
    conversation.add_assistant_with_tool_calls("", (call,))
    conversation.add_tool_results((result,))
    conversation.add_assistant("done")

    assert appended == list(conversation.messages()[1:])


def test_from_messages_restores_non_system_history_without_callbacks() -> None:
    restored = [
        Message(MessageRole.USER, "remember this"),
        Message(MessageRole.ASSISTANT, "remembered"),
    ]
    appended: list[Message] = []

    conversation = Conversation.from_messages(
        SYSTEM_PROMPT,
        restored,
        on_append=appended.append,
    )
    restored.append(Message(MessageRole.USER, "later mutation"))

    assert conversation.messages() == (
        Message(MessageRole.SYSTEM, SYSTEM_PROMPT),
        Message(MessageRole.USER, "remember this"),
        Message(MessageRole.ASSISTANT, "remembered"),
    )
    assert appended == []


def test_from_messages_rejects_system_and_invalid_values() -> None:
    with pytest.raises(ValueError, match="must not contain a system"):
        Conversation.from_messages(SYSTEM_PROMPT, [Message(MessageRole.SYSTEM, "old")])
    with pytest.raises(ValueError, match="only Message"):
        Conversation.from_messages(SYSTEM_PROMPT, ["invalid"])  # type: ignore[list-item]


def test_replace_system_prompt_does_not_trigger_persistence_callbacks() -> None:
    appended: list[Message] = []
    replaced: list[list[Message]] = []
    conversation = Conversation(
        SYSTEM_PROMPT,
        on_append=appended.append,
        on_replace=replaced.append,
    )
    conversation.add_user("hello")

    conversation.replace_system_prompt("Updated system prompt")

    assert conversation.messages()[0] == Message(MessageRole.SYSTEM, "Updated system prompt")
    assert appended == [Message(MessageRole.USER, "hello")]
    assert replaced == []


def test_replace_system_prompt_validates_before_mutation() -> None:
    conversation = Conversation(SYSTEM_PROMPT)
    original = conversation.messages()

    with pytest.raises(ValueError, match="content must not be empty"):
        conversation.replace_system_prompt("  ")

    assert conversation.messages() == original
