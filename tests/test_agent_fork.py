"""Tests for isolated fork conversation construction."""

from codewright.agent.fork import (
    FORK_BOILERPLATE,
    build_fork_conversation,
    build_fork_conversation_from_messages,
    is_fork_context,
)
from codewright.conversation import Conversation
from codewright.llm import MessageRole, ToolCall, ToolResult


def test_fork_preserves_system_copies_history_and_adds_task_once() -> None:
    parent = Conversation("parent system")
    parent.add_user("original")
    parent.add_assistant("answer")

    fork = build_fork_conversation(parent, "new task")

    assert fork.messages()[0].content == "parent system"
    assert fork.messages()[1:] == parent.messages()[1:] + (fork.messages()[-1],)
    assert fork.messages()[1] is not parent.messages()[1]
    assert fork.messages()[-1].content == FORK_BOILERPLATE + "new task"
    assert sum(FORK_BOILERPLATE in item.content for item in fork.messages()) == 1
    assert is_fork_context(fork)
    assert not is_fork_context(parent)


def test_fork_repairs_every_unmatched_tool_call() -> None:
    parent = Conversation("system")
    parent.add_user("work")
    parent.add_assistant_with_tool_calls(
        "",
        (ToolCall("one", "read_file"), ToolCall("two", "glob")),
    )
    parent.add_tool_results((ToolResult("one", "read_file", "ok"),))

    fork = build_fork_conversation(parent, "continue")

    repair = fork.messages()[-2]
    assert repair.role is MessageRole.TOOL
    assert [result.tool_call_id for result in repair.tool_results] == ["two"]
    assert repair.tool_results[0].error_code == "fork_missing_tool_result"


def test_explicit_builder_rejects_system_in_history() -> None:
    parent = Conversation("system")

    try:
        build_fork_conversation_from_messages("system", parent.messages(), "task")
    except ValueError as error:
        assert "system" in str(error)
    else:
        raise AssertionError("system history must be rejected")
