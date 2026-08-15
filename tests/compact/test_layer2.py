"""Tests for recent-history selection and LLM-backed summarization."""

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from codewright.compact.compact import ManageInput
from codewright.compact.layer2 import (
    _join_after_summary,
    auto_compact,
    force_compact,
    group_by_user_turn,
    pick_recent_tail,
    run_summary,
    summarize_once,
)
from codewright.compact.state import (
    CompactCircuitBreaker,
    ContentReplacementState,
    RecoveryState,
    SessionContext,
)
from codewright.conversation import Conversation
from codewright.llm import (
    ChatResult,
    Message,
    MessageRole,
    PromptTooLongError,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class SummaryProvider:
    """Deterministic summary stream used by compact unit tests."""

    def __init__(self, scripts: Sequence[Sequence[StreamEvent]]) -> None:
        self.scripts = list(scripts)
        self.calls: list[tuple[Message, ...]] = []
        self.tool_arguments: list[tuple[ToolDefinition, ...]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-summary"

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append(tuple(messages))
        self.tool_arguments.append(tuple(tools))
        for event in self.scripts.pop(0):
            yield event


def compact_input(
    tmp_path: Path,
    provider: SummaryProvider,
    messages: Sequence[Message],
    *,
    estimated_token: int = 1_000,
    context_window: int = 200_000,
    usage_anchor: int = 0,
) -> ManageInput:
    conversation = Conversation("system")
    conversation.replace_history(messages)
    return ManageInput(
        conv=conversation,
        provider=provider,
        context_window=context_window,
        tool_defs=(),
        replacement=ContentReplacementState(),
        recovery=RecoveryState(),
        auto_tracking=CompactCircuitBreaker(),
        session=SessionContext("test", str(tmp_path / "spill")),
        usage_anchor=usage_anchor,
        anchor_msg_len=len(messages) if usage_anchor else 0,
        estimated_token=estimated_token,
    )


def summary_events(text: str = "condensed") -> list[StreamEvent]:
    return [StreamEvent.delta(f"<summary>{text}</summary>"), StreamEvent.completed()]


def user(content: str) -> Message:
    return Message(MessageRole.USER, content)


def assistant(content: str) -> Message:
    return Message(MessageRole.ASSISTANT, content)


def test_pick_recent_tail_excludes_system_and_exhausts_short_history() -> None:
    messages = [Message(MessageRole.SYSTEM, "system"), user("u"), assistant("a")]

    assert pick_recent_tail(messages) == messages[1:]


def test_pick_recent_tail_requires_both_size_and_count_bounds() -> None:
    messages = [Message(MessageRole.SYSTEM, "system")]
    messages.extend(user("x" * 9_000) for _ in range(6))

    selected = pick_recent_tail(messages)

    assert len(selected) == 5


def test_pick_recent_tail_keeps_tool_call_and_result_together() -> None:
    call = ToolCall("call-1", "read_file")
    result = ToolResult("call-1", "read_file", "x" * 40_000)
    messages = [
        Message(MessageRole.SYSTEM, "system"),
        user("old"),
        Message(MessageRole.ASSISTANT, "", tool_calls=(call,)),
        Message(MessageRole.TOOL, "", tool_results=(result,)),
        assistant("done"),
        user("later"),
        assistant("later answer"),
        user("newest"),
    ]

    selected = pick_recent_tail(messages)

    assert selected[0].role is MessageRole.ASSISTANT
    assert selected[0].tool_calls == (call,)
    assert selected[1].tool_results == (result,)


def test_join_after_summary_inserts_bridge_before_user() -> None:
    summary = user("summary")

    joined = _join_after_summary(summary, [user("recent"), assistant("answer")])

    assert [message.role for message in joined] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


def test_join_after_summary_defensively_drops_leading_tool() -> None:
    result = ToolResult("call-1", "read_file", "content")
    tool = Message(MessageRole.TOOL, "", tool_results=(result,))

    joined = _join_after_summary(user("summary"), [tool, assistant("after")])

    assert joined == [user("summary"), assistant("after")]


def test_group_by_user_turn_keeps_complete_exchanges_and_skips_system() -> None:
    messages = [
        Message(MessageRole.SYSTEM, "system"),
        user("one"),
        assistant("one answer"),
        user("two"),
        assistant("two answer"),
    ]

    groups = group_by_user_turn(messages)

    assert groups == [messages[1:3], messages[3:5]]


def test_group_by_user_turn_attaches_leading_non_user_messages() -> None:
    messages = [assistant("preamble"), user("one"), assistant("answer")]

    assert group_by_user_turn(messages) == [messages]


async def test_summarize_once_uses_no_tools_and_passes_through_ptl(tmp_path: Path) -> None:
    error = PromptTooLongError()
    provider = SummaryProvider([[StreamEvent.failed(error)]])
    messages = [Message(MessageRole.SYSTEM, "system"), user("hello")]
    in_ = compact_input(tmp_path, provider, messages)

    with pytest.raises(PromptTooLongError) as raised:
        await summarize_once(in_, messages)

    assert raised.value is error
    assert provider.tool_arguments == [()]


async def test_run_summary_retries_ptl_by_dropping_oldest_groups(tmp_path: Path) -> None:
    ptl = [StreamEvent.failed(PromptTooLongError())]
    provider = SummaryProvider([ptl, ptl, ptl, summary_events()])
    messages = [Message(MessageRole.SYSTEM, "system")]
    for number in range(5):
        messages.extend((user(f"turn-{number}"), assistant(f"answer-{number}")))
    in_ = compact_input(tmp_path, provider, messages)

    rewritten = await run_summary(in_)

    serialized_calls = [call[0].content for call in provider.calls]
    retained_group_counts = [
        sum(f"turn-{number}" in prompt for number in range(5)) for prompt in serialized_calls
    ]
    assert retained_group_counts == [5, 4, 3, 2]
    assert rewritten[0] is messages[0]
    assert "## 历史会话摘要\ncondensed" in rewritten[1].content
    assert all(
        not (left.role is MessageRole.USER and right.role is MessageRole.USER)
        for left, right in zip(rewritten, rewritten[1:], strict=False)
    )


async def test_auto_and_force_compact_update_tracking_differently(tmp_path: Path) -> None:
    messages = [Message(MessageRole.SYSTEM, "system"), user("hello"), assistant("answer")]
    automatic = compact_input(
        tmp_path,
        SummaryProvider([summary_events()]),
        messages,
        estimated_token=10_000,
    )
    automatic.auto_tracking.record_failure()

    _, before, after = await auto_compact(automatic)

    assert before == 10_000
    assert after > 0
    assert automatic.auto_tracking.tripped() is False

    forced = compact_input(
        tmp_path,
        SummaryProvider([[StreamEvent.failed(PromptTooLongError())]]),
        messages,
    )
    with pytest.raises(PromptTooLongError):
        await force_compact(forced)
    assert forced.auto_tracking.tripped() is False
