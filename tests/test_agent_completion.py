"""Tests for consuming the existing Agent event stream to completion."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from codewright.agent import Agent, Event, MaxTurnsReached
from codewright.agent.completion import consume_events
from codewright.conversation import Conversation
from codewright.llm import (
    ChatResult,
    LLMResponseError,
    Message,
    RequestContext,
    RequestParameters,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from codewright.permission import Engine, Mode
from codewright.permission.rule import RuleSet
from codewright.tool import Registry


class Provider:
    provider_name = "test"
    model_name = "test"

    def __init__(self, replies: Sequence[Sequence[StreamEvent]]) -> None:
        self.replies = tuple(replies)
        self.requests: list[tuple[Message, ...]] = []

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del messages, parameters, tools, request_context
        raise AssertionError("streaming expected")

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters, tools, request_context
        index = len(self.requests)
        self.requests.append(tuple(messages))
        for event in self.replies[index]:
            yield event


def _engine(tmp_path: Path) -> Engine:
    return Engine(
        root=tmp_path,
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=tmp_path / ".codewright/settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


def _conversation() -> Conversation:
    return Conversation("system")


@pytest.mark.asyncio
async def test_completion_appends_task_once_and_accumulates_text_and_usage(
    tmp_path: Path,
) -> None:
    provider = Provider(
        [
            (
                StreamEvent.delta("hello "),
                StreamEvent.usage_report(TokenUsage(2, 3, 5, 1, 0)),
                StreamEvent.delta("world"),
                StreamEvent.usage_report(TokenUsage(4, 5, 9, 0, 2)),
                StreamEvent.completed(),
            )
        ]
    )
    conversation = _conversation()

    result = await Agent(provider, Registry(), _engine(tmp_path)).run_to_completion(
        conversation, "do it"
    )

    assert result.text == "hello world"
    assert result.usage == TokenUsage(6, 8, 14, 1, 2)
    assert [message.content for message in conversation.messages()].count("do it") == 1


@pytest.mark.asyncio
async def test_completion_empty_task_does_not_append_and_sink_gets_sentinel(
    tmp_path: Path,
) -> None:
    provider = Provider([(StreamEvent.delta("done"), StreamEvent.completed())])
    conversation = _conversation()
    conversation.add_user("already present")
    sink: asyncio.Queue = asyncio.Queue()

    result = await Agent(provider, Registry(), _engine(tmp_path)).run_to_completion(
        conversation, event_sink=sink
    )

    assert result.text == "done"
    assert [message.content for message in conversation.messages()].count("already present") == 1
    forwarded = [sink.get_nowait() for _ in range(sink.qsize())]
    assert forwarded[-1] is None


@pytest.mark.asyncio
async def test_bounded_event_sink_never_blocks_completion(tmp_path: Path) -> None:
    provider = Provider([(StreamEvent.delta("done"), StreamEvent.completed())])
    sink: asyncio.Queue = asyncio.Queue(maxsize=1)

    result = await asyncio.wait_for(
        Agent(provider, Registry(), _engine(tmp_path)).run_to_completion(
            _conversation(), "task", event_sink=sink
        ),
        timeout=1,
    )

    assert result.text == "done"
    assert sink.qsize() == 1


@pytest.mark.asyncio
async def test_child_max_turns_raises_with_last_text(tmp_path: Path) -> None:
    provider = Provider(
        [
            (
                StreamEvent.delta("working"),
                StreamEvent.tool_calls_ready((ToolCall("missing", "missing", "{}"),)),
                StreamEvent.completed(),
            )
        ]
    )
    agent = Agent(
        provider,
        Registry(),
        _engine(tmp_path),
        max_turns=1,
        subagent_kind="defined",
    )

    with pytest.raises(MaxTurnsReached) as captured:
        await agent.run_to_completion(_conversation(), "task")

    assert captured.value.last_text == "working"


@pytest.mark.asyncio
async def test_completion_propagates_agent_error_and_cancellation() -> None:
    async def failed():
        yield Event.failed(LLMResponseError("safe failure"))

    with pytest.raises(LLMResponseError):
        await consume_events(failed())

    async def cancelled():
        raise asyncio.CancelledError
        yield Event.completed()

    with pytest.raises(asyncio.CancelledError):
        await consume_events(cancelled())
