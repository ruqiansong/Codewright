"""Tests for background task lifecycle, adoption, continuation, and ownership."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

import pytest

from codewright.agent import Agent, CompletionResult, Event, Phase, ToolEvent
from codewright.conversation import Conversation
from codewright.llm import (
    ChatResult,
    Message,
    RequestContext,
    RequestParameters,
    StreamEvent,
    TokenUsage,
    ToolDefinition,
)
from codewright.task import Manager, ManagerError, Status


class NoopProvider:
    provider_name = "test"
    model_name = "test"

    def __init__(self) -> None:
        self.closed = 0

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del messages, parameters, tools, request_context
        raise AssertionError("not used")

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del messages, parameters, tools, request_context
        yield StreamEvent.completed()

    async def close(self) -> None:
        self.closed += 1


class StubAgent(Agent):
    def __init__(self, outcomes: list[CompletionResult | Exception | asyncio.Event]) -> None:
        self.outcomes = outcomes
        self.calls: list[str] = []

    async def run_to_completion(
        self,
        conversation: Conversation,
        task: str = "",
        *,
        stream: bool = True,
        cancel_event: asyncio.Event | None = None,
        event_sink: asyncio.Queue[Event | None] | None = None,
    ) -> CompletionResult:
        del stream, cancel_event
        self.calls.append(task)
        if task:
            conversation.add_user(task)
        outcome = self.outcomes.pop(0)
        try:
            if event_sink is not None:
                event_sink.put_nowait(
                    Event.tool_event(ToolEvent("call", "read_file", "{}", Phase.END, summary="ok"))
                )
                event_sink.put_nowait(Event.delta("activity"))
            if isinstance(outcome, asyncio.Event):
                await outcome.wait()
                return CompletionResult("released", TokenUsage(0, 0, 0))
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        finally:
            if event_sink is not None:
                event_sink.put_nowait(None)


def _conversation(prompt: str = "initial") -> Conversation:
    value = Conversation("system")
    value.add_user(prompt)
    return value


async def _settled(task, status: Status) -> None:
    for _ in range(20):
        if task.status is status:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"task did not reach {status}")


@pytest.mark.asyncio
async def test_launch_completes_aggregates_and_notifies_once() -> None:
    manager = Manager()
    result = CompletionResult("result", TokenUsage(2, 3, 5))
    agent = StubAgent([result])

    task = await manager.launch(agent, _conversation(), "initial", "description", name="one")
    await _settled(task, Status.COMPLETED)

    assert task.result == "result"
    assert task.usage == TokenUsage(2, 3, 5)
    assert task.tool_count == 1
    assert task.last_activity == "activity"
    assert agent.calls == [""]
    assert await manager.subscribe_done().get() is task
    assert manager.subscribe_done().empty()
    await manager.aclose()


@pytest.mark.asyncio
async def test_completion_listeners_broadcast_without_consuming_done_queue() -> None:
    manager = Manager()
    observed = []

    async def listener(task) -> None:
        observed.append(task.id)

    def broken_listener(task) -> None:
        del task
        raise RuntimeError("listener failure")

    manager.add_completion_listener(listener)
    manager.add_completion_listener(broken_listener)
    task = await manager.launch(
        StubAgent([CompletionResult("done", TokenUsage(0, 0, 0))]),
        _conversation(),
        "initial",
        "description",
    )
    await _settled(task, Status.COMPLETED)
    for _ in range(10):
        if observed:
            break
        await asyncio.sleep(0)

    assert observed == [task.id]
    assert await manager.subscribe_done().get() is task
    manager.remove_completion_listener(listener)
    await manager.aclose()


@pytest.mark.asyncio
async def test_failure_and_stop_have_safe_terminal_states() -> None:
    manager = Manager()
    failed = await manager.launch(
        StubAgent([RuntimeError("secret detail")]),
        _conversation(),
        "initial",
        "failure",
    )
    await _settled(failed, Status.FAILED)
    assert failed.error_type == "RuntimeError"
    assert failed.error_message == "Background task failed."
    assert "secret" not in failed.error_message

    blocker = asyncio.Event()
    running = await manager.launch(StubAgent([blocker]), _conversation(), "initial", "running")
    stopped = await manager.stop(running.id)
    assert stopped.status is Status.CANCELLED
    with pytest.raises(ManagerError) as terminal:
        await manager.stop(running.id)
    assert terminal.value.code == "task_not_running"
    with pytest.raises(ManagerError) as unknown:
        await manager.stop("missing")
    assert unknown.value.code == "unknown_task"
    await manager.aclose()


@pytest.mark.asyncio
async def test_adopt_observes_same_handle_without_rerunning_agent() -> None:
    manager = Manager()
    agent = StubAgent([])

    async def existing() -> CompletionResult:
        await asyncio.sleep(0)
        return CompletionResult("adopted", TokenUsage(1, 1, 2))

    handle = asyncio.create_task(existing())
    task = await manager.adopt_running(
        agent,
        _conversation(),
        "initial",
        "adopted",
        handle,
    )
    await _settled(task, Status.COMPLETED)

    assert task.handle is handle
    assert task.result == "adopted"
    assert agent.calls == []
    await manager.aclose()


@pytest.mark.asyncio
async def test_later_named_task_wins_and_send_message_reuses_id_once() -> None:
    manager = Manager()
    first = await manager.launch(
        StubAgent([CompletionResult("first", TokenUsage(0, 0, 0))]),
        _conversation("first"),
        "first",
        "first",
        name="worker",
    )
    second_agent = StubAgent(
        [
            CompletionResult("second", TokenUsage(0, 0, 0)),
            CompletionResult("continued", TokenUsage(1, 2, 3)),
        ]
    )
    conversation = _conversation("second")
    second = await manager.launch(
        second_agent,
        conversation,
        "second",
        "second",
        name="worker",
    )
    await _settled(first, Status.COMPLETED)
    await _settled(second, Status.COMPLETED)
    first_notification = await manager.subscribe_done().get()
    second_notification = await manager.subscribe_done().get()
    assert {first_notification.id, second_notification.id} == {first.id, second.id}

    continued = await manager.send_message("worker", "follow up")
    assert continued.id == second.id
    await _settled(continued, Status.COMPLETED)
    assert continued.result == "continued"
    assert second_agent.calls == ["", "follow up"]
    assert [item.content for item in conversation.messages()].count("follow up") == 1
    assert await manager.subscribe_done().get() is second
    assert second.notification_generation == 2
    await manager.aclose()


@pytest.mark.asyncio
async def test_aclose_is_idempotent_and_closes_owned_provider_once() -> None:
    manager = Manager(done_queue_size=1)
    provider = NoopProvider()
    blocker = asyncio.Event()
    task = await manager.launch(
        StubAgent([blocker]),
        _conversation(),
        "initial",
        "running",
        owned_provider=provider,
    )

    await manager.aclose()
    await manager.aclose()

    assert task.status is Status.CANCELLED
    assert provider.closed == 1
    assert await manager.subscribe_done().get() is None
