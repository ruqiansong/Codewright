"""Cross-module acceptance tests for background subagents and TUI routing."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from codewright.agent import Agent, ApprovalRequest, CompletionResult, Event
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
from codewright.permission import Engine, Mode, Outcome
from codewright.permission.rule import RuleSet
from codewright.subagent import Catalog
from codewright.task import (
    BackgroundTask,
    Manager,
    SendMessageTool,
    Status,
    SubagentApprovalBroker,
    TaskGetTool,
    TaskListTool,
    TaskStopTool,
)
from codewright.tui import ChatScreen, ChatState, CodewrightApp
from codewright.tui.widgets.approval import ApprovalWidget


class TurnProvider:
    """Pause only the first turn and record request-only reminders."""

    provider_name = "turns"
    model_name = "turns"

    def __init__(self) -> None:
        self.requests: list[tuple[Message, ...]] = []
        self.contexts: list[RequestContext | None] = []
        self.first_delta = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

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
        del parameters, tools
        index = len(self.requests)
        self.requests.append(tuple(messages))
        self.contexts.append(request_context)
        yield StreamEvent.delta(f"reply-{index + 1}")
        if index == 0:
            self.first_delta.set()
            await self.release.wait()
        yield StreamEvent.completed()

    async def close(self) -> None:
        self.closed = True


class MiniAgent(Agent):
    """Deterministic completion Agent for task lifecycle acceptance."""

    def __init__(self, outcomes: list[CompletionResult | asyncio.Event]) -> None:
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
            if isinstance(outcome, asyncio.Event):
                await outcome.wait()
                return CompletionResult("released", TokenUsage(0, 0, 0))
            return outcome
        finally:
            if event_sink is not None:
                event_sink.put_nowait(None)


class OwnedProvider:
    provider_name = "owned"
    model_name = "owned"

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


def _engine(root: Path) -> Engine:
    return Engine(
        root=root.resolve(),
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=root / ".codewright/settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


def _conversation() -> Conversation:
    value = Conversation("system")
    value.add_user("initial")
    return value


async def _terminal(task: BackgroundTask, status: Status) -> None:
    for _ in range(200):
        if task.status is status:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"task did not reach {status.value}")


async def _idle(screen: ChatScreen) -> None:
    for _ in range(200):
        if screen.state is ChatState.IDLE:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("chat screen did not become idle")


@pytest.mark.asyncio
async def test_busy_task_notification_reaches_next_turn_once(tmp_path: Path) -> None:
    provider = TurnProvider()
    manager = Manager()
    broker = SubagentApprovalBroker()
    app = CodewrightApp(
        provider,
        Conversation("main system"),
        engine=_engine(tmp_path),
        working_directory=tmp_path,
        subagent_catalog=Catalog(),
        task_manager=manager,
        approval_broker=broker,
    )

    async with app.run_test() as pilot:
        await pilot.press(*"first", "enter")
        await asyncio.wait_for(provider.first_delta.wait(), timeout=1)
        screen = app.screen
        assert isinstance(screen, ChatScreen)
        assert screen.state is ChatState.STREAMING
        completed = BackgroundTask(
            id="task-1",
            name="worker",
            description="inspect",
            sub_agent=app.main_agent,
            conversation=Conversation("child"),
            initial_prompt="inspect",
            status=Status.COMPLETED,
            result="three files",
            notification_generation=1,
        )
        done_queue = manager.subscribe_done()
        done_queue.put_nowait(completed)
        for _ in range(100):
            if done_queue.empty():
                break
            await asyncio.sleep(0.01)
        assert app.runtime.take_reminders() == []

        provider.release.set()
        await _idle(screen)
        await pilot.press(*"second", "enter")
        await _idle(screen)
        await pilot.press(*"third", "enter")
        await _idle(screen)

        reminders = [
            context.reminder if context is not None else "" for context in provider.contexts
        ]
        assert "<task-notification>" not in reminders[0]
        assert reminders[1].count("<task-notification>") == 1
        assert '"task_id":"task-1"' in reminders[1]
        assert "<task-notification>" not in reminders[2]


@pytest.mark.asyncio
async def test_task_tools_complete_continue_and_stop_full_chain() -> None:
    manager = Manager()
    agent = MiniAgent(
        [
            CompletionResult("first", TokenUsage(1, 2, 3)),
            CompletionResult("second", TokenUsage(2, 3, 5)),
        ]
    )
    task = await manager.launch(
        agent,
        _conversation(),
        "initial",
        "worker",
        name="worker",
    )
    await _terminal(task, Status.COMPLETED)
    first_done = await manager.subscribe_done().get()
    assert first_done is task

    listed = json.loads((await TaskListTool(manager).execute("{}")).content)
    assert listed["tasks"] == [
        {
            "task_id": task.id,
            "name": "worker",
            "description": "worker",
            "status": "completed",
            "tool_count": 0,
            "last_activity": "",
        }
    ]
    detail = json.loads(
        (await TaskGetTool(manager).execute(json.dumps({"task_id": task.id}))).content
    )
    assert detail["result"] == "first"
    assert detail["notification_generation"] == 1

    resumed = json.loads(
        (
            await SendMessageTool(manager).execute(
                json.dumps({"name": "worker", "message": "follow up"})
            )
        ).content
    )
    assert resumed == {"task_id": task.id, "status": "running"}
    await _terminal(task, Status.COMPLETED)
    assert await manager.subscribe_done().get() is task
    assert task.notification_generation == 2
    assert task.result == "second"
    assert agent.calls == ["", "follow up"]

    blocker = asyncio.Event()
    running = await manager.launch(MiniAgent([blocker]), _conversation(), "block", "blocking")
    stopped = await TaskStopTool(manager).execute(json.dumps({"task_id": running.id}))
    assert json.loads(stopped.content)["status"] == "cancellation_requested"
    assert running.status is Status.CANCELLED
    await manager.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [Outcome.ALLOW_ONCE, Outcome.DENY_ONCE])
async def test_subagent_approval_routes_through_tui(
    tmp_path: Path,
    outcome: Outcome,
) -> None:
    provider = TurnProvider()
    manager = Manager()
    broker = SubagentApprovalBroker()
    app = CodewrightApp(
        provider,
        Conversation("main system"),
        engine=_engine(tmp_path),
        working_directory=tmp_path,
        subagent_catalog=Catalog(),
        task_manager=manager,
        approval_broker=broker,
    )

    async with app.run_test():
        screen = app.screen
        assert isinstance(screen, ChatScreen)
        response = asyncio.get_running_loop().create_future()
        request = ApprovalRequest(
            "approval-1",
            "bash",
            '{"command":"echo ok"}',
            "Execution requires approval.",
            response,
        )
        routed = asyncio.create_task(broker.request("explore", request))
        for _ in range(200):
            if screen.pending_approval is request:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("approval was not presented")
        widget = screen.query_one(ApprovalWidget)
        assert "来自 SubAgent explore" in widget.render().plain

        await screen.submit_approval(ApprovalWidget.Selected(outcome))
        assert await asyncio.wait_for(routed, timeout=1) is outcome
        await _idle(screen)


@pytest.mark.asyncio
async def test_app_shutdown_closes_running_task_owned_provider_and_approval(
    tmp_path: Path,
) -> None:
    provider = TurnProvider()
    owned = OwnedProvider()
    manager = Manager()
    broker = SubagentApprovalBroker()
    blocker = asyncio.Event()
    task = await manager.launch(
        MiniAgent([blocker]),
        _conversation(),
        "block",
        "blocking",
        owned_provider=owned,
    )
    app = CodewrightApp(
        provider,
        Conversation("main system"),
        engine=_engine(tmp_path),
        working_directory=tmp_path,
        subagent_catalog=Catalog(),
        task_manager=manager,
        approval_broker=broker,
    )
    response = asyncio.get_running_loop().create_future()
    request = ApprovalRequest(
        "approval-close",
        "bash",
        '{"command":"echo waiting"}',
        "Execution requires approval.",
        response,
    )

    async with app.run_test():
        screen = app.screen
        assert isinstance(screen, ChatScreen)
        routed = asyncio.create_task(broker.request("worker", request))
        for _ in range(200):
            if screen.pending_approval is request:
                break
            await asyncio.sleep(0.01)

    await _terminal(task, Status.CANCELLED)
    assert owned.closed == 1
    assert request.respond.done()
    await asyncio.gather(routed, return_exceptions=True)
