"""Tests for model-facing background task tools."""

from __future__ import annotations

import asyncio
import json

import pytest

from codewright.agent import Agent, CompletionResult, Event
from codewright.conversation import Conversation
from codewright.llm import TokenUsage
from codewright.task import (
    Manager,
    SendMessageTool,
    TaskGetTool,
    TaskListTool,
    TaskStopTool,
)
from codewright.tool import Tool


class MiniAgent(Agent):
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


def _conversation() -> Conversation:
    value = Conversation("system")
    value.add_user("initial")
    return value


async def _wait_completed(task) -> None:
    for _ in range(20):
        if task.status.value == "completed":
            return
        await asyncio.sleep(0)
    raise AssertionError("task did not complete")


def test_task_tools_satisfy_protocol_and_have_correct_safety_flags() -> None:
    manager = Manager()
    tools = (
        TaskListTool(manager),
        TaskGetTool(manager),
        TaskStopTool(manager),
        SendMessageTool(manager),
    )

    assert all(isinstance(tool, Tool) for tool in tools)
    assert [tool.name for tool in tools] == ["TaskList", "TaskGet", "TaskStop", "SendMessage"]
    assert [tool.read_only for tool in tools] == [True, True, False, False]
    assert all(tool.parameters["type"] == "object" for tool in tools)
    assert all(not hasattr(tool, "is_system") for tool in tools)


@pytest.mark.asyncio
async def test_list_get_and_send_message_return_safe_json() -> None:
    manager = Manager()
    agent = MiniAgent(
        [
            CompletionResult("first", TokenUsage(1, 2, 3)),
            CompletionResult("second", TokenUsage(2, 3, 5)),
        ]
    )
    task = await manager.launch(
        agent, _conversation(), "initial", "worker description", name="worker"
    )
    await _wait_completed(task)

    listed = await TaskListTool(manager).execute("{}")
    listing = json.loads(listed.content)
    assert listing["tasks"][0]["task_id"] == task.id
    assert listing["tasks"][0]["status"] == "completed"

    fetched = await TaskGetTool(manager).execute(json.dumps({"task_id": task.id}))
    detail = json.loads(fetched.content)
    assert detail["result"] == "first"
    assert detail["usage"]["total_tokens"] == 3
    assert "sub_agent" not in detail
    assert "owned_provider" not in detail

    sent = await SendMessageTool(manager).execute(
        json.dumps({"name": "worker", "message": "follow up"})
    )
    assert json.loads(sent.content) == {"status": "running", "task_id": task.id}
    await _wait_completed(task)
    assert agent.calls == ["", "follow up"]
    await manager.aclose()


@pytest.mark.asyncio
async def test_stop_tool_and_terminal_errors_are_stable() -> None:
    manager = Manager()
    blocker = asyncio.Event()
    task = await manager.launch(MiniAgent([blocker]), _conversation(), "initial", "blocking")

    stopped = await TaskStopTool(manager).execute(json.dumps({"task_id": task.id}))
    assert json.loads(stopped.content)["status"] == "cancellation_requested"
    stopped_again = await TaskStopTool(manager).execute(json.dumps({"task_id": task.id}))
    assert stopped_again.error_code == "task_not_running"
    missing = await TaskGetTool(manager).execute('{"task_id":"missing"}')
    assert missing.error_code == "unknown_task"
    await manager.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("list", "not-json"),
        ("list", '{"extra":true}'),
        ("get", "[]"),
        ("get", '{"task_id":" "}'),
        ("stop", "{}"),
        ("send", '{"name":"worker"}'),
        ("send", '{"name":" worker","message":"x"}'),
    ],
)
async def test_task_tools_reject_invalid_arguments(tool_name: str, arguments: str) -> None:
    manager = Manager()
    tools = {
        "list": TaskListTool(manager),
        "get": TaskGetTool(manager),
        "stop": TaskStopTool(manager),
        "send": SendMessageTool(manager),
    }

    result = await tools[tool_name].execute(arguments)

    assert result.is_error
    assert result.error_code == "invalid_arguments"
    await manager.aclose()
