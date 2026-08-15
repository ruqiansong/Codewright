"""Tests for child Agent permission overrides and approval upgrades."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codewright.agent import Agent, ApprovalRequest, Event
from codewright.conversation import Conversation
from codewright.llm import (
    ChatResult,
    Message,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)
from codewright.permission import Decision, Engine, Mode, Outcome
from codewright.permission.matcher import ExactMatcher
from codewright.permission.rule import Rule, RuleSet
from codewright.tool import Registry, Result


class Provider:
    provider_name = "test"
    model_name = "test"

    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self.count = 0

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
        del messages, parameters, tools, request_context
        self.count += 1
        if self.count == 1:
            yield StreamEvent.tool_calls_ready((self.call,))
        else:
            yield StreamEvent.delta("done")
        yield StreamEvent.completed()


@dataclass(slots=True)
class WriteTool:
    name: str = "write_file"
    description: str = "write"
    parameters: Mapping[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only: bool = False
    calls: list[str] = field(default_factory=list)

    async def execute(self, arguments_json: str) -> Result:
        self.calls.append(arguments_json)
        return Result("written")


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
    value = Conversation("system")
    value.add_user("task")
    return value


def _setup(tmp_path: Path) -> tuple[Provider, Registry, WriteTool, ToolCall, Engine]:
    call = ToolCall("write", "write_file", '{"path":"generated.txt","content":"x"}')
    provider = Provider(call)
    tool = WriteTool()
    registry = Registry()
    registry.register(tool)
    return provider, registry, tool, call, _engine(tmp_path)


@pytest.mark.asyncio
async def test_dont_ask_promotes_only_ask(tmp_path: Path) -> None:
    provider, registry, tool, _, engine = _setup(tmp_path)
    agent = Agent(provider, registry, engine, dont_ask=True, subagent_kind="defined")

    events = [event async for event in agent.run(_conversation())]

    assert len(tool.calls) == 1
    assert not any(event.approval for event in events)


@pytest.mark.asyncio
async def test_dont_ask_cannot_override_explicit_deny(tmp_path: Path) -> None:
    provider, registry, tool, _, engine = _setup(tmp_path)
    engine.local.deny.append(Rule("Write", ExactMatcher("generated.txt"), False, "=generated.txt"))
    agent = Agent(provider, registry, engine, dont_ask=True, subagent_kind="defined")
    conversation = _conversation()

    _ = [event async for event in agent.run(conversation)]

    assert tool.calls == []
    assert conversation.messages()[3].tool_results[0].error_code == "permission_denied"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "execution_count"),
    [(Outcome.ALLOW_ONCE, 1), (Outcome.DENY_ONCE, 0), (Outcome.ALLOW_FOREVER, 1)],
)
async def test_approval_upgrader_controls_execution_and_persistence(
    tmp_path: Path,
    outcome: Outcome,
    execution_count: int,
) -> None:
    provider, registry, tool, call, engine = _setup(tmp_path)
    requests: list[ApprovalRequest] = []

    async def upgrade(request: ApprovalRequest) -> Outcome:
        requests.append(request)
        return outcome

    agent = Agent(
        provider,
        registry,
        engine,
        approval_upgrader=upgrade,
        subagent_kind="defined",
    )
    events = [event async for event in agent.run(_conversation())]

    assert len(requests) == 1
    assert len(tool.calls) == execution_count
    assert not any(event.approval for event in events)
    if outcome is Outcome.ALLOW_FOREVER:
        assert engine.check(Mode.DEFAULT, call, False)[0] is Decision.ALLOW


@pytest.mark.asyncio
async def test_permission_mode_overrides_run_argument(tmp_path: Path) -> None:
    provider, registry, tool, _, engine = _setup(tmp_path)
    agent = Agent(
        provider,
        registry,
        engine,
        permission_mode=Mode.ACCEPT_EDITS,
        subagent_kind="defined",
    )

    events = [event async for event in agent.run(_conversation(), mode=Mode.DEFAULT)]

    assert len(tool.calls) == 1
    assert not any(event.approval for event in events)


@pytest.mark.asyncio
async def test_cancel_event_cancels_pending_approval_upgrade(tmp_path: Path) -> None:
    provider, registry, tool, _, engine = _setup(tmp_path)
    started = asyncio.Event()
    requests: list[ApprovalRequest] = []

    async def upgrade(request: ApprovalRequest) -> Outcome:
        requests.append(request)
        started.set()
        await asyncio.Event().wait()
        return Outcome.ALLOW_ONCE

    cancel_event = asyncio.Event()
    agent = Agent(
        provider,
        registry,
        engine,
        approval_upgrader=upgrade,
        subagent_kind="defined",
    )
    run = asyncio.create_task(_collect(agent, _conversation(), cancel_event))
    await started.wait()
    cancel_event.set()
    await asyncio.wait_for(run, timeout=1)

    assert tool.calls == []
    assert requests[0].respond.cancelled()


async def _collect(
    agent: Agent,
    conversation: Conversation,
    cancel_event: asyncio.Event,
) -> list[Event]:
    return [event async for event in agent.run(conversation, cancel_event=cancel_event)]
