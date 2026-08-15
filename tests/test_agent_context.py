"""Tests for task-local Agent execution context."""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codewright.agent import Agent
from codewright.agent.context import (
    ExecutionContext,
    bind_execution_context,
    current_execution_context,
    reset_execution_context,
)
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
from codewright.permission import Decision
from codewright.tool import Registry, Result


class AllowEngine:
    root = Path.cwd().resolve()

    def check(self, mode, call, read_only):
        del mode, call, read_only
        return Decision.ALLOW, ""


class ScriptedProvider:
    provider_name = "context-test"
    model_name = "context-test"

    def __init__(self) -> None:
        self.requests = 0

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
        self.requests += 1
        if self.requests == 1:
            yield StreamEvent.tool_calls_ready((ToolCall("ctx-1", "context_probe", "{}"),))
        else:
            yield StreamEvent.delta("done")
        yield StreamEvent.completed()


@dataclass(slots=True)
class ContextProbeTool:
    name: str = "context_probe"
    description: str = "Capture the active execution context."
    parameters: Mapping[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only: bool = True
    seen: list[ExecutionContext | None] = field(default_factory=list)

    async def execute(self, arguments_json: str) -> Result:
        del arguments_json
        self.seen.append(current_execution_context())
        return Result("ok")


def _conversation(prompt: str) -> Conversation:
    conversation = Conversation("system")
    conversation.add_user(prompt)
    return conversation


def test_context_binding_restores_nested_value() -> None:
    assert current_execution_context() is None
    registry = Registry()
    agent = Agent(ScriptedProvider(), registry, AllowEngine())  # type: ignore[arg-type]
    outer = ExecutionContext(agent, _conversation("outer"))
    inner = ExecutionContext(agent, _conversation("inner"))

    outer_token = bind_execution_context(outer)
    try:
        assert current_execution_context() is outer
        inner_token = bind_execution_context(inner)
        try:
            assert current_execution_context() is inner
        finally:
            reset_execution_context(inner_token)
        assert current_execution_context() is outer
    finally:
        reset_execution_context(outer_token)
    assert current_execution_context() is None


@pytest.mark.asyncio
async def test_agent_tools_see_owner_context_and_it_is_restored() -> None:
    provider = ScriptedProvider()
    tool = ContextProbeTool()
    registry = Registry()
    registry.register(tool)
    agent = Agent(provider, registry, AllowEngine())  # type: ignore[arg-type]
    conversation = _conversation("run")

    _ = [event async for event in agent.run(conversation)]

    assert len(tool.seen) == 1
    assert tool.seen[0] == ExecutionContext(agent, conversation)
    assert current_execution_context() is None


@pytest.mark.asyncio
async def test_contextvars_are_isolated_between_concurrent_tasks() -> None:
    registry = Registry()
    first = ExecutionContext(
        Agent(ScriptedProvider(), registry, AllowEngine()),  # type: ignore[arg-type]
        _conversation("first"),
    )
    second = ExecutionContext(
        Agent(ScriptedProvider(), registry, AllowEngine()),  # type: ignore[arg-type]
        _conversation("second"),
    )

    async def observe(context: ExecutionContext) -> ExecutionContext | None:
        token = bind_execution_context(context)
        try:
            await asyncio.sleep(0)
            return current_execution_context()
        finally:
            reset_execution_context(token)

    assert await asyncio.gather(observe(first), observe(second)) == [first, second]
    assert current_execution_context() is None
