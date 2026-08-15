"""Integration test for main Agent delegation and child safety boundaries."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codewright.agent import Agent
from codewright.agent.agent_tool import AgentTool
from codewright.conversation import Conversation
from codewright.hook import Action as HookAction
from codewright.hook import ActionType as HookActionType
from codewright.hook import Engine as HookEngine
from codewright.hook import Event as HookEvent
from codewright.hook import ExecutionResult as HookExecutionResult
from codewright.hook import PromptAction as HookPromptAction
from codewright.hook import Rule as HookRule
from codewright.hook.executor import Executor as HookExecutor
from codewright.llm import (
    ChatResult,
    Message,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)
from codewright.permission import Engine, Mode
from codewright.permission.rule import RuleSet
from codewright.subagent import Catalog, Definition, Source
from codewright.task import Manager, SubagentApprovalBroker
from codewright.tool import Registry, Result


class RoutingProvider:
    provider_name = "routing"
    model_name = "routing"

    def __init__(self) -> None:
        self.main_requests = 0
        self.child_requests = 0
        self.child_definitions: list[tuple[str, ...]] = []
        self.forged_error_code = ""

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
        del parameters, request_context
        if messages[0].content == "worker system":
            self.child_requests += 1
            self.child_definitions.append(tuple(tool.name for tool in tools))
            if self.child_requests == 1:
                yield StreamEvent.tool_calls_ready(
                    (
                        ToolCall("read", "read_file", '{"path":"README.md"}'),
                        ToolCall("forged", "Agent", '{"prompt":"x"}'),
                    )
                )
            else:
                tool_messages = [message for message in messages if message.tool_results]
                results = [result for message in tool_messages for result in message.tool_results]
                self.forged_error_code = next(
                    result.error_code or "" for result in results if result.tool_call_id == "forged"
                )
                yield StreamEvent.delta("delegated answer")
            yield StreamEvent.completed()
            return

        self.main_requests += 1
        if self.main_requests == 1:
            yield StreamEvent.tool_calls_ready(
                (
                    ToolCall(
                        "delegate",
                        "Agent",
                        json.dumps(
                            {
                                "prompt": "inspect project",
                                "description": "inspection",
                                "subagent_type": "worker",
                            }
                        ),
                    ),
                )
            )
        else:
            yield StreamEvent.delta("main final")
        yield StreamEvent.completed()


@dataclass(slots=True)
class ReadTool:
    name: str = "read_file"
    description: str = "read"
    parameters: Mapping[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only: bool = True
    calls: int = 0

    async def execute(self, arguments_json: str) -> Result:
        del arguments_json
        self.calls += 1
        return Result("contents")


class ProbeExecutor(HookExecutor):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def run(self, rule, payload, *, blocking):
        del blocking
        self.calls.append((rule.name, str(payload.get("tool_name", ""))))
        return HookExecutionResult()

    async def aclose(self) -> None:
        return None


def _engine(tmp_path: Path) -> Engine:
    return Engine(
        root=tmp_path,
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=tmp_path / ".codewright/settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


@pytest.mark.asyncio
async def test_main_agent_runs_child_with_filtered_tools_and_independent_hooks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_immediately(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("codewright.agent.asyncio.to_thread", run_immediately)
    provider = RoutingProvider()
    manager = Manager()
    broker = SubagentApprovalBroker()
    catalog = Catalog()
    catalog.add_all(
        (
            Definition(
                "worker",
                "worker",
                tools=("read_file", "Agent"),
                system_prompt="worker system",
                source=Source.PROJECT,
            ),
        ),
        Source.PROJECT,
    )
    probe = ProbeExecutor()
    hooks = HookEngine(
        [
            HookRule(
                "pre-once",
                HookEvent.PRE_TOOL_USE,
                HookAction(HookActionType.PROMPT, prompt=HookPromptAction("")),
                only_once=True,
            ),
            HookRule(
                "post",
                HookEvent.POST_TOOL_USE,
                HookAction(HookActionType.PROMPT, prompt=HookPromptAction("")),
            ),
        ],
        [],
        executor=probe,
    )
    registry = Registry()
    read_tool = ReadTool()
    registry.register(read_tool)
    agent_tool = AgentTool(catalog, manager, broker)
    registry.register(agent_tool)
    main = Agent(
        provider,
        registry,
        _engine(tmp_path),
        runtime=None,
        hook_engine=hooks,
    )
    agent_tool.set_parent(main)
    conversation = Conversation("main system")
    conversation.add_user("delegate")

    events = [event async for event in main.run(conversation, mode=Mode.BYPASS)]

    assert events[-1].done
    assert conversation.messages()[-1].content == "main final"
    assert read_tool.calls == 1
    assert provider.child_definitions == [("read_file",), ("read_file",)]
    assert provider.forged_error_code == "tool_not_allowed"
    assert ("pre-once", "Agent") in probe.calls
    assert ("pre-once", "read_file") in probe.calls
    assert ("post", "read_file") in probe.calls
    assert ("post", "Agent") in probe.calls
    await manager.aclose()
    await broker.aclose()
    await hooks.aclose()
