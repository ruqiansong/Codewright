"""Unit tests for AgentTool validation, lifecycle, and background transfer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from codewright.agent import Agent
from codewright.agent.agent_tool import AgentTool
from codewright.agent.context import (
    ExecutionContext,
    bind_execution_context,
    reset_execution_context,
)
from codewright.agent.fork import FORK_BOILERPLATE
from codewright.agent.team_hook import TeamSpawnRequest
from codewright.config import ProviderConfig
from codewright.conversation import Conversation
from codewright.llm import (
    ChatResult,
    Message,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolDefinition,
)
from codewright.permission import Engine, Mode
from codewright.permission.rule import RuleSet
from codewright.subagent import Catalog, Definition, Source
from codewright.task import Manager, Status, SubagentApprovalBroker
from codewright.tool import Registry, Result, Tool


class Provider:
    provider_name = "test"
    model_name = "test"

    def __init__(self, replies: Sequence[Sequence[StreamEvent]] = ()) -> None:
        self.replies = list(replies)
        self.requests: list[tuple[Message, ...]] = []
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
        self.requests.append(tuple(messages))
        for event in self.replies.pop(0):
            yield event

    async def close(self) -> None:
        self.closed += 1


class BlockingProvider(Provider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters, tools, request_context
        self.requests.append(tuple(messages))
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        yield StreamEvent.delta("released")
        yield StreamEvent.completed()


class SlowReturnManager(Manager):
    def __init__(self) -> None:
        super().__init__()
        self.adopted = asyncio.Event()
        self.release_return = asyncio.Event()

    async def adopt_running(self, *args, **kwargs):
        task = await super().adopt_running(*args, **kwargs)
        self.adopted.set()
        await self.release_return.wait()
        return task


class RecordingTeamHook:
    def __init__(self) -> None:
        self.requests: list[TeamSpawnRequest] = []

    async def spawn_teammate(self, request: TeamSpawnRequest) -> Result:
        self.requests.append(request)
        return Result(json.dumps({"member": request.member_name}))


def _engine(tmp_path: Path) -> Engine:
    return Engine(
        root=tmp_path,
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=tmp_path / ".codewright/settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


def _catalog(*, background: bool = False, model: str = "inherit") -> Catalog:
    catalog = Catalog()
    catalog.add_all(
        (
            Definition(
                "worker",
                "worker role",
                model=model,
                background=background,
                system_prompt="worker system",
                source=Source.PROJECT,
            ),
        ),
        Source.PROJECT,
    )
    return catalog


def _parent(provider: Provider, tmp_path: Path) -> Agent:
    return Agent(provider, Registry(), _engine(tmp_path))


def _conversation() -> Conversation:
    value = Conversation("main system")
    value.add_user("main task")
    return value


def _arguments(**changes: object) -> str:
    values: dict[str, object] = {
        "prompt": "delegated task",
        "description": "delegation",
        "subagent_type": "worker",
    }
    values.update(changes)
    return json.dumps(values)


def _tool(
    parent: Agent,
    catalog: Catalog,
    manager: Manager,
    broker: SubagentApprovalBroker,
    **options,
) -> AgentTool:
    tool = AgentTool(catalog, manager, broker, **options)
    tool.set_parent(parent)
    return tool


async def _execute(tool: AgentTool, parent: Agent, conversation: Conversation, raw: str):
    token = bind_execution_context(ExecutionContext(parent, conversation))
    try:
        return await tool.execute(raw)
    finally:
        reset_execution_context(token)


async def _terminal(task) -> None:
    for _ in range(200):
        if task.status is not Status.RUNNING:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("background task did not settle")


def test_agent_tool_protocol_schema_and_timeout_convention() -> None:
    manager = Manager()
    broker = SubagentApprovalBroker()
    tool = AgentTool(_catalog(), manager, broker)

    assert isinstance(tool, Tool)
    assert tool.name == "Agent"
    assert tool.read_only is False
    assert tool.execution_timeout is None
    assert set(tool.parameters["properties"]) == {
        "prompt",
        "description",
        "subagent_type",
        "model",
        "run_in_background",
        "name",
        "team_name",
        "plan_mode_required",
    }


@pytest.mark.asyncio
async def test_agent_tool_rejects_invalid_arguments_and_initialization(tmp_path: Path) -> None:
    manager = Manager()
    broker = SubagentApprovalBroker()
    parent = _parent(Provider(), tmp_path)
    tool = AgentTool(_catalog(), manager, broker)

    assert (await tool.execute(_arguments())).error_code == "not_initialized"
    for raw in ("{}", "[]", "not-json", _arguments(run_in_background="yes")):
        result = await _execute(tool, parent, _conversation(), raw)
        assert result.error_code in {"invalid_arguments", "not_initialized"}
    tool.set_parent(parent)
    unknown = await _execute(tool, parent, _conversation(), _arguments(subagent_type="missing"))
    assert unknown.error_code == "unknown_subagent"
    await manager.aclose()
    await broker.aclose()


@pytest.mark.asyncio
async def test_agent_tool_delegates_team_spawn_without_changing_plain_path(
    tmp_path: Path,
) -> None:
    parent = _parent(Provider(), tmp_path)
    manager = Manager()
    broker = SubagentApprovalBroker()
    hook = RecordingTeamHook()
    tool = _tool(parent, _catalog(), manager, broker, team_hook=hook)

    result = await _execute(
        tool,
        parent,
        _conversation(),
        _arguments(team_name="demo", name="alice", plan_mode_required=True),
    )

    assert not result.is_error
    assert hook.requests == [
        TeamSpawnRequest(
            team_name="demo",
            member_name="alice",
            prompt="delegated task",
            description="delegation",
            subagent_type="worker",
            plan_mode_required=True,
        )
    ]
    await manager.aclose()
    await broker.aclose()


@pytest.mark.asyncio
async def test_defined_foreground_returns_final_text(tmp_path: Path) -> None:
    provider = Provider([(StreamEvent.delta("child result"), StreamEvent.completed())])
    parent = _parent(provider, tmp_path)
    manager = Manager()
    broker = SubagentApprovalBroker()
    tool = _tool(parent, _catalog(), manager, broker)

    result = await _execute(tool, parent, _conversation(), _arguments())

    assert result.content == "child result"
    assert not result.is_error
    assert len(provider.requests) == 1
    await manager.aclose()
    await broker.aclose()


@pytest.mark.asyncio
async def test_explicit_background_and_fork_launch_without_duplicate_prompt(
    tmp_path: Path,
) -> None:
    provider = Provider(
        [
            (StreamEvent.delta("background"), StreamEvent.completed()),
            (StreamEvent.delta("fork"), StreamEvent.completed()),
        ]
    )
    parent = _parent(provider, tmp_path)
    manager = Manager()
    broker = SubagentApprovalBroker()
    tool = _tool(parent, _catalog(), manager, broker)
    conversation = _conversation()

    background = await _execute(
        tool,
        parent,
        conversation,
        _arguments(run_in_background=True, name="worker-one"),
    )
    assert json.loads(background.content)["status"] == "async_launched"
    fork = await _execute(
        tool,
        parent,
        conversation,
        _arguments(subagent_type="", prompt="fork task"),
    )
    assert json.loads(fork.content)["status"] == "async_launched"
    tasks = await manager.list()
    assert len(tasks) == 2
    assert (
        sum(message.content.count("fork task") for message in tasks[1].conversation.messages()) == 1
    )
    assert FORK_BOILERPLATE in tasks[1].conversation.messages()[-1].content
    for task in tasks:
        await _terminal(task)
    await manager.aclose()
    await broker.aclose()


@pytest.mark.asyncio
async def test_foreground_timeout_adopts_same_single_execution(tmp_path: Path) -> None:
    provider = BlockingProvider()
    parent = Agent(provider, Registry(default_timeout=0.0001), _engine(tmp_path))
    manager = Manager()
    broker = SubagentApprovalBroker()
    tool = _tool(
        parent,
        _catalog(),
        manager,
        broker,
        foreground_timeout=0.001,
    )

    parent.registry.register(tool)
    token = bind_execution_context(ExecutionContext(parent, _conversation()))
    try:
        result = await parent.registry.execute("Agent", _arguments())
    finally:
        reset_execution_context(token)

    payload = json.loads(result.content)
    assert payload["status"] == "timed_out_to_background"
    assert result.error_code != "tool_timeout"
    tasks = await manager.list()
    assert len(tasks) == 1
    assert tasks[0].handle is not None
    await asyncio.wait_for(provider.started.wait(), timeout=1)
    assert len(provider.requests) == 1
    provider.release.set()
    await _terminal(tasks[0])
    assert tasks[0].result == "released"
    assert len(provider.requests) == 1
    await manager.aclose()
    await broker.aclose()


@pytest.mark.asyncio
async def test_background_disabled_rejects_all_background_routes(tmp_path: Path) -> None:
    provider = Provider([(StreamEvent.delta("ok"), StreamEvent.completed())])
    parent = _parent(provider, tmp_path)
    manager = Manager()
    broker = SubagentApprovalBroker()
    tool = _tool(
        parent,
        _catalog(),
        manager,
        broker,
        enable_subagent_background=False,
    )

    explicit = await _execute(tool, parent, _conversation(), _arguments(run_in_background=True))
    fork = await _execute(tool, parent, _conversation(), _arguments(subagent_type=""))
    role_background_tool = _tool(
        parent,
        _catalog(background=True),
        manager,
        broker,
        enable_subagent_background=False,
    )
    role_background = await _execute(role_background_tool, parent, _conversation(), _arguments())
    foreground = await _execute(tool, parent, _conversation(), _arguments())

    assert explicit.error_code == "background_disabled"
    assert fork.error_code == "background_disabled"
    assert role_background.error_code == "background_disabled"
    assert foreground.content == "ok"
    await manager.aclose()
    await broker.aclose()


@pytest.mark.asyncio
async def test_nested_and_fork_context_calls_are_rejected(tmp_path: Path) -> None:
    provider = Provider()
    child = Agent(
        provider,
        Registry(),
        _engine(tmp_path),
        subagent_kind="defined",
    )
    manager = Manager()
    broker = SubagentApprovalBroker()
    nested_tool = _tool(child, _catalog(), manager, broker)
    nested = await _execute(nested_tool, child, _conversation(), _arguments())

    main = _parent(provider, tmp_path)
    fork_tool = _tool(main, _catalog(), manager, broker)
    fork_conversation = Conversation("system")
    fork_conversation.add_user(FORK_BOILERPLATE + "task")
    forked = await _execute(fork_tool, main, fork_conversation, _arguments())

    assert nested.error_code == "nested_subagent"
    assert forked.error_code == "nested_subagent"
    await manager.aclose()
    await broker.aclose()


@pytest.mark.asyncio
async def test_unknown_provider_and_owned_provider_close_are_safe(tmp_path: Path) -> None:
    parent = _parent(Provider(), tmp_path)
    manager = Manager()
    broker = SubagentApprovalBroker()
    config = ProviderConfig(
        name="secondary",
        protocol="openai-compatible",
        api_key=SecretStr("secret"),
        base_url="https://example.com/v1",
        model="model",
    )
    secondary = Provider([(StreamEvent.delta("owned result"), StreamEvent.completed())])
    tool = _tool(
        parent,
        _catalog(),
        manager,
        broker,
        provider_configs=(config,),
        provider_factory=lambda selected: secondary,
    )

    unknown = await _execute(tool, parent, _conversation(), _arguments(model="missing"))
    completed = await _execute(tool, parent, _conversation(), _arguments(model="secondary"))

    assert unknown.error_code == "unknown_provider"
    assert completed.content == "owned result"
    assert secondary.closed == 1
    await manager.aclose()
    await broker.aclose()


@pytest.mark.asyncio
async def test_outer_cancellation_cleans_unadopted_handle_and_provider(tmp_path: Path) -> None:
    parent = _parent(Provider(), tmp_path)
    manager = Manager()
    broker = SubagentApprovalBroker()
    config = ProviderConfig(
        name="secondary",
        protocol="openai-compatible",
        api_key=SecretStr("secret"),
        base_url="https://example.com/v1",
        model="model",
    )
    secondary = BlockingProvider()
    tool = _tool(
        parent,
        _catalog(),
        manager,
        broker,
        provider_configs=(config,),
        provider_factory=lambda selected: secondary,
        foreground_timeout=10,
    )
    conversation = _conversation()
    token = bind_execution_context(ExecutionContext(parent, conversation))
    try:
        execution = asyncio.create_task(tool.execute(_arguments(model="secondary")))
        await secondary.started.wait()
        execution.cancel()
        with pytest.raises(asyncio.CancelledError):
            await execution
    finally:
        reset_execution_context(token)

    assert secondary.cancelled
    assert secondary.closed == 1
    assert await manager.list() == ()
    await manager.aclose()
    await broker.aclose()


@pytest.mark.asyncio
async def test_outer_cancellation_after_adopt_leaves_manager_owned_task_running(
    tmp_path: Path,
) -> None:
    provider = BlockingProvider()
    parent = _parent(provider, tmp_path)
    manager = SlowReturnManager()
    broker = SubagentApprovalBroker()
    tool = _tool(
        parent,
        _catalog(),
        manager,
        broker,
        foreground_timeout=0.001,
    )
    conversation = _conversation()
    token = bind_execution_context(ExecutionContext(parent, conversation))
    try:
        execution = asyncio.create_task(tool.execute(_arguments()))
        await manager.adopted.wait()
        execution.cancel()
        manager.release_return.set()
        with pytest.raises(asyncio.CancelledError):
            await execution
    finally:
        reset_execution_context(token)

    tasks = await manager.list()
    assert len(tasks) == 1
    assert tasks[0].status is Status.RUNNING
    assert tasks[0].handle is not None and not tasks[0].handle.cancelled()
    provider.release.set()
    await _terminal(tasks[0])
    assert tasks[0].status is Status.COMPLETED
    await manager.aclose()
    await broker.aclose()
