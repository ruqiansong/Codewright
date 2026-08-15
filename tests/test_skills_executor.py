"""Tests for inline and isolated fork Skill execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
from pydantic import SecretStr

from codewright.agent import Agent
from codewright.agent.runtime import SessionRuntime
from codewright.compact import (
    CompactCircuitBreaker,
    ContentReplacementState,
    RecoveryState,
    SessionContext,
)
from codewright.config import ProviderConfig
from codewright.conversation import Conversation
from codewright.llm import (
    ChatResult,
    LLMServiceError,
    Message,
    MessageRole,
    RequestContext,
    RequestParameters,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from codewright.permission import Engine, Mode
from codewright.permission.rule import RuleSet
from codewright.skills import ForkResult, SkillExecutionError, SkillExecutor, SkillLoader
from codewright.skills.models import SkillContext, SkillDef, SkillSource
from codewright.tool import LoadSkillTool, Registry, Result


class RecordingProvider:
    provider_name = "primary"
    model_name = "test-model"

    def __init__(self, replies: Sequence[Sequence[StreamEvent]]) -> None:
        self._replies = tuple(tuple(reply) for reply in replies)
        self.requests: list[tuple[Message, ...]] = []
        self.contexts: list[RequestContext | None] = []
        self.tool_definitions: list[tuple[ToolDefinition, ...]] = []
        self.closed = False

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del parameters
        index = len(self.requests)
        self.requests.append(tuple(messages))
        self.contexts.append(request_context)
        self.tool_definitions.append(tuple(tools))
        reply = self._replies[index]
        text = "".join(event.text for event in reply)
        usage = next((event.usage for event in reversed(reply) if event.usage), None)
        error = next((event.error for event in reply if event.error), None)
        if error is not None:
            raise error
        return ChatResult(Message(MessageRole.ASSISTANT, text), self.model_name, usage=usage)

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters
        index = len(self.requests)
        self.requests.append(tuple(messages))
        self.contexts.append(request_context)
        self.tool_definitions.append(tuple(tools))
        for event in self._replies[index]:
            yield event

    async def close(self) -> None:
        self.closed = True


class BlockingProvider(RecordingProvider):
    def __init__(self) -> None:
        super().__init__(((),))
        self.started = asyncio.Event()
        self.cancelled = False

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters, tools
        self.requests.append(tuple(messages))
        self.contexts.append(request_context)
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        yield StreamEvent.completed()


def skill(
    tmp_path: Path,
    *,
    mode: str = "fork",
    context: SkillContext = "none",
    model: str | None = None,
    body: str = "Do $ARGUMENTS now.",
) -> SkillDef:
    path = (tmp_path / "skill.md").resolve()
    return SkillDef(
        name="review",
        description="Review changes",
        prompt_body=body,
        mode=mode,  # type: ignore[arg-type]
        model=model,
        context=context,
        source_path=path,
        source_dir=path.parent,
        is_directory=False,
        source=SkillSource.PROJECT,
    )


def agent_for(
    provider: RecordingProvider,
    tmp_path: Path,
    registry: Registry | None = None,
) -> Agent:
    root = tmp_path.resolve()
    engine = Engine(
        root=root,
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=root / ".codewright" / "settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )
    runtime = SessionRuntime(
        ContentReplacementState(),
        RecoveryState(),
        CompactCircuitBreaker(),
        SessionContext("main", str(root / "spill")),
        context_window=12_345,
    )
    return Agent(provider, registry or Registry(), engine, runtime=runtime, version="test")


def reply(text: str = "fork result") -> tuple[StreamEvent, ...]:
    return (StreamEvent.delta(text), StreamEvent.completed())


def main_conversation() -> Conversation:
    conversation = Conversation("Main system prompt")
    conversation.add_user("user one")
    conversation.add_assistant("assistant one")
    conversation.add_assistant_with_tool_calls(
        "checking",
        (ToolCall("call-1", "read_file", '{"path":"a"}'),),
    )
    conversation.add_tool_results((ToolResult("call-1", "read_file", "data"),))
    conversation.add_user("user two")
    conversation.add_assistant("assistant two")
    conversation.add_user("user three")
    conversation.add_assistant("assistant three")
    conversation.add_user("user four")
    return conversation


def test_inline_substitutes_arguments_activates_and_returns_body(tmp_path: Path) -> None:
    agent = agent_for(RecordingProvider((reply(),)), tmp_path)
    executor = SkillExecutor(agent)

    rendered = executor.execute_inline(
        skill(tmp_path, mode="inline", body="Review $ARGUMENTS and $ARGUMENTS."),
        "src",
    )

    assert rendered == "Review src and src."
    active = agent.list_active_skills()
    assert [(entry.name, entry.body) for entry in active] == [("review", rendered)]


@pytest.mark.asyncio
@pytest.mark.parametrize("context", ["none", "recent", "full"])
async def test_fork_builds_expected_context(tmp_path: Path, context: SkillContext) -> None:
    provider = RecordingProvider((reply(),))
    agent = agent_for(provider, tmp_path)
    main = main_conversation()

    result = await SkillExecutor(agent).execute_fork(
        skill(tmp_path, context=context),
        "carefully",
        main,
    )

    assert result == ForkResult("fork result", TokenUsage(0, 0, 0))
    request = provider.requests[0]
    assert request[0] == Message(MessageRole.SYSTEM, "Main system prompt")
    assert request[-1] == Message(MessageRole.USER, "Do carefully now.")
    assert sum(message.role is MessageRole.SYSTEM for message in request) == 1
    if context == "none":
        assert len(request) == 2
    elif context == "recent":
        assert [message.content for message in request[1:-1]] == [
            "user two",
            "assistant two",
            "user three",
            "assistant three",
            "user four",
        ]
        assert all(not message.tool_calls and not message.tool_results for message in request)
    else:
        assert request[1:-1] == main.messages()[1:]


@pytest.mark.asyncio
async def test_fork_does_not_mutate_main_history_or_active_skills(tmp_path: Path) -> None:
    provider = RecordingProvider((reply(),))
    agent = agent_for(provider, tmp_path)
    agent.activate_skill("main-only", "MAIN SECRET", tmp_path.resolve())
    main = main_conversation()
    before = main.messages()

    await SkillExecutor(agent).execute_fork(skill(tmp_path), "work", main)

    assert main.messages() == before
    assert agent.runtime.active_skills.names() == ("main-only",)
    assert provider.contexts[0] is not None
    assert "MAIN SECRET" not in provider.contexts[0].environment


@pytest.mark.asyncio
async def test_fork_load_skill_activates_only_child_runtime(tmp_path: Path) -> None:
    path = tmp_path / ".codewright" / "skills" / "child.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\nname: child\ndescription: Child SOP\n---\nCHILD ONLY BODY\n",
        encoding="utf-8",
    )
    loader = SkillLoader(tmp_path, tmp_path / "home")
    loader.load_all()
    load_tool = LoadSkillTool(loader)
    registry = Registry()
    registry.register(load_tool)
    provider = RecordingProvider(
        (
            (
                StreamEvent.tool_calls_ready(
                    (ToolCall("load-child", "load_skill", '{"name":"child"}'),)
                ),
                StreamEvent.completed(),
            ),
            reply("done"),
        )
    )
    agent = agent_for(provider, tmp_path, registry)
    load_tool.set_agent(agent)

    result = await SkillExecutor(agent).execute_fork(
        skill(tmp_path), "work", Conversation("system")
    )

    assert result.text == "done"
    assert agent.list_active_skills() == ()
    assert len(provider.contexts) == 2
    assert provider.contexts[1] is not None
    assert "CHILD ONLY BODY" in provider.contexts[1].environment


@pytest.mark.asyncio
async def test_fork_accumulates_every_usage_event(tmp_path: Path) -> None:
    first = TokenUsage(10, 2, 12, 3, 4)
    second = TokenUsage(20, 5, 25, 6, 7)
    provider = RecordingProvider(
        ((StreamEvent.usage_report(first), StreamEvent.usage_report(second), *reply("done")),)
    )

    result = await SkillExecutor(agent_for(provider, tmp_path)).execute_fork(
        skill(tmp_path), "work", Conversation("system")
    )

    assert result == ForkResult("done", TokenUsage(30, 7, 37, 9, 11))


@pytest.mark.asyncio
async def test_fork_marks_skill_child_and_filters_subagent_meta_tools(tmp_path: Path) -> None:
    registry = Registry()
    registry.register(LoadSkillTool(SkillLoader(tmp_path, tmp_path / "home")))

    class MetaTool:
        description = "meta"
        parameters = {"type": "object", "properties": {}}
        read_only = False

        def __init__(self, name: str) -> None:
            self.name = name

        async def execute(self, arguments_json: str) -> Result:
            del arguments_json
            raise AssertionError("meta tool must not execute")

    for name in ("Agent", "TaskList", "TaskGet", "TaskStop", "SendMessage"):
        registry.register(MetaTool(name))  # type: ignore[arg-type]
    provider = RecordingProvider((reply("done"),))
    parent = agent_for(provider, tmp_path, registry)

    result = await SkillExecutor(parent).execute_fork(
        skill(tmp_path), "work", Conversation("system")
    )

    assert result.text == "done"
    assert [definition.name for definition in provider.tool_definitions[0]] == ["load_skill"]


@pytest.mark.asyncio
async def test_fork_converts_agent_error_to_safe_execution_error(tmp_path: Path) -> None:
    provider = RecordingProvider(((StreamEvent.failed(LLMServiceError("Safe provider failure")),),))

    with pytest.raises(SkillExecutionError, match="Safe provider failure") as caught:
        await SkillExecutor(agent_for(provider, tmp_path)).execute_fork(
            skill(tmp_path), "work", Conversation("system")
        )

    assert caught.value.safe_message == "Safe provider failure"


@pytest.mark.asyncio
async def test_fork_supports_cancel_event(tmp_path: Path) -> None:
    provider = BlockingProvider()
    cancel_event = asyncio.Event()
    task = asyncio.create_task(
        SkillExecutor(agent_for(provider, tmp_path)).execute_fork(
            skill(tmp_path), "work", Conversation("system"), cancel_event
        )
    )
    await asyncio.wait_for(provider.started.wait(), timeout=1)

    cancel_event.set()

    with pytest.raises(SkillExecutionError, match="cancelled"):
        await asyncio.wait_for(task, timeout=1)
    assert provider.cancelled is True


def secondary_config() -> ProviderConfig:
    return ProviderConfig(
        name="secondary",
        protocol="openai-compatible",
        api_key=SecretStr("secret"),
        base_url="https://example.com/v1",
        model="secondary-model",
        stream=True,
        context_window=8_000,
    )


@pytest.mark.asyncio
async def test_named_provider_is_selected_and_owned_provider_is_closed(tmp_path: Path) -> None:
    primary = RecordingProvider((reply("unused"),))
    secondary = RecordingProvider((reply("secondary result"),))
    selected: list[str] = []

    def factory(config: ProviderConfig) -> RecordingProvider:
        selected.append(config.name)
        return secondary

    executor = SkillExecutor(
        agent_for(primary, tmp_path),
        (secondary_config(),),
        provider_factory=factory,
    )

    result = await executor.execute_fork(
        skill(tmp_path, model="secondary"), "work", Conversation("system")
    )

    assert result.text == "secondary result"
    assert selected == ["secondary"]
    assert secondary.closed is True
    assert primary.closed is False
    assert primary.requests == []


@pytest.mark.asyncio
async def test_unknown_named_provider_fails_safely(tmp_path: Path) -> None:
    provider = RecordingProvider((reply(),))

    with pytest.raises(SkillExecutionError, match="not configured: missing"):
        await SkillExecutor(agent_for(provider, tmp_path)).execute_fork(
            skill(tmp_path, model="missing"), "work", Conversation("system")
        )

    assert provider.requests == []


def test_executor_rejects_wrong_execution_mode(tmp_path: Path) -> None:
    executor = SkillExecutor(agent_for(RecordingProvider((reply(),)), tmp_path))

    with pytest.raises(SkillExecutionError, match="inline"):
        executor.execute_inline(skill(tmp_path, mode="fork"), "args")
