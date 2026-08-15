"""Tests for centralized child Agent and Provider construction."""

from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import SecretStr

from codewright.agent import Agent
from codewright.agent.factory import (
    SubagentFactoryError,
    build_defined_agent_and_conversation,
    build_fork_agent_and_conversation,
)
from codewright.agent.fork import is_fork_context
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
from codewright.subagent import Definition, Source
from codewright.tool import Registry, Result


class Provider:
    provider_name = "test"
    model_name = "test"

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del messages, parameters, tools, request_context
        raise AssertionError("not executed")

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


@dataclass(slots=True)
class Tool:
    name: str
    description: str = "tool"
    parameters: Mapping[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only: bool = True

    async def execute(self, arguments_json: str) -> Result:
        return Result(arguments_json)


def _parent(tmp_path: Path) -> Agent:
    registry = Registry()
    for name in ("read_file", "write_file", "Agent", "TaskList", "mcp__docs__search"):
        registry.register(Tool(name))
    engine = Engine(
        root=tmp_path,
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=tmp_path / ".codewright/settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )
    return Agent(Provider(), registry, engine)


def _definition(**changes) -> Definition:
    values = {
        "name": "reviewer",
        "description": "review",
        "system_prompt": "review system",
        "source": Source.PROJECT,
    }
    values.update(changes)
    return Definition(**values)


def _config(name: str = "secondary", *, context_window: int = 8_000) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        protocol="openai-compatible",
        api_key=SecretStr("test-key"),
        base_url="https://example.com/v1",
        model="model-id",
        context_window=context_window,
    )


def test_defined_inherit_uses_parent_provider_and_independent_runtime(tmp_path: Path) -> None:
    parent = _parent(tmp_path)

    child, conversation, owned = build_defined_agent_and_conversation(
        parent,
        _definition(tools=("read_file", "Agent")),
        "review this",
    )

    assert child.provider is parent.provider
    assert owned is None
    assert child.runtime is not parent.runtime
    assert child.runtime.context_window == parent.runtime.context_window
    assert child.subagent_kind == "defined"
    assert child.allowed_tools == frozenset({"read_file"})
    assert conversation.messages()[0].content == "review system"
    assert conversation.messages()[-1].content == "review this"


def test_requested_provider_precedes_definition_and_is_owned(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    configs = (_config("definition-provider", context_window=4_000), _config())
    created: list[ProviderConfig] = []

    def factory(config: ProviderConfig) -> Provider:
        created.append(config)
        return Provider()

    child, _, owned = build_defined_agent_and_conversation(
        parent,
        _definition(model="definition-provider"),
        "task",
        configs,
        model="secondary",
        provider_factory=factory,
    )

    assert created == [configs[1]]
    assert child.provider is owned
    assert owned is not parent.provider
    assert child.runtime.context_window == 8_000


def test_unknown_and_failed_provider_are_safe(tmp_path: Path) -> None:
    parent = _parent(tmp_path)

    with pytest.raises(SubagentFactoryError) as unknown:
        build_defined_agent_and_conversation(parent, _definition(model="missing"), "task")
    assert unknown.value.error_code == "unknown_provider"

    def fail(config: ProviderConfig) -> Provider:
        del config
        raise RuntimeError("secret provider detail")

    with pytest.raises(SubagentFactoryError) as failed:
        build_defined_agent_and_conversation(
            parent,
            _definition(model="secondary"),
            "task",
            (_config(),),
            provider_factory=fail,
        )
    assert failed.value.error_code == "provider_creation_failed"
    assert "secret" not in failed.value.safe_message


def test_fork_uses_parent_system_history_and_forced_background_tools(tmp_path: Path) -> None:
    parent = _parent(tmp_path)
    parent_conversation = Conversation("parent system")
    parent_conversation.add_user("history")
    definition = _definition(
        name="__fork__",
        system_prompt="",
        tools=("read_file", "write_file", "mcp__docs__search"),
    )

    child, conversation, owned = build_fork_agent_and_conversation(
        parent, definition, parent_conversation, "fork task"
    )

    assert owned is None
    assert child.subagent_kind == "fork"
    assert child.allowed_tools == frozenset({"read_file", "write_file", "mcp__docs__search"})
    assert conversation.messages()[0].content == "parent system"
    assert is_fork_context(conversation)
