"""Centralized child Agent, Provider, runtime, and conversation construction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal

from codewright.agent.filter import filter_tool_names
from codewright.agent.fork import build_fork_conversation
from codewright.agent.runtime import SessionRuntime
from codewright.compact import (
    CompactCircuitBreaker,
    ContentReplacementState,
    RecoveryState,
    new_session_context,
)
from codewright.config import ProviderConfig, effective_context_window
from codewright.conversation import Conversation
from codewright.llm import Provider
from codewright.llm.factory import create_provider
from codewright.subagent import Definition

if TYPE_CHECKING:
    from codewright.agent import Agent, ApprovalUpgrader
    from codewright.skills import SkillDef

type ProviderFactory = Callable[[ProviderConfig], Provider]


class SubagentFactoryError(RuntimeError):
    """A child construction failure safe to expose to a caller."""

    def __init__(self, safe_message: str, *, error_code: str) -> None:
        self.safe_message = safe_message
        self.error_code = error_code
        super().__init__(safe_message)


def build_defined_agent_and_conversation(
    parent: Agent,
    definition: Definition,
    task: str,
    provider_configs: Sequence[ProviderConfig] = (),
    *,
    model: str | None = None,
    background: bool | None = None,
    approval_upgrader: ApprovalUpgrader | None = None,
    provider_factory: ProviderFactory = create_provider,
) -> tuple[Agent, Conversation, Provider | None]:
    """Build a definition-backed child and return explicit Provider ownership."""
    _validate_inputs(
        parent, definition, task, provider_configs, model, background, provider_factory
    )
    if definition.is_fork():
        raise ValueError("a fork definition requires build_fork_agent_and_conversation")
    provider, config, owned_provider = _select_provider(
        parent,
        definition,
        provider_configs,
        model=model,
        provider_factory=provider_factory,
    )
    conversation = Conversation(definition.system_prompt)
    conversation.add_user(task)
    child = _build_agent(
        parent,
        definition,
        provider,
        config,
        background=definition.background if background is None else background,
        approval_upgrader=approval_upgrader,
        kind="defined",
    )
    return child, conversation, owned_provider


def build_fork_agent_and_conversation(
    parent: Agent,
    definition: Definition,
    parent_conversation: Conversation,
    task: str,
    provider_configs: Sequence[ProviderConfig] = (),
    *,
    model: str | None = None,
    approval_upgrader: ApprovalUpgrader | None = None,
    provider_factory: ProviderFactory = create_provider,
) -> tuple[Agent, Conversation, Provider | None]:
    """Build a fork child with copied parent context and forced background filtering."""
    _validate_inputs(parent, definition, task, provider_configs, model, True, provider_factory)
    if not isinstance(parent_conversation, Conversation):
        raise TypeError("parent_conversation must be a Conversation")
    provider, config, owned_provider = _select_provider(
        parent,
        definition,
        provider_configs,
        model=model,
        provider_factory=provider_factory,
    )
    conversation = build_fork_conversation(parent_conversation, task)
    child = _build_agent(
        parent,
        definition,
        provider,
        config,
        background=True,
        approval_upgrader=approval_upgrader,
        kind="fork",
    )
    return child, conversation, owned_provider


def build_skill_agent_and_conversation(
    parent: Agent,
    skill: SkillDef,
    task: str,
    conversation: Conversation,
    provider_configs: Sequence[ProviderConfig] = (),
    *,
    approval_upgrader: ApprovalUpgrader | None = None,
    provider_factory: ProviderFactory = create_provider,
) -> tuple[Agent, Conversation, Provider | None, ProviderConfig | None]:
    """Build an isolated Skill child while preserving its context policy."""
    from codewright.agent import Agent
    from codewright.skills import SkillDef
    from codewright.skills.executor import fork_skill_conversation

    if not isinstance(parent, Agent):
        raise TypeError("parent must be an Agent")
    if not isinstance(skill, SkillDef):
        raise TypeError("skill must be a SkillDef")
    if not isinstance(task, str):
        raise TypeError("task must be a string")
    if not isinstance(conversation, Conversation):
        raise TypeError("conversation must be a Conversation")
    configs = tuple(provider_configs)
    if not all(isinstance(config, ProviderConfig) for config in configs):
        raise TypeError("provider_configs must contain only ProviderConfig values")
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("provider config names must be unique")
    if not callable(provider_factory):
        raise TypeError("provider_factory must be callable")

    config = None
    owned_provider = None
    provider = parent.provider
    if skill.model is not None:
        config = next((item for item in configs if item.name == skill.model), None)
        if config is None:
            raise SubagentFactoryError(
                f"Skill provider is not configured: {skill.model}",
                error_code="unknown_provider",
            )
        try:
            provider = provider_factory(config)
        except Exception:
            raise SubagentFactoryError(
                "The Skill provider could not be created.",
                error_code="provider_creation_failed",
            ) from None
        if not isinstance(provider, Provider):
            raise SubagentFactoryError(
                "The Skill provider could not be created.",
                error_code="provider_creation_failed",
            )
        owned_provider = provider

    child_conversation = fork_skill_conversation(skill, task, conversation)
    names = tuple(item.name for item in parent.registry.definitions())
    allowed = frozenset(
        name
        for name in names
        if name not in {"Agent", "TaskList", "TaskGet", "TaskStop", "SendMessage"}
    )
    runtime = SessionRuntime(
        replacement=ContentReplacementState(),
        recovery=RecoveryState(),
        auto_tracking=CompactCircuitBreaker(),
        session=new_session_context(str(parent.permission_engine.root)),
        context_window=(
            effective_context_window(config)
            if config is not None
            else parent.runtime.context_window
        ),
    )
    child = Agent(
        provider,
        parent.registry,
        parent.permission_engine,
        version=parent.version,
        runtime=runtime,
        hook_engine=parent.hook_engine,
        allowed_tools=allowed,
        approval_upgrader=approval_upgrader,
        subagent_name=skill.name,
        subagent_kind="skill",
    )
    return child, child_conversation, owned_provider, config


def _select_provider(
    parent: Agent,
    definition: Definition,
    provider_configs: Sequence[ProviderConfig],
    *,
    model: str | None,
    provider_factory: ProviderFactory,
) -> tuple[Provider, ProviderConfig | None, Provider | None]:
    selected = model if model is not None else definition.model
    if selected == "inherit":
        return parent.provider, None, None
    configs = {config.name: config for config in provider_configs}
    config = configs.get(selected)
    if config is None:
        raise SubagentFactoryError(
            f"Subagent provider is not configured: {selected}",
            error_code="unknown_provider",
        )
    try:
        provider = provider_factory(config)
    except Exception:
        raise SubagentFactoryError(
            "The subagent provider could not be created.",
            error_code="provider_creation_failed",
        ) from None
    if not isinstance(provider, Provider):
        raise SubagentFactoryError(
            "The subagent provider could not be created.",
            error_code="provider_creation_failed",
        )
    return provider, config, provider


def _build_agent(
    parent: Agent,
    definition: Definition,
    provider: Provider,
    config: ProviderConfig | None,
    *,
    background: bool,
    approval_upgrader: ApprovalUpgrader | None,
    kind: Literal["defined", "fork"],
) -> Agent:
    from codewright.agent import Agent

    names = tuple(item.name for item in parent.registry.definitions())
    allowed = frozenset(
        filter_tool_names(
            names,
            source=definition.source,
            tools=definition.tools,
            disallowed_tools=definition.disallowed_tools,
            background=background,
        )
    )
    runtime = SessionRuntime(
        replacement=ContentReplacementState(),
        recovery=RecoveryState(),
        auto_tracking=CompactCircuitBreaker(),
        session=new_session_context(str(parent.permission_engine.root)),
        context_window=(
            effective_context_window(config)
            if config is not None
            else parent.runtime.context_window
        ),
    )
    return Agent(
        provider,
        parent.registry,
        parent.permission_engine,
        version=parent.version,
        runtime=runtime,
        hook_engine=parent.hook_engine,
        max_turns=definition.max_turns,
        allowed_tools=allowed,
        permission_mode=definition.permission_mode,
        dont_ask=definition.dont_ask,
        approval_upgrader=approval_upgrader,
        subagent_name=definition.name,
        subagent_kind=kind,
    )


def _validate_inputs(
    parent: Agent,
    definition: Definition,
    task: str,
    provider_configs: Sequence[ProviderConfig],
    model: str | None,
    background: bool | None,
    provider_factory: ProviderFactory,
) -> None:
    from codewright.agent import Agent

    if not isinstance(parent, Agent):
        raise TypeError("parent must be an Agent")
    if not isinstance(definition, Definition):
        raise TypeError("definition must be a Definition")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    configs = tuple(provider_configs)
    if not all(isinstance(config, ProviderConfig) for config in configs):
        raise TypeError("provider_configs must contain only ProviderConfig values")
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("provider config names must be unique")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("model must be a non-empty string or None")
    if model is not None and model != model.strip():
        raise ValueError("model must be trimmed")
    if background is not None and not isinstance(background, bool):
        raise TypeError("background must be a boolean or None")
    if not callable(provider_factory):
        raise TypeError("provider_factory must be callable")


__all__ = [
    "ProviderFactory",
    "SubagentFactoryError",
    "build_defined_agent_and_conversation",
    "build_fork_agent_and_conversation",
    "build_skill_agent_and_conversation",
]
