"""Inline and isolated fork execution for validated Skills."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codewright.agent.factory import (
    SubagentFactoryError,
    build_skill_agent_and_conversation,
)
from codewright.config import ProviderConfig
from codewright.conversation import Conversation
from codewright.llm import LLMError, Message, MessageRole, Provider, TokenUsage
from codewright.llm.factory import create_provider
from codewright.permission import Outcome
from codewright.skills.models import SkillDef
from codewright.skills.parser import substitute_arguments

if TYPE_CHECKING:
    from codewright.agent import Agent, ApprovalRequest

logger = logging.getLogger(__name__)

RECENT_MESSAGE_LIMIT = 5
PROVIDER_CLOSE_TIMEOUT = 5.0
type ProviderFactory = Callable[[ProviderConfig], Provider]
type ApprovalHandler = Callable[["ApprovalRequest"], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ForkResult:
    """Final fork response and usage accumulated across every child iteration."""

    text: str
    usage: TokenUsage


class SkillExecutionError(RuntimeError):
    """A Skill failure containing only a message safe to display."""

    def __init__(self, safe_message: str) -> None:
        if not isinstance(safe_message, str) or not safe_message.strip():
            raise ValueError("safe_message must be a non-empty string")
        self.safe_message = safe_message.strip()
        super().__init__(self.safe_message)


class SkillExecutor:
    """Execute inline Skills or isolated fork Skills using Agent dependencies."""

    def __init__(
        self,
        agent: Agent,
        provider_configs: Sequence[ProviderConfig] = (),
        *,
        provider_factory: ProviderFactory = create_provider,
        stream: bool = True,
    ) -> None:
        if not callable(provider_factory):
            raise TypeError("provider_factory must be callable")
        if not isinstance(stream, bool):
            raise TypeError("stream must be a boolean")
        configs = tuple(provider_configs)
        if not all(isinstance(config, ProviderConfig) for config in configs):
            raise TypeError("provider_configs must contain only ProviderConfig values")
        if len({config.name for config in configs}) != len(configs):
            raise ValueError("provider config names must be unique")
        self._agent = agent
        self._provider_configs = {config.name: config for config in configs}
        self._provider_factory = provider_factory
        self._stream = stream

    def execute_inline(self, skill: SkillDef, args: str) -> str:
        """Render arguments, activate the Skill, and return its complete body."""
        _validate_execution(skill, args, expected_mode="inline")
        rendered = substitute_arguments(skill.prompt_body, args)
        self._agent.activate_skill(skill.name, rendered, skill.source_dir)
        return rendered

    async def execute_fork(
        self,
        skill: SkillDef,
        args: str,
        main_conversation: Conversation,
        cancel_event: asyncio.Event | None = None,
        *,
        approval_handler: ApprovalHandler | None = None,
    ) -> ForkResult:
        """Run a fork Skill in an isolated Agent and Conversation."""
        _validate_execution(skill, args, expected_mode="fork")
        if not isinstance(main_conversation, Conversation):
            raise TypeError("main_conversation must be a Conversation")
        if cancel_event is not None and not isinstance(cancel_event, asyncio.Event):
            raise TypeError("cancel_event must be an asyncio.Event or None")
        if cancel_event is not None and cancel_event.is_set():
            raise SkillExecutionError("Skill execution was cancelled.")

        async def upgrade(request: ApprovalRequest) -> Outcome:
            if approval_handler is None:
                return Outcome.DENY_ONCE
            await approval_handler(request)
            if request.respond.cancelled():
                return Outcome.DENY_ONCE
            return request.respond.result()

        try:
            child_agent, child_conversation, owned_provider, config = (
                build_skill_agent_and_conversation(
                    self._agent,
                    skill,
                    args,
                    main_conversation,
                    tuple(self._provider_configs.values()),
                    approval_upgrader=upgrade,
                    provider_factory=self._provider_factory,
                )
            )
        except SubagentFactoryError as error:
            raise SkillExecutionError(error.safe_message) from None
        try:
            result = await child_agent.run_to_completion(
                child_conversation,
                "",
                stream=self._stream if config is None else config.stream,
                cancel_event=cancel_event,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise SkillExecutionError("Skill execution was cancelled.")
            return ForkResult(result.text, result.usage)
        except LLMError as error:
            raise SkillExecutionError(error.safe_message) from None
        finally:
            if owned_provider is not None:
                await _close_owned_provider(owned_provider)


def _validate_execution(skill: SkillDef, args: str, *, expected_mode: str) -> None:
    if not isinstance(skill, SkillDef):
        raise TypeError("skill must be a SkillDef")
    if not isinstance(args, str):
        raise TypeError("args must be a string")
    if skill.mode != expected_mode:
        raise SkillExecutionError(f"Skill is not configured for {expected_mode} execution.")


def fork_skill_conversation(
    skill: SkillDef,
    args: str,
    main_conversation: Conversation,
) -> Conversation:
    messages = main_conversation.messages()
    system_prompt = messages[0].content
    copied: tuple[Message, ...]
    if skill.context == "none":
        copied = ()
    elif skill.context == "recent":
        ordinary = tuple(
            message
            for message in messages[1:]
            if message.role in {MessageRole.USER, MessageRole.ASSISTANT}
            and not message.tool_calls
            and bool(message.content.strip())
        )
        copied = ordinary[-RECENT_MESSAGE_LIMIT:]
    else:
        copied = messages[1:]
    conversation = Conversation.from_messages(system_prompt, copied)
    conversation.add_user(substitute_arguments(skill.prompt_body, args))
    return conversation


async def _close_owned_provider(provider: Provider) -> None:
    close = getattr(provider, "close", None)
    if close is None:
        return
    try:
        result = close()
        if result is not None:
            await asyncio.wait_for(result, timeout=PROVIDER_CLOSE_TIMEOUT)
    except Exception as error:
        logger.warning("Fork provider close failed error=%s", type(error).__name__)


__all__ = [
    "ForkResult",
    "PROVIDER_CLOSE_TIMEOUT",
    "RECENT_MESSAGE_LIMIT",
    "SkillExecutionError",
    "SkillExecutor",
    "fork_skill_conversation",
]
