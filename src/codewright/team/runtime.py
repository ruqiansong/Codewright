"""Session-bound teammate runtime construction and recovery."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codewright.compact import new_session_context
from codewright.conversation import Conversation
from codewright.llm import MessageRole, Provider
from codewright.session import Writer, load_session
from codewright.team.types import TeammateInfo

if TYPE_CHECKING:
    from codewright.agent import Agent

type BuildResult = tuple[Agent, Conversation, Provider | None]
type RuntimeBuilder = Callable[..., BuildResult]


@dataclass(slots=True)
class TeammateRuntime:
    agent: Agent
    conversation: Conversation
    initial_prompt: str
    description: str
    writer: Writer
    owned_provider: Provider | None = None

    async def aclose(self) -> None:
        self.writer.close()
        if self.owned_provider is None:
            return
        close = getattr(self.owned_provider, "close", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result


class TeammateRuntimeFactory:
    """Bind factory-created conversations to their own durable Team Writer."""

    def __init__(self, project_root: str, builder: RuntimeBuilder) -> None:
        if not callable(builder):
            raise TypeError("builder must be callable")
        self.project_root = project_root
        self._builder = builder

    def create(
        self,
        *,
        initial_prompt: str,
        description: str,
        model: str = "inherit",
        **builder_arguments: object,
    ) -> TeammateRuntime:
        agent, source, provider = self._builder(initial_prompt=initial_prompt, **builder_arguments)
        context = new_session_context(self.project_root)
        writer = Writer(context.session_dir, model)
        messages = list(source.messages())
        non_system = [message for message in messages if message.role is not MessageRole.SYSTEM]
        writer.append_all(non_system)
        conversation = Conversation.from_messages(
            messages[0].content,
            non_system,
            on_append=writer.on_append,
            on_replace=writer.on_replace,
        )
        return TeammateRuntime(
            agent,
            conversation,
            initial_prompt,
            description,
            writer,
            provider,
        )

    def resume(
        self,
        info: TeammateInfo,
        *,
        system_prompt: str,
        description: str,
        **builder_arguments: object,
    ) -> TeammateRuntime:
        loaded = load_session(info.session_dir)
        agent, _, provider = self._builder(initial_prompt="", **builder_arguments)
        writer = Writer.open_existing(info.session_dir, loaded.model)
        conversation = Conversation.from_messages(
            system_prompt,
            loaded.messages,
            on_append=writer.on_append,
            on_replace=writer.on_replace,
        )
        return TeammateRuntime(agent, conversation, "resume", description, writer, provider)


__all__ = ["TeammateRuntime", "TeammateRuntimeFactory"]
