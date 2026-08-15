"""Task-local execution context exposed to tools invoked by an Agent."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codewright.conversation import Conversation

if TYPE_CHECKING:
    from codewright.agent import Agent


@dataclass(frozen=True, slots=True)
class TeamExecutionContext:
    """Stable Team identity attached to a teammate's tool executions."""

    team_slug: str
    member_name: str
    agent_id: str
    is_lead: bool = False

    def __post_init__(self) -> None:
        for field_name in ("team_slug", "member_name", "agent_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"{field_name} must be a non-empty trimmed string")
        if not isinstance(self.is_lead, bool):
            raise TypeError("is_lead must be a boolean")


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """The Agent and Conversation owning the current tool execution batch."""

    agent: Agent
    conversation: Conversation
    team: TeamExecutionContext | None = None

    def __post_init__(self) -> None:
        from codewright.agent import Agent

        if not isinstance(self.agent, Agent):
            raise TypeError("agent must be an Agent")
        if not isinstance(self.conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if self.team is not None and not isinstance(self.team, TeamExecutionContext):
            raise TypeError("team must be a TeamExecutionContext or None")


_EXECUTION_CONTEXT: ContextVar[ExecutionContext | None] = ContextVar(
    "codewright_execution_context", default=None
)


def bind_execution_context(context: ExecutionContext) -> Token[ExecutionContext | None]:
    """Bind an execution context in the current asynchronous task."""
    if not isinstance(context, ExecutionContext):
        raise TypeError("context must be an ExecutionContext")
    return _EXECUTION_CONTEXT.set(context)


def current_execution_context() -> ExecutionContext | None:
    """Return the active execution context, if any."""
    return _EXECUTION_CONTEXT.get()


def reset_execution_context(token: Token[ExecutionContext | None]) -> None:
    """Restore the execution context represented by a binding token."""
    _EXECUTION_CONTEXT.reset(token)


__all__ = [
    "ExecutionContext",
    "TeamExecutionContext",
    "bind_execution_context",
    "current_execution_context",
    "reset_execution_context",
]
