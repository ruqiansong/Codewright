"""Dependency-inversion contract between Agent delegation and Team runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from codewright.tool import Result


@dataclass(frozen=True, slots=True)
class TeamSpawnRequest:
    team_name: str
    member_name: str
    prompt: str
    description: str
    subagent_type: str = ""
    model: str | None = None
    plan_mode_required: bool = False


@runtime_checkable
class TeamHook(Protocol):
    async def spawn_teammate(self, request: TeamSpawnRequest) -> Result:
        """Create and launch one teammate for the active Lead."""
        ...


__all__ = ["TeamHook", "TeamSpawnRequest"]
