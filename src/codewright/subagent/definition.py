"""Validated definitions for built-in and locally configured subagents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from codewright.permission import Mode

DEFAULT_MAX_TURNS = 25
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,31}$")


class Source(StrEnum):
    """A subagent definition source without implicit numeric precedence."""

    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"
    PLUGIN = "plugin"


@dataclass(frozen=True, slots=True)
class Definition:
    """One immutable, fully validated subagent role definition."""

    name: str
    description: str
    tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    model: str = "inherit"
    max_turns: int = DEFAULT_MAX_TURNS
    permission_mode: Mode = Mode.DEFAULT
    dont_ask: bool = False
    background: bool = False
    plan_mode_required: bool = False
    system_prompt: str = ""
    file_path: str = ""
    source: Source = Source.BUILTIN
    isolation: str = ""

    def __post_init__(self) -> None:
        if self.name != "__fork__" and _NAME_PATTERN.fullmatch(self.name) is None:
            raise ValueError("subagent name must match ^[a-z][a-z0-9-]{0,31}$")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("subagent description must be a non-empty string")
        _validate_tool_names(self.tools, field_name="tools")
        _validate_tool_names(self.disallowed_tools, field_name="disallowed_tools")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("subagent model must be a non-empty string")
        if self.model != self.model.strip():
            raise ValueError("subagent model must be trimmed")
        if (
            not isinstance(self.max_turns, int)
            or isinstance(self.max_turns, bool)
            or self.max_turns <= 0
        ):
            raise ValueError("subagent max_turns must be a positive integer")
        if not isinstance(self.permission_mode, Mode):
            raise TypeError("subagent permission_mode must be a Mode")
        if not isinstance(self.dont_ask, bool):
            raise TypeError("subagent dont_ask must be a boolean")
        if not isinstance(self.background, bool):
            raise TypeError("subagent background must be a boolean")
        if not isinstance(self.plan_mode_required, bool):
            raise TypeError("subagent plan_mode_required must be a boolean")
        if not isinstance(self.system_prompt, str):
            raise TypeError("subagent system_prompt must be a string")
        if not self.is_fork() and not self.system_prompt.strip():
            raise ValueError("subagent system_prompt must not be empty")
        if not isinstance(self.file_path, str):
            raise TypeError("subagent file_path must be a string")
        if not isinstance(self.source, Source):
            raise TypeError("subagent source must be a Source")
        if self.isolation not in {"", "worktree"}:
            raise ValueError("subagent isolation must be empty or worktree")

    def is_fork(self) -> bool:
        """Return whether this internal definition represents a Fork launch."""
        return self.name == "__fork__"


def _validate_tool_names(values: tuple[str, ...], *, field_name: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"subagent {field_name} must be a tuple")
    if any(not isinstance(value, str) or not value or value != value.strip() for value in values):
        raise ValueError(f"subagent {field_name} must contain trimmed non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"subagent {field_name} must not contain duplicates")


__all__ = ["DEFAULT_MAX_TURNS", "Definition", "Source"]
