"""Domain models for vendor- and UI-neutral slash commands."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codewright.command.ui import UI


type Handler = Callable[["UI", str], Awaitable[None]]
type CommandSource = Literal["builtin", "skill"]


class Kind(StrEnum):
    """Observable execution category for one slash command."""

    LOCAL = "local"
    UI = "ui"
    PROMPT = "prompt"


@dataclass(frozen=True, slots=True)
class Command:
    """One validated slash command definition without its leading slash."""

    name: str
    description: str
    kind: Kind
    handler: Handler
    aliases: tuple[str, ...] = ()
    hidden: bool = False
    accepts_args: bool = False
    source: CommandSource = "builtin"

    def __post_init__(self) -> None:
        _validate_identifier(self.name, "name")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if self.description != self.description.strip() or "\n" in self.description:
            raise ValueError("description must be a single trimmed line")
        if not isinstance(self.kind, Kind):
            raise TypeError("kind must be a Kind")
        if not callable(self.handler):
            raise TypeError("handler must be callable")
        if not isinstance(self.aliases, tuple):
            raise TypeError("aliases must be a tuple")
        for alias in self.aliases:
            _validate_identifier(alias, "alias")
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("aliases must not contain duplicates")
        if self.name in self.aliases:
            raise ValueError("an alias must not duplicate the command name")
        if not isinstance(self.hidden, bool):
            raise TypeError("hidden must be a boolean")
        if not isinstance(self.accepts_args, bool):
            raise TypeError("accepts_args must be a boolean")
        if not isinstance(self.source, str) or self.source not in {"builtin", "skill"}:
            raise ValueError("source must be builtin or skill")


def _validate_identifier(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.lower() or value != value.strip():
        raise ValueError(f"{field_name} must be lowercase and trimmed")
    if value.startswith("/") or any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must not contain a slash prefix or whitespace")


__all__ = ["Command", "CommandSource", "Handler", "Kind"]
