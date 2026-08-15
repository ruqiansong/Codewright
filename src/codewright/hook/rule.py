"""Validated in-memory data structures for lifecycle Hook rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from codewright.hook.event import Event
from codewright.permission.matcher import Matcher

type Payload = dict[str, Any]


class CombineMode(StrEnum):
    """How the atomic conditions in one rule are combined."""

    ALL_OF = "all_of"
    ANY_OF = "any_of"


class ActionType(StrEnum):
    """Supported Hook side-effect kinds."""

    SHELL = "shell"
    PROMPT = "prompt"
    HTTP = "http"
    SUBAGENT = "subagent"


@dataclass(frozen=True, slots=True)
class AtomCondition:
    field: str
    matcher: Matcher


@dataclass(frozen=True, slots=True)
class Condition:
    mode: CombineMode
    atoms: tuple[AtomCondition, ...]


@dataclass(frozen=True, slots=True)
class ShellAction:
    command: str


@dataclass(frozen=True, slots=True)
class PromptAction:
    text: str


@dataclass(frozen=True, slots=True)
class HttpAction:
    url: str
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)
    body: str | None = None


@dataclass(frozen=True, slots=True)
class SubagentAction:
    agent_name: str
    prompt: str


@dataclass(frozen=True, slots=True)
class Action:
    type: ActionType
    shell: ShellAction | None = None
    prompt: PromptAction | None = None
    http: HttpAction | None = None
    subagent: SubagentAction | None = None


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    event: Event
    action: Action
    condition: Condition | None = None
    only_once: bool = False
    asyncio_mode: bool = False
    timeout_s: float = 30.0
    source: str = ""


__all__ = [
    "Action",
    "ActionType",
    "AtomCondition",
    "CombineMode",
    "Condition",
    "HttpAction",
    "Payload",
    "PromptAction",
    "Rule",
    "ShellAction",
    "SubagentAction",
]
