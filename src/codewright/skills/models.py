"""Validated Skill definitions and session-scoped activation state."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

type SkillMode = Literal["inline", "fork"]
type SkillContext = Literal["full", "recent", "none"]


class SkillSource(StrEnum):
    """Supported precedence layers for locally installed Skills."""

    PROJECT = "project"
    USER = "user"


@dataclass(frozen=True, slots=True)
class SkillDef:
    """One fully parsed Skill and its selected source metadata."""

    name: str
    description: str
    prompt_body: str
    mode: SkillMode
    model: str | None
    context: SkillContext
    source_path: Path
    source_dir: Path
    is_directory: bool
    source: SkillSource


@dataclass(frozen=True, slots=True)
class ActiveEntry:
    """One Skill body pinned to the current session environment."""

    name: str
    body: str
    source_dir: Path

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.body, str) or not self.body.strip():
            raise ValueError("body must be a non-empty string")
        if not isinstance(self.source_dir, Path) or not self.source_dir.is_absolute():
            raise ValueError("source_dir must be an absolute Path")


class ActiveSkills:
    """Maintain an ordered, thread-safe snapshot of activated Skills."""

    def __init__(self) -> None:
        self._entries: dict[str, ActiveEntry] = {}
        self._lock = threading.RLock()

    def activate(self, name: str, body: str, source_dir: Path) -> None:
        """Activate a Skill, replacing its content without changing its position."""
        entry = ActiveEntry(name, body, source_dir)
        with self._lock:
            self._entries[name] = entry

    def clear(self) -> None:
        """Remove all session-scoped Skill activations."""
        with self._lock:
            self._entries.clear()

    def snapshot(self) -> tuple[ActiveEntry, ...]:
        """Return an immutable activation-order snapshot."""
        with self._lock:
            return tuple(self._entries.values())

    def names(self) -> tuple[str, ...]:
        """Return activated names in activation order."""
        with self._lock:
            return tuple(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


__all__ = [
    "ActiveEntry",
    "ActiveSkills",
    "SkillContext",
    "SkillDef",
    "SkillMode",
    "SkillSource",
]
