"""Immutable public models for Git worktree isolation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class WorktreeError(RuntimeError):
    """A safe, user-displayable worktree lifecycle error."""


@dataclass(frozen=True, slots=True)
class Worktree:
    name: str
    path: str
    branch: str
    based_on: str
    head_commit: str
    created: datetime
    manual: bool


@dataclass(frozen=True, slots=True)
class WorktreeSession:
    original_cwd: str
    worktree_path: str
    worktree_name: str
    session_id: str


class ExitAction(StrEnum):
    KEEP = "keep"
    REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class ExitOptions:
    discard_changes: bool = False


@dataclass(frozen=True, slots=True)
class ExitReport:
    removed: bool
    path: str
    branch: str
    restore_cwd: str


@dataclass(frozen=True, slots=True)
class AutoCleanupReport:
    kept: bool
    path: str = ""
    branch: str = ""
    reason: str = ""


__all__ = [
    "AutoCleanupReport",
    "ExitAction",
    "ExitOptions",
    "ExitReport",
    "Worktree",
    "WorktreeError",
    "WorktreeSession",
]
