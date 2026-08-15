"""Safe Git worktree isolation."""

from codewright.worktree.manager import Manager, random_agent_name
from codewright.worktree.models import (
    AutoCleanupReport,
    ExitAction,
    ExitOptions,
    ExitReport,
    Worktree,
    WorktreeError,
    WorktreeSession,
)
from codewright.worktree.slug import flat_slug, validate_slug

__all__ = [
    "AutoCleanupReport",
    "ExitAction",
    "ExitOptions",
    "ExitReport",
    "Manager",
    "Worktree",
    "WorktreeError",
    "WorktreeSession",
    "flat_slug",
    "random_agent_name",
    "validate_slug",
]
