"""Dependency-free permission types shared by the engine, Agent, and TUI."""

from enum import IntEnum


class Mode(IntEnum):
    """Permission policy applied when no explicit rule matches."""

    DEFAULT = 0
    ACCEPT_EDITS = 1
    PLAN = 2
    BYPASS = 3

    def __str__(self) -> str:
        return _MODE_NAMES[self]


_MODE_NAMES: dict[Mode, str] = {
    Mode.DEFAULT: "default",
    Mode.ACCEPT_EDITS: "acceptEdits",
    Mode.PLAN: "plan",
    Mode.BYPASS: "bypassPermissions",
}
_MODES_BY_NAME = {name.casefold(): mode for mode, name in _MODE_NAMES.items()}


def parse_mode(value: str) -> tuple[Mode, bool]:
    """Parse a public mode name, falling back safely for unknown values."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    mode = _MODES_BY_NAME.get(value.strip().casefold())
    if mode is None:
        return Mode.DEFAULT, False
    return mode, True


class Decision(IntEnum):
    """Intermediate permission-engine decision."""

    ALLOW = 0
    DENY = 1
    ASK = 2


class Category(IntEnum):
    """Security category assigned to a known built-in tool."""

    READ = 0
    WRITE = 1
    EXEC = 2


class Outcome(IntEnum):
    """One user response to an interactive approval request."""

    DENY_ONCE = 0
    ALLOW_ONCE = 1
    ALLOW_FOREVER = 2


class PermissionSetupError(RuntimeError):
    """Raised when a safe permission-engine foundation cannot be established."""


__all__ = [
    "Category",
    "Decision",
    "Mode",
    "Outcome",
    "PermissionSetupError",
    "parse_mode",
]
