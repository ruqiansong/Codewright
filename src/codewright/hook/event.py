"""Lifecycle event names exposed to declarative Hook rules."""

from __future__ import annotations

from enum import StrEnum


class Event(StrEnum):
    """One stable point in the Codewright lifecycle."""

    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    SESSION_RESUME = "SessionResume"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    STOP = "Stop"
    PRE_USER_MESSAGE = "PreUserMessage"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    PRE_COMPACT = "PreCompact"
    POST_COMPACT = "PostCompact"
    NOTIFICATION = "Notification"


BLOCKING_EVENTS: frozenset[Event] = frozenset({Event.PRE_TOOL_USE, Event.USER_PROMPT_SUBMIT})


def is_blocking(event: Event) -> bool:
    """Return whether a synchronous Hook may block this event."""
    return event in BLOCKING_EVENTS


def parse_event(value: str) -> Event | None:
    """Parse one exact public event name without raising for unknown values."""
    try:
        return Event(value)
    except (TypeError, ValueError):
        return None


__all__ = ["BLOCKING_EVENTS", "Event", "is_blocking", "parse_event"]
