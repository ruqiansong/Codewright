"""Long-lived context-management state for one Codewright session."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from codewright.compact import (
    CompactCircuitBreaker,
    ContentReplacementState,
    RecoveryState,
    SessionContext,
)
from codewright.skills.models import ActiveSkills


@dataclass(slots=True)
class SessionRuntime:
    """State shared by every Agent turn within one process session."""

    replacement: ContentReplacementState
    recovery: RecoveryState
    auto_tracking: CompactCircuitBreaker
    session: SessionContext
    active_skills: ActiveSkills = field(default_factory=ActiveSkills)
    context_window: int = 200_000
    usage_anchor: int = 0
    anchor_msg_len: int = 0
    turn_count: int = 0
    pending_reminders: list[str] = field(default_factory=list)
    hook_once_fired: set[str] = field(default_factory=set)
    session_end_emitted: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        for field_name in ("context_window", "usage_anchor", "anchor_msg_len", "turn_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.context_window == 0:
            raise ValueError("context_window must be positive")

    def anchor_snapshot(self) -> tuple[int, int, int]:
        """Return a consistent usage anchor and context-window snapshot."""
        with self._lock:
            return self.usage_anchor, self.anchor_msg_len, self.context_window

    def update_anchor(self, total_tokens: int, message_count: int) -> None:
        """Atomically replace the previous main-request usage anchor."""
        if total_tokens < 0 or message_count < 0:
            raise ValueError("anchor values must not be negative")
        with self._lock:
            self.usage_anchor = total_tokens
            self.anchor_msg_len = message_count

    def reset_anchor(self) -> None:
        """Invalidate usage accounting after history has been rewritten."""
        with self._lock:
            self.usage_anchor = 0
            self.anchor_msg_len = 0

    def increment_turn_count(self) -> int:
        """Increment and return the process-local completed-turn count."""
        with self._lock:
            self.turn_count += 1
            return self.turn_count

    def append_reminders(self, reminders: list[str]) -> None:
        """Append non-empty Hook reminders in dispatch order."""
        if not isinstance(reminders, list) or not all(isinstance(item, str) for item in reminders):
            raise TypeError("reminders must be a list of strings")
        with self._lock:
            self.pending_reminders.extend(item for item in reminders if item)

    def take_reminders(self) -> list[str]:
        """Atomically consume all reminders pending for the next request."""
        with self._lock:
            reminders = list(self.pending_reminders)
            self.pending_reminders.clear()
            return reminders

    def claim_hook_once(self, name: str) -> bool:
        """Atomically claim one only-once Hook for the current session."""
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        with self._lock:
            if name in self.hook_once_fired:
                return False
            self.hook_once_fired.add(name)
            return True

    def claim_session_end(self) -> bool:
        """Atomically claim the single SessionEnd emission for this session."""
        with self._lock:
            if self.session_end_emitted:
                return False
            self.session_end_emitted = True
            return True

    def reset_for_new_session(self, session: SessionContext) -> None:
        """Atomically replace all session-scoped state while preserving capacity."""
        if not isinstance(session, SessionContext):
            raise TypeError("session must be a SessionContext")
        with self._lock:
            self.replacement = ContentReplacementState()
            self.recovery = RecoveryState()
            self.auto_tracking = CompactCircuitBreaker()
            self.session = session
            self.active_skills = ActiveSkills()
            self.usage_anchor = 0
            self.anchor_msg_len = 0
            self.turn_count = 0
            self.pending_reminders.clear()
            self.hook_once_fired.clear()
            self.session_end_emitted = False
