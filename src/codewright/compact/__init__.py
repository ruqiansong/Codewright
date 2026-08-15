"""Context management primitives for long-running Codewright sessions."""

from codewright.compact.compact import ManageInput, ManageOutput, TriggerKind, manage_context
from codewright.compact.state import (
    CompactCircuitBreaker,
    ContentReplacementState,
    FileReadRecord,
    RecoveryState,
    SessionContext,
    new_session_context,
    open_session_context,
    parse_session_time,
)

__all__ = [
    "CompactCircuitBreaker",
    "ContentReplacementState",
    "FileReadRecord",
    "ManageInput",
    "ManageOutput",
    "RecoveryState",
    "SessionContext",
    "TriggerKind",
    "manage_context",
    "new_session_context",
    "open_session_context",
    "parse_session_time",
]
