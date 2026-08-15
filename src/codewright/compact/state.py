"""Thread-safe session state used by context management."""

from __future__ import annotations

import copy
import hashlib
import logging
import random
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from codewright.compact.const import MAX_CONSECUTIVE_AUTO_COMPACT_FAILURES

logger = logging.getLogger(__name__)

type ReplacementDecision = tuple[str, str]


@dataclass(frozen=True, slots=True)
class SessionContext:
    """Paths and identity shared by one in-process conversation session."""

    session_id: str
    spill_dir: str
    session_dir: str = ""


_SESSION_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


def _new_session_id() -> str:
    """Return a process-local session identifier suitable for a directory name."""
    try:
        random_part = secrets.token_hex(2)
    except Exception:
        logger.warning("Secure session id generation failed; using a local fallback")
        random_part = random.Random(time.time_ns()).randbytes(2).hex()
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{random_part}"


def new_session_context(workspace: str) -> SessionContext:
    """Create session metadata without creating its spill directory eagerly."""
    if not isinstance(workspace, str):
        raise TypeError("workspace must be a string")
    session_id = _new_session_id()
    session_dir = Path(workspace).resolve() / ".codewright" / "sessions" / session_id
    spill_dir = session_dir / "tool-results"
    return SessionContext(
        session_id=session_id,
        spill_dir=str(spill_dir),
        session_dir=str(session_dir),
    )


def parse_session_time(session_id: str) -> datetime:
    """Parse a current-format session identifier into a naive local datetime."""
    if not isinstance(session_id, str):
        raise TypeError("session_id must be a string")
    if _SESSION_ID_PATTERN.fullmatch(session_id) is None:
        raise ValueError("session_id must match YYYYMMDD-HHMMSS-xxxx")
    try:
        return datetime.strptime(session_id[:15], "%Y%m%d-%H%M%S")
    except ValueError as error:
        raise ValueError("session_id contains an invalid timestamp") from error


def open_session_context(workspace: str, session_id: str) -> SessionContext:
    """Return context for an existing, contained current-format session directory."""
    if not isinstance(workspace, str):
        raise TypeError("workspace must be a string")
    parse_session_time(session_id)
    sessions_root = (Path(workspace).resolve() / ".codewright" / "sessions").resolve()
    session_dir = (sessions_root / session_id).resolve()
    if session_dir.parent != sessions_root:
        raise ValueError("session directory escapes the sessions root")
    if not session_dir.is_dir():
        raise FileNotFoundError(f"session directory does not exist: {session_id}")
    return SessionContext(
        session_id=session_id,
        spill_dir=str(session_dir / "tool-results"),
        session_dir=str(session_dir),
    )


class ContentReplacementState:
    """Freeze keep/offload decisions for tool results within one session."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._seen_ids: set[str] = set()
        self._replacements: dict[str, str] = {}
        self._content_digests: dict[str, str] = {}

    def decide_once(
        self,
        tool_use_id: str,
        original: str,
        decide: Callable[[], ReplacementDecision],
    ) -> str:
        """Atomically reuse or record one keep/offload decision."""
        if not isinstance(tool_use_id, str) or not tool_use_id.strip():
            raise ValueError("tool_use_id must be a non-empty string")
        if not isinstance(original, str):
            raise TypeError("original must be a string")
        if not callable(decide):
            raise TypeError("decide must be callable")

        digest = hashlib.sha256(original.encode("utf-8")).hexdigest()
        with self._lock:
            if tool_use_id in self._seen_ids:
                replacement = self._replacements.get(tool_use_id)
                if replacement is not None and original == replacement:
                    return replacement
                if self._content_digests[tool_use_id] != digest:
                    logger.warning("Duplicate tool result id carried different content")
                    return original
                return replacement or original

            decision, preview = decide()
            if decision == "skip":
                return original
            if decision == "kept":
                self._seen_ids.add(tool_use_id)
                self._content_digests[tool_use_id] = digest
                return original
            if decision == "replaced":
                if not isinstance(preview, str) or not preview:
                    raise ValueError("a replacement decision requires a non-empty preview")
                self._seen_ids.add(tool_use_id)
                self._content_digests[tool_use_id] = digest
                self._replacements[tool_use_id] = preview
                return preview
            raise ValueError(f"unsupported replacement decision: {decision!r}")


class CompactCircuitBreaker:
    """Track consecutive automatic compaction failures."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._consecutive_failures = 0

    def record_success(self) -> None:
        """Reset the failure count after a successful automatic compaction."""
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Record one completed automatic compaction failure."""
        with self._lock:
            self._consecutive_failures += 1

    def tripped(self) -> bool:
        """Return whether automatic compaction is currently disabled."""
        with self._lock:
            return self._consecutive_failures >= MAX_CONSECUTIVE_AUTO_COMPACT_FAILURES


@dataclass(frozen=True, slots=True)
class FileReadRecord:
    """Latest clean content observed for one successfully read file."""

    path: str
    content: str
    timestamp: datetime


class RecoveryState:
    """Keep bounded recovery inputs without exposing mutable internal state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._files: dict[str, FileReadRecord] = {}

    def record_file(self, path: str, content: str) -> None:
        """Record the latest clean content for an absolute file path."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        normalized_path = str(Path(path).resolve())
        record = FileReadRecord(normalized_path, content, datetime.now(UTC))
        with self._lock:
            self._files[normalized_path] = record

    def snapshot(self) -> list[FileReadRecord]:
        """Return detached file records ordered from newest to oldest."""
        with self._lock:
            records = [copy.copy(record) for record in self._files.values()]
        return sorted(records, key=lambda record: record.timestamp, reverse=True)
