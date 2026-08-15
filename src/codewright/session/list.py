"""Discover resumable Codewright sessions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from codewright.compact import parse_session_time

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SessionInfo:
    """Bounded metadata displayed by the session resume list."""

    id: str
    title: str
    modified_at: datetime
    model: str
    size: int
    dir: str


def list_sessions(sessions_dir: str) -> list[SessionInfo]:
    """Return valid current-format sessions ordered newest modification first."""
    if not isinstance(sessions_dir, str) or not sessions_dir.strip():
        raise ValueError("sessions_dir must be a non-empty string")
    root = Path(sessions_dir)
    if not root.is_dir():
        return []

    sessions: list[SessionInfo] = []
    try:
        children = tuple(root.iterdir())
    except OSError as error:
        logger.warning("Session directory could not be scanned error=%s", type(error).__name__)
        return []
    for directory in children:
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            parse_session_time(directory.name)
        except (TypeError, ValueError):
            continue
        path = directory / "conversation.jsonl"
        if not path.is_file() or path.is_symlink():
            continue
        try:
            stat = path.stat()
            title, model = _read_summary(path)
        except OSError as error:
            logger.warning(
                "Session metadata could not be read id=%s error=%s",
                directory.name,
                type(error).__name__,
            )
            continue
        if not title:
            continue
        sessions.append(
            SessionInfo(
                id=directory.name,
                title=_truncate_title(title),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                model=model,
                size=stat.st_size,
                dir=str(directory.resolve()),
            )
        )
    return sorted(sessions, key=lambda session: session.modified_at, reverse=True)


def _read_summary(path: Path) -> tuple[str, str]:
    title = ""
    model = ""
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(value, dict):
                continue
            raw_model = value.get("model")
            if not model and isinstance(raw_model, str):
                model = raw_model.strip()
            if value.get("role") == "user" and isinstance(value.get("content"), str):
                title = value["content"].strip()
                if title:
                    break
    return title, model or "unknown"


def _truncate_title(title: str) -> str:
    normalized = " ".join(title.split())
    return normalized if len(normalized) <= 50 else normalized[:49] + "…"
