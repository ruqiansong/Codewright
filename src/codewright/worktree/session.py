"""Atomic worktree session persistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from codewright.worktree.models import WorktreeSession


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_session(path: Path, session: WorktreeSession | None) -> None:
    value = None
    if session is not None:
        value = {
            "original_cwd": session.original_cwd,
            "session_id": session.session_id,
            "worktree_name": session.worktree_name,
            "worktree_path": session.worktree_path,
        }
    _atomic_json(path, value)


def load_session(path: Path) -> WorktreeSession | None:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "original_cwd",
        "session_id",
        "worktree_name",
        "worktree_path",
    }:
        raise ValueError("Worktree session 数据格式无效")
    if any(not isinstance(value[key], str) or not value[key] for key in value):
        raise ValueError("Worktree session 字段无效")
    return WorktreeSession(**value)


__all__ = ["load_session", "save_session"]
