"""Best-effort retention cleanup for current-format session directories."""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from codewright.compact import parse_session_time

logger = logging.getLogger(__name__)


def clean_expired(sessions_dir: str, max_age: timedelta) -> None:
    """Delete current-format session directories older than max_age."""
    if not isinstance(sessions_dir, str) or not sessions_dir.strip():
        raise ValueError("sessions_dir must be a non-empty string")
    if not isinstance(max_age, timedelta) or max_age < timedelta(0):
        raise ValueError("max_age must be a non-negative timedelta")
    root = Path(sessions_dir)
    if not root.is_dir():
        return
    now = datetime.now()
    try:
        children = tuple(root.iterdir())
    except OSError as error:
        logger.warning("Session cleanup scan failed error=%s", type(error).__name__)
        return
    for directory in children:
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            created = parse_session_time(directory.name)
        except (TypeError, ValueError):
            continue
        if now - created <= max_age:
            continue
        try:
            shutil.rmtree(directory)
        except OSError as error:
            logger.warning(
                "Expired session cleanup failed id=%s error=%s",
                directory.name,
                type(error).__name__,
            )
