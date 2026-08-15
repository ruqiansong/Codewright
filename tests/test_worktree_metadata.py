from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codewright.worktree import Worktree, WorktreeSession
from codewright.worktree.metadata import load_metadata, metadata_path, save_metadata
from codewright.worktree.session import load_session, save_session


def test_session_null_and_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    save_session(path, None)
    assert path.read_text(encoding="utf-8") == "null"
    assert load_session(path) is None
    session = WorktreeSession("/repo", "/repo/w", "w", "id")
    save_session(path, session)
    assert load_session(path) == session


def test_metadata_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "worktrees"
    metadata = root / ".metadata"
    item = Worktree(
        "team/alice",
        str(root / "team+alice"),
        "worktree-team+alice",
        "HEAD",
        "a" * 40,
        datetime.now(UTC),
        False,
    )
    save_metadata(metadata, item)
    assert load_metadata(metadata_path(metadata, item.name), root) == item


def test_metadata_rejects_naive_datetime_and_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "worktrees"
    metadata = root / ".metadata"
    item = Worktree("a", str(root / "a"), "worktree-a", "HEAD", "a" * 40, datetime.now(), False)
    with pytest.raises(ValueError):
        save_metadata(metadata, item)
    metadata.mkdir(parents=True)
    path = metadata / "a.json"
    path.write_text(
        json.dumps(
            {
                "name": "a",
                "path": "/outside",
                "branch": "worktree-a",
                "based_on": "HEAD",
                "head_commit": "a" * 40,
                "created": datetime.now(UTC).isoformat(),
                "manual": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_metadata(path, root)
