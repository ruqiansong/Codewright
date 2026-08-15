from __future__ import annotations

import json
from pathlib import Path

import pytest

from codewright.team.persistence import (
    atomic_write_json,
    contained_team_dir,
    sanitize_team_name,
)


@pytest.mark.parametrize("value", ["", "  ", ".", "..", "CON", "config", "___"])
def test_sanitize_rejects_empty_and_reserved_names(value: str) -> None:
    with pytest.raises(ValueError):
        sanitize_team_name(value)


def test_sanitize_produces_bounded_safe_slug() -> None:
    assert sanitize_team_name("../Team: alpha") == "Team-alpha"
    assert sanitize_team_name("a" * 80) == "a" * 48


def test_contained_team_dir_rejects_escape_symlink(tmp_path: Path) -> None:
    root = tmp_path / "teams"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escaped").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        contained_team_dir(root, "escaped")


def test_atomic_write_json_replaces_complete_document(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    atomic_write_json(path, {"generation": 1})
    atomic_write_json(path, {"generation": 2})
    assert json.loads(path.read_text()) == {"generation": 2}
    assert not list(tmp_path.glob("*.tmp"))
