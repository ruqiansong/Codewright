from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codewright.worktree import Worktree, flat_slug, validate_slug
from codewright.worktree.slug import contained_child


@pytest.mark.parametrize("name", ["alice", "team/alice", "v1.0", "a_b"])
def test_validate_slug_accepts_safe_names(name: str) -> None:
    assert validate_slug(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "a" * 65,
        "/a",
        "a/",
        "a//b",
        ".",
        "..",
        "a..b",
        "x.",
        "x.lock",
        ".metadata",
        ".METADATA",
        "a b",
        "a;b",
        "a+b",
    ],
)
def test_validate_slug_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError):
        validate_slug(name)


def test_flat_slug_and_containment(tmp_path: Path) -> None:
    assert flat_slug("team/alice") == "team+alice"
    assert contained_child(tmp_path, "team/alice") == tmp_path / "team+alice"


def test_worktree_is_frozen_and_slotted() -> None:
    item = Worktree("a", "/tmp/a", "worktree-a", "HEAD", "0" * 40, datetime.now(UTC), True)
    with pytest.raises(FrozenInstanceError):
        item.name = "b"  # type: ignore[misc]
    assert not hasattr(item, "__dict__")
