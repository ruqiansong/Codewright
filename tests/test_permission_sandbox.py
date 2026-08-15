"""Tests for project-root containment and symlink resolution."""

from pathlib import Path

import pytest

from codewright.permission.sandbox import (
    eval_symlinks_or_ancestor,
    resolve_root,
    sandbox_ok,
)


def test_resolve_root_requires_existing_directory(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    assert resolve_root(root) == root.resolve()

    with pytest.raises(FileNotFoundError):
        resolve_root(tmp_path / "missing")

    file_path = tmp_path / "file.txt"
    file_path.write_text("content", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        resolve_root(file_path)


def test_sandbox_accepts_root_existing_and_new_nested_paths(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    existing = root / "existing.txt"
    existing.write_text("content", encoding="utf-8")

    assert sandbox_ok(root, "")
    assert sandbox_ok(root, "existing.txt")
    assert sandbox_ok(root, "new/deep/file.txt")


def test_sandbox_rejects_absolute_and_relative_escape(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    assert not sandbox_ok(root, str(outside))
    assert not sandbox_ok(root, "../outside.txt")


def test_sandbox_rejects_existing_and_dangling_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "outside-link").symlink_to(outside, target_is_directory=True)
    (root / "future-link").symlink_to(outside / "future", target_is_directory=True)

    assert not sandbox_ok(root, "outside-link/secret.txt")
    assert not sandbox_ok(root, "future-link/secret.txt")


def test_eval_symlinks_requires_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        eval_symlinks_or_ancestor(Path("relative/path"))
