from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from codewright.worktree.git import _has_worktree_changes, _resolve_head_sha_from_fs, _run_git


def _repo(path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    (path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)
    return path


@pytest.mark.asyncio
async def test_git_helpers_detect_clean_dirty_and_new_commit(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    head = await _run_git(repo, "rev-parse", "HEAD")
    assert _resolve_head_sha_from_fs(repo) == head
    assert not await _has_worktree_changes(repo, head)
    (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    assert await _has_worktree_changes(repo, head)


def test_resolve_detached_head(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    subprocess.run(["git", "checkout", "--detach", "-q", head], cwd=repo, check=True)
    assert _resolve_head_sha_from_fs(repo) == head
