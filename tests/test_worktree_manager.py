from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from codewright.worktree import ExitAction, ExitOptions, Manager, WorktreeError


def _repo(path: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    (path / ".gitignore").write_text(
        ".codewright/worktrees/\n.codewright/worktree_session.json\n", encoding="utf-8"
    )
    (path / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=path, check=True)
    return path


def test_manager_rejects_non_repo_and_subdirectory(tmp_path: Path) -> None:
    with pytest.raises(WorktreeError):
        Manager(tmp_path)
    repo = _repo(tmp_path / "repo")
    child = repo / "child"
    child.mkdir()
    with pytest.raises(WorktreeError):
        Manager(child)


@pytest.mark.asyncio
async def test_create_recover_enter_exit_and_remove(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "repo")
    original = Path.cwd()
    manager = Manager(repo)
    item = await manager.create("team/alice", manual=True)
    assert Path(item.path).is_dir()
    assert item.branch == "worktree-team+alice"
    assert (Path(item.path) / ".git").is_file()
    assert await manager.create("team/alice", manual=True) == item
    session = await manager.enter(item.name)
    assert Path.cwd() == original
    assert manager.current_session() == session
    report = await manager.exit(item.name, ExitAction.KEEP)
    assert report.restore_cwd == str(repo.resolve())
    assert Path.cwd() == original
    recovered = Manager(repo)
    assert recovered.get(item.name) == item
    await recovered.remove(item.name)
    assert not Path(item.path).exists()


@pytest.mark.asyncio
async def test_change_protection_and_discard(tmp_path: Path) -> None:
    manager = Manager(_repo(tmp_path / "repo"))
    item = await manager.create("alice", manual=False)
    (Path(item.path) / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(WorktreeError):
        await manager.remove(item.name)
    assert Path(item.path).exists()
    await manager.remove(item.name, ExitOptions(discard_changes=True))
    assert not Path(item.path).exists()


@pytest.mark.asyncio
async def test_auto_cleanup_and_sweep(tmp_path: Path) -> None:
    manager = Manager(_repo(tmp_path / "repo"))
    clean = await manager.create("agent-a1234567", manual=False)
    report = await manager.auto_cleanup(clean.name)
    assert not report.kept
    manual = await manager.create("manual", manual=True)
    assert (await manager.auto_cleanup(manual.name)).kept
    assert await manager.sweep_stale(datetime.now(UTC) + timedelta(days=1)) == []
    await manager.remove(manual.name)
