from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from codewright.agent.team_hook import TeamSpawnRequest
from codewright.team.manager import Manager
from codewright.team.spawn import Spawner
from codewright.team.types import BackendType, SpawnResult


@dataclass
class Worktree:
    name: str
    path: str
    branch: str


class Worktrees:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls = []
        self.removed = []

    async def create(self, name, base_ref="HEAD", manual=False):
        self.calls.append((name, base_ref, manual))
        return Worktree(name, str(self.root / "wt"), "real-branch")

    async def remove(self, name, options):
        self.removed.append((name, options))


class Runtime:
    def __init__(self, root: Path) -> None:
        path = root / "sessions" / "20260101-000000-abcd" / "conversation.jsonl"
        self.writer = SimpleNamespace(path=path)
        self.closed = False

    async def aclose(self):
        self.closed = True


class Runtimes:
    def __init__(self, root: Path) -> None:
        self.runtime = Runtime(root)

    def create(self, **kwargs):
        del kwargs
        return self.runtime


class Backend:
    type = BackendType.IN_PROCESS

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    async def spawn(self, request):
        if self.fail:
            raise RuntimeError("failed")
        return SpawnResult("task-7", runtime_task_id="task-7")

    async def kill(self, handle):
        del handle


async def test_spawn_uses_relative_team_worktree_and_persists_real_values(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    worktrees = Worktrees(tmp_path)
    runtimes = Runtimes(tmp_path)
    manager = Manager(
        home_dir=tmp_path,
        project_root=project,
        worktree_manager=worktrees,
        task_manager=None,
        runtime_factory=runtimes,
    )
    team = await manager.create("Demo")
    spawner = Spawner(manager, lambda team: Backend())
    result = await spawner.spawn_teammate(
        TeamSpawnRequest("Demo", "alice", "work", "description")
    )

    assert not result.is_error
    assert worktrees.calls == [("team-Demo/alice", "HEAD", True)]
    member = manager.resolve_member(team.slug, "alice")
    assert member is not None
    assert (member.agent_id, member.branch) == ("task-7", "real-branch")


async def test_spawn_failure_compensates_worktree_and_marks_member_failed(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    worktrees = Worktrees(tmp_path)
    manager = Manager(
        home_dir=tmp_path,
        project_root=project,
        worktree_manager=worktrees,
        task_manager=None,
        runtime_factory=Runtimes(tmp_path),
    )
    team = await manager.create("Demo")
    result = await Spawner(manager, lambda team: Backend(fail=True)).spawn_teammate(
        TeamSpawnRequest("Demo", "alice", "work", "description")
    )

    assert result.error_code == "team_spawn_failed"
    assert worktrees.removed[0][0] == "team-Demo/alice"
    assert manager.resolve_member(team.slug, "alice").state.value == "failed"  # type: ignore[union-attr]
