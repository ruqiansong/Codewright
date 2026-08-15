from __future__ import annotations

from pathlib import Path

from codewright.team.manager import Manager
from codewright.team.types import TeammateInfo


def manager(home: Path, project: Path) -> Manager:
    return Manager(
        home_dir=home,
        project_root=project,
        worktree_manager=None,
        task_manager=None,
        runtime_factory=None,
    )


async def test_create_suffix_load_filter_and_active_team(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    first_manager = manager(tmp_path, project)
    first = await first_manager.create("Demo Team")
    second = await first_manager.create("Demo Team")
    assert (first.slug, second.slug) == ("Demo-Team", "Demo-Team-2")
    assert first_manager.active_team == second
    assert manager(tmp_path, project).get(first.slug) is not None
    other = tmp_path / "other"
    other.mkdir()
    assert manager(tmp_path, other).list() == ()


async def test_member_names_are_scoped_to_team(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    instance = manager(tmp_path, project)
    first = await instance.create("First")
    second = await instance.create("Second")
    await instance.reserve_member(first.slug, TeammateInfo(name="alice"))
    await instance.reserve_member(second.slug, TeammateInfo(name="alice"))
    assert instance.resolve_member(first.slug, "alice") is not None
    assert instance.resolve_member(second.slug, "alice") is not None
