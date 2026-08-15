from __future__ import annotations

import json
from pathlib import Path

from codewright.task import Manager as TaskManager
from codewright.task.tools import SendMessageTool, TaskGetTool, TaskListTool, TaskStopTool
from codewright.team.manager import Manager
from codewright.team.tools import register_team_tools
from codewright.tool import Registry


def team_manager(tmp_path: Path) -> Manager:
    project = tmp_path / "project"
    project.mkdir()
    return Manager(
        home_dir=tmp_path,
        project_root=project,
        worktree_manager=None,
        task_manager=None,
        runtime_factory=None,
    )


async def test_seven_team_tools_register_without_legacy_name_collisions(tmp_path: Path) -> None:
    registry = Registry()
    task_manager = TaskManager()
    for tool in (
        TaskListTool(task_manager),
        TaskGetTool(task_manager),
        TaskStopTool(task_manager),
        SendMessageTool(task_manager),
    ):
        registry.register(tool)
    names = register_team_tools(registry, team_manager(tmp_path))

    assert names == (
        "TeamCreate",
        "TeamDelete",
        "TeamTaskCreate",
        "TeamTaskGet",
        "TeamTaskList",
        "TeamTaskUpdate",
        "TeamSendMessage",
    )
    assert all(
        definition.input_schema["additionalProperties"] is False
        for definition in registry.definitions()
    )
    await task_manager.aclose()


async def test_team_task_tools_delegate_to_active_team_store(tmp_path: Path) -> None:
    registry = Registry()
    manager = team_manager(tmp_path)
    register_team_tools(registry, manager)
    created = await registry.execute(
        "TeamCreate", json.dumps({"team_name": "Demo", "description": ""})
    )
    assert not created.is_error
    task = await registry.execute("TeamTaskCreate", json.dumps({"title": "Build"}))
    task_id = json.loads(task.content)["task"]["id"]
    listed = await registry.execute("TeamTaskList", "{}")
    assert json.loads(listed.content)["tasks"][0]["id"] == task_id
