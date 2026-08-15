from __future__ import annotations

import pytest

from codewright.agent.context import TeamExecutionContext
from codewright.agent.team_hook import TeamHook, TeamSpawnRequest
from codewright.tool import Result


class Hook:
    async def spawn_teammate(self, request: TeamSpawnRequest) -> Result:
        return Result(request.member_name)


def test_team_execution_context_and_hook_contract() -> None:
    context = TeamExecutionContext("demo", "alice", "task-1")
    assert context.team_slug == "demo"
    assert isinstance(Hook(), TeamHook)


def test_team_execution_context_rejects_empty_identity() -> None:
    with pytest.raises(ValueError):
        TeamExecutionContext("", "alice", "task-1")
