from __future__ import annotations

import pytest

from codewright.team.types import BackendType, MemberState, Team, TeammateInfo


def test_team_round_trip_is_strict_and_lead_is_not_a_member(tmp_path) -> None:
    member = TeammateInfo(name="alice", state=MemberState.IDLE)
    team = Team(
        name="Demo",
        slug="Demo",
        description="",
        project_root=str(tmp_path),
        backend=BackendType.IN_PROCESS,
        members=[member],
    )
    restored = Team.from_dict(team.to_dict(), config_dir=str(tmp_path / "team"))
    assert restored.to_dict() == team.to_dict()
    assert all(item.name != "lead" for item in restored.members)


def test_team_rejects_unknown_schema_and_enum(tmp_path) -> None:
    team = Team(name="Demo", slug="Demo", description="", project_root=str(tmp_path))
    payload = team.to_dict()
    payload["schema_version"] = 2
    with pytest.raises(ValueError, match="schema"):
        Team.from_dict(payload)
    payload = team.to_dict()
    payload["backend"] = "unknown"
    with pytest.raises(ValueError, match="backend"):
        Team.from_dict(payload)
