"""Long-lived Agent Team collaboration primitives."""

from codewright.team.manager import Manager, TeamError
from codewright.team.types import (
    BackendType,
    DeleteReport,
    MemberState,
    RuntimeHandle,
    SpawnRequest,
    SpawnResult,
    Team,
    TeammateInfo,
)

__all__ = [
    "BackendType",
    "DeleteReport",
    "Manager",
    "MemberState",
    "RuntimeHandle",
    "SpawnRequest",
    "SpawnResult",
    "Team",
    "TeamError",
    "TeammateInfo",
]
