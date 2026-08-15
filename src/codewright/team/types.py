"""Strict domain models for persisted and running Agent Teams."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_PERMISSION_MODES = {"default", "acceptEdits", "plan", "bypassPermissions"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _strict_dict(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid {label} fields")
    return value


def _text(value: object, field_name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"{field_name} must be a trimmed string")
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    return value


def _optional_text(value: object, field_name: str) -> str:
    return _text(value, field_name, allow_empty=True)


def _timestamp(value: object, field_name: str) -> str:
    result = _text(value, field_name)
    try:
        parsed = datetime.fromisoformat(result.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a UTC offset")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} must be in UTC")
    return result


class BackendType(StrEnum):
    IN_PROCESS = "in-process"
    TMUX = "tmux"
    ITERM2 = "iterm2"


class MemberState(StrEnum):
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class TeammateInfo:
    name: str
    agent_id: str = ""
    agent_type: str = "general-purpose"
    model: str = "inherit"
    worktree_name: str = ""
    worktree_path: str = ""
    branch: str = ""
    backend_type: BackendType = BackendType.IN_PROCESS
    pane_id: str = ""
    runtime_task_id: str = ""
    state: MemberState = MemberState.STARTING
    plan_mode_required: bool = False
    pending_plan_request_id: str = ""
    session_dir: str = ""
    last_error: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self.name = _text(self.name, "member name")
        for name in (
            "agent_id",
            "agent_type",
            "model",
            "worktree_name",
            "worktree_path",
            "branch",
            "pane_id",
            "runtime_task_id",
            "pending_plan_request_id",
            "session_dir",
            "last_error",
        ):
            setattr(self, name, _optional_text(getattr(self, name), name))
        if not self.agent_type or not self.model:
            raise ValueError("agent_type and model must not be empty")
        if not isinstance(self.backend_type, BackendType):
            raise TypeError("backend_type must be a BackendType")
        if not isinstance(self.state, MemberState):
            raise TypeError("state must be a MemberState")
        if not isinstance(self.plan_mode_required, bool):
            raise TypeError("plan_mode_required must be a boolean")
        self.created_at = _timestamp(self.created_at, "created_at")
        self.updated_at = _timestamp(self.updated_at, "updated_at")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "model": self.model,
            "worktree_name": self.worktree_name,
            "worktree_path": self.worktree_path,
            "branch": self.branch,
            "backend_type": self.backend_type.value,
            "pane_id": self.pane_id,
            "runtime_task_id": self.runtime_task_id,
            "state": self.state.value,
            "plan_mode_required": self.plan_mode_required,
            "pending_plan_request_id": self.pending_plan_request_id,
            "session_dir": self.session_dir,
            "last_error": self.last_error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: object) -> TeammateInfo:
        keys = set(cls(name="placeholder").to_dict())
        data = _strict_dict(value, keys, "teammate")
        try:
            backend = BackendType(data["backend_type"])
            state = MemberState(data["state"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid teammate enum") from error
        plan = data["plan_mode_required"]
        if not isinstance(plan, bool):
            raise ValueError("plan_mode_required must be a boolean")
        return cls(
            name=data["name"],
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            model=data["model"],
            worktree_name=data["worktree_name"],
            worktree_path=data["worktree_path"],
            branch=data["branch"],
            backend_type=backend,
            pane_id=data["pane_id"],
            runtime_task_id=data["runtime_task_id"],
            state=state,
            plan_mode_required=plan,
            pending_plan_request_id=data["pending_plan_request_id"],
            session_dir=data["session_dir"],
            last_error=data["last_error"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )


@dataclass(slots=True)
class Team:
    name: str
    slug: str
    description: str
    project_root: str
    backend: BackendType = BackendType.IN_PROCESS
    lead_agent_id: str = "lead"
    lead_permission_mode: str = "default"
    members: list[TeammateInfo] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    cleanup_errors: list[str] = field(default_factory=list)
    config_dir: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        self.name = _text(self.name, "team name")
        self.slug = _text(self.slug, "team slug")
        self.description = _optional_text(self.description, "description")
        self.project_root = str(Path(_text(self.project_root, "project_root")).resolve())
        self.lead_agent_id = _text(self.lead_agent_id, "lead_agent_id")
        if self.lead_permission_mode not in _PERMISSION_MODES:
            raise ValueError("invalid lead_permission_mode")
        if not isinstance(self.backend, BackendType):
            raise TypeError("backend must be a BackendType")
        if not isinstance(self.members, list) or any(
            not isinstance(member, TeammateInfo) for member in self.members
        ):
            raise TypeError("members must be a list of TeammateInfo")
        folded = [member.name.casefold() for member in self.members]
        if len(folded) != len(set(folded)):
            raise ValueError("member names must be unique within a team")
        if not isinstance(self.cleanup_errors, list) or any(
            not isinstance(error, str) for error in self.cleanup_errors
        ):
            raise TypeError("cleanup_errors must be a list of strings")
        self.created_at = _timestamp(self.created_at, "created_at")
        self.updated_at = _timestamp(self.updated_at, "updated_at")

    @property
    def config_path(self) -> Path:
        return Path(self.config_dir) / "config.json"

    @property
    def tasks_path(self) -> Path:
        return Path(self.config_dir) / "tasks.json"

    @property
    def mailbox_dir(self) -> Path:
        return Path(self.config_dir) / "mailbox"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "project_root": self.project_root,
            "lead_agent_id": self.lead_agent_id,
            "lead_permission_mode": self.lead_permission_mode,
            "backend": self.backend.value,
            "members": [member.to_dict() for member in self.members],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cleanup_errors": list(self.cleanup_errors),
        }

    @classmethod
    def from_dict(cls, value: object, *, config_dir: str = "") -> Team:
        keys = set(
            cls(name="x", slug="x", description="", project_root=".").to_dict()
        )
        data = _strict_dict(value, keys, "team")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported team schema version")
        members = data["members"]
        errors = data["cleanup_errors"]
        if not isinstance(members, list) or not isinstance(errors, list):
            raise ValueError("invalid team collection fields")
        try:
            backend = BackendType(data["backend"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid team backend") from error
        return cls(
            name=data["name"],
            slug=data["slug"],
            description=data["description"],
            project_root=data["project_root"],
            lead_agent_id=data["lead_agent_id"],
            lead_permission_mode=data["lead_permission_mode"],
            backend=backend,
            members=[TeammateInfo.from_dict(member) for member in members],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            cleanup_errors=list(errors),
            config_dir=config_dir,
        )


@dataclass(frozen=True, slots=True)
class RuntimeHandle:
    team_slug: str
    member_name: str
    agent_id: str = ""
    pane_id: str = ""
    runtime_task_id: str = ""


@dataclass(frozen=True, slots=True)
class SpawnRequest:
    team_slug: str
    member_name: str
    initial_prompt: str
    runtime: object | None = None


@dataclass(frozen=True, slots=True)
class SpawnResult:
    agent_id: str
    pane_id: str = ""
    runtime_task_id: str = ""


@dataclass(frozen=True, slots=True)
class DeleteReport:
    deleted: bool
    slug: str
    errors: tuple[str, ...] = ()
    retained_sessions: tuple[str, ...] = ()


__all__ = [
    "BackendType",
    "DeleteReport",
    "MemberState",
    "RuntimeHandle",
    "SCHEMA_VERSION",
    "SpawnRequest",
    "SpawnResult",
    "Team",
    "TeammateInfo",
    "utc_now",
]
