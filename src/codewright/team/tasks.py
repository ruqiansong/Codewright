"""Cross-process-safe shared task graph for one Agent Team."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from codewright.team.persistence import FileLock, atomic_write_json, read_json
from codewright.team.types import utc_now


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class TeamTask:
    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    assignee: str = ""
    blocked_by: tuple[str, ...] = ()
    blocks: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if not self.id.startswith("team-task-") or len(self.id) != 22:
            raise ValueError("invalid team task id")
        if not self.title.strip() or self.title != self.title.strip():
            raise ValueError("task title must be non-empty and trimmed")
        if self.description != self.description.strip() or self.assignee != self.assignee.strip():
            raise ValueError("task text fields must be trimmed")
        if not isinstance(self.status, TaskStatus):
            raise TypeError("status must be a TaskStatus")
        for values in (self.blocked_by, self.blocks):
            if not isinstance(values, tuple) or len(values) != len(set(values)):
                raise ValueError("task dependency lists must be unique tuples")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "assignee": self.assignee,
            "blocked_by": list(self.blocked_by),
            "blocks": list(self.blocks),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: object) -> TeamTask:
        keys = {
            "id",
            "title",
            "description",
            "status",
            "assignee",
            "blocked_by",
            "blocks",
            "created_at",
            "updated_at",
        }
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("invalid team task fields")
        blocked_by = value["blocked_by"]
        blocks = value["blocks"]
        if not isinstance(blocked_by, list) or not all(
            isinstance(item, str) for item in blocked_by
        ):
            raise ValueError("invalid blocked_by")
        if not isinstance(blocks, list) or not all(isinstance(item, str) for item in blocks):
            raise ValueError("invalid blocks")
        try:
            status = TaskStatus(value["status"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid team task status") from error
        return cls(
            id=_string(value["id"]),
            title=_string(value["title"]),
            description=_string(value["description"]),
            status=status,
            assignee=_string(value["assignee"]),
            blocked_by=tuple(blocked_by),
            blocks=tuple(blocks),
            created_at=_string(value["created_at"]),
            updated_at=_string(value["updated_at"]),
        )


def _string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected string")
    return value


class Store:
    """Persist a Team task graph in one locked JSON transaction per mutation."""

    def __init__(self, team_dir: Path) -> None:
        self.team_dir = team_dir
        self.path = team_dir / "tasks.json"
        self.lock_path = team_dir / "tasks.lock"

    async def create(
        self,
        title: str,
        description: str = "",
        *,
        assignee: str = "",
        blocked_by: tuple[str, ...] | list[str] = (),
    ) -> TeamTask:
        task_id = f"team-task-{uuid.uuid4().hex[:12]}"
        dependency_ids = tuple(blocked_by)
        task = TeamTask(
            id=task_id,
            title=title,
            description=description,
            assignee=assignee,
            blocked_by=dependency_ids,
        )
        async with FileLock(self.lock_path):
            tasks = self._load()
            self._validate_dependencies(task.id, dependency_ids, tasks)
            tasks[task.id] = task
            for dependency_id in dependency_ids:
                dependency = tasks[dependency_id]
                tasks[dependency_id] = replace(
                    dependency,
                    blocks=tuple((*dependency.blocks, task.id)),
                    updated_at=utc_now(),
                )
            self._save(tasks)
        return task

    async def get(self, task_id: str) -> TeamTask | None:
        async with FileLock(self.lock_path):
            return self._load().get(task_id)

    async def list(self) -> tuple[TeamTask, ...]:
        async with FileLock(self.lock_path):
            return tuple(self._load().values())

    async def update(
        self,
        task_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        status: TaskStatus | str | None = None,
        assignee: str | None = None,
        blocked_by: tuple[str, ...] | list[str] | None = None,
    ) -> TeamTask:
        async with FileLock(self.lock_path):
            tasks = self._load()
            current = tasks.get(task_id)
            if current is None:
                raise KeyError(f"unknown team task: {task_id}")
            dependencies = current.blocked_by if blocked_by is None else tuple(blocked_by)
            self._validate_dependencies(task_id, dependencies, tasks)
            self._validate_no_cycle(task_id, dependencies, tasks)
            new_status = current.status if status is None else TaskStatus(status)
            updated = replace(
                current,
                title=current.title if title is None else title,
                description=current.description if description is None else description,
                status=new_status,
                assignee=current.assignee if assignee is None else assignee,
                blocked_by=dependencies,
                updated_at=utc_now(),
            )
            removed = set(current.blocked_by) - set(dependencies)
            added = set(dependencies) - set(current.blocked_by)
            for dependency_id in removed:
                dependency = tasks[dependency_id]
                tasks[dependency_id] = replace(
                    dependency,
                    blocks=tuple(item for item in dependency.blocks if item != task_id),
                    updated_at=utc_now(),
                )
            for dependency_id in added:
                dependency = tasks[dependency_id]
                tasks[dependency_id] = replace(
                    dependency,
                    blocks=tuple((*dependency.blocks, task_id)),
                    updated_at=utc_now(),
                )
            tasks[task_id] = updated
            self._save(tasks)
            return updated

    async def is_ready(self, task_id: str) -> bool:
        async with FileLock(self.lock_path):
            tasks = self._load()
            task = tasks.get(task_id)
            if task is None:
                raise KeyError(f"unknown team task: {task_id}")
            return task.status is TaskStatus.PENDING and all(
                tasks[dependency].status is TaskStatus.COMPLETED
                for dependency in task.blocked_by
            )

    def _load(self) -> dict[str, TeamTask]:
        if not self.path.exists():
            return {}
        value = read_json(self.path)
        if not isinstance(value, list):
            raise ValueError("tasks.json must contain a list")
        tasks = [TeamTask.from_dict(item) for item in value]
        if len(tasks) != len({task.id for task in tasks}):
            raise ValueError("duplicate team task id")
        return {task.id: task for task in tasks}

    def _save(self, tasks: dict[str, TeamTask]) -> None:
        atomic_write_json(self.path, [task.to_dict() for task in tasks.values()])

    @staticmethod
    def _validate_dependencies(
        task_id: str, dependency_ids: tuple[str, ...], tasks: dict[str, TeamTask]
    ) -> None:
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("duplicate task dependency")
        if task_id in dependency_ids:
            raise ValueError("a task cannot depend on itself")
        unknown = set(dependency_ids) - set(tasks)
        if unknown:
            raise ValueError(f"unknown task dependencies: {', '.join(sorted(unknown))}")

    @staticmethod
    def _validate_no_cycle(
        task_id: str, dependency_ids: tuple[str, ...], tasks: dict[str, TeamTask]
    ) -> None:
        def reaches_target(node: str, seen: set[str]) -> bool:
            if node == task_id:
                return True
            if node in seen:
                return False
            seen.add(node)
            dependencies = dependency_ids if node == task_id else tasks[node].blocked_by
            return any(reaches_target(dependency, seen) for dependency in dependencies)

        if any(reaches_target(dependency, set()) for dependency in dependency_ids):
            raise ValueError("task dependency cycle detected")


__all__ = ["Store", "TaskStatus", "TeamTask"]
