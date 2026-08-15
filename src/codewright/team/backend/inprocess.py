"""In-process Team backend built on the existing background Task Manager."""

from __future__ import annotations

from codewright.task import Manager, Status
from codewright.team.runtime import TeammateRuntime
from codewright.team.types import BackendType, RuntimeHandle, SpawnRequest, SpawnResult


class InProcessBackend:
    type = BackendType.IN_PROCESS

    def __init__(self, task_manager: Manager) -> None:
        if not isinstance(task_manager, Manager):
            raise TypeError("task_manager must be a task Manager")
        self._task_manager = task_manager

    async def spawn(self, request: SpawnRequest) -> SpawnResult:
        runtime = request.runtime
        if not isinstance(runtime, TeammateRuntime):
            raise TypeError("in-process spawn requires a TeammateRuntime")
        task = await self._task_manager.launch(
            runtime.agent,
            runtime.conversation,
            runtime.initial_prompt,
            runtime.description,
            name=f"team:{request.team_slug}:{request.member_name}",
            owned_provider=runtime.owned_provider,
        )
        return SpawnResult(agent_id=task.id, runtime_task_id=task.id)

    async def wake(self, handle: RuntimeHandle) -> None:
        del handle

    async def is_alive(self, handle: RuntimeHandle) -> bool:
        task = await self._task_manager.get(handle.runtime_task_id)
        return task is not None and task.status is Status.RUNNING

    async def kill(self, handle: RuntimeHandle) -> None:
        task = await self._task_manager.get(handle.runtime_task_id)
        if task is not None and task.status is Status.RUNNING:
            await self._task_manager.stop(task.id)


__all__ = ["InProcessBackend"]
