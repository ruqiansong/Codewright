from __future__ import annotations

import asyncio
from pathlib import Path

from codewright.agent import Agent, CompletionResult
from codewright.compact import new_session_context
from codewright.conversation import Conversation
from codewright.llm import TokenUsage
from codewright.session import Writer
from codewright.task import Manager, Status
from codewright.team.backend.inprocess import InProcessBackend
from codewright.team.runtime import TeammateRuntime
from codewright.team.types import RuntimeHandle, SpawnRequest


class StubAgent(Agent):
    def __init__(self, release: asyncio.Event) -> None:
        self.release = release

    async def run_to_completion(self, *args, **kwargs):
        del args, kwargs
        await self.release.wait()
        return CompletionResult("done", TokenUsage(0, 0, 0))


async def test_inprocess_uses_real_task_id_and_scoped_internal_name(tmp_path: Path) -> None:
    manager = Manager()
    release = asyncio.Event()
    context = new_session_context(str(tmp_path))
    runtime = TeammateRuntime(
        StubAgent(release),
        Conversation("system"),
        "initial",
        "description",
        Writer(context.session_dir, "test"),
    )
    backend = InProcessBackend(manager)

    result = await backend.spawn(SpawnRequest("demo", "alice", "initial", runtime))
    task = await manager.get(result.runtime_task_id)
    assert result.agent_id == result.runtime_task_id == task.id  # type: ignore[union-attr]
    assert task.name == "team:demo:alice"  # type: ignore[union-attr]
    assert await backend.is_alive(RuntimeHandle("demo", "alice", runtime_task_id=task.id))
    release.set()
    for _ in range(20):
        if task.status is Status.COMPLETED:  # type: ignore[union-attr]
            break
        await asyncio.sleep(0)
    await manager.aclose()
    runtime.writer.close()
