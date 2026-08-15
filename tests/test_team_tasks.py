from __future__ import annotations

from pathlib import Path

import pytest

from codewright.team.tasks import Store, TaskStatus


async def test_task_graph_updates_both_edges_and_readiness(tmp_path: Path) -> None:
    store = Store(tmp_path)
    dependency = await store.create("dependency")
    dependent = await store.create("dependent", blocked_by=(dependency.id,))
    assert dependent.id.startswith("team-task-")
    assert dependent.id in (await store.get(dependency.id)).blocks  # type: ignore[union-attr]
    assert not await store.is_ready(dependent.id)
    await store.update(dependency.id, status=TaskStatus.COMPLETED)
    assert await store.is_ready(dependent.id)


async def test_task_graph_rejects_unknown_self_and_cycle(tmp_path: Path) -> None:
    store = Store(tmp_path)
    with pytest.raises(ValueError, match="unknown"):
        await store.create("bad", blocked_by=("team-task-000000000000",))
    first = await store.create("first")
    second = await store.create("second", blocked_by=(first.id,))
    with pytest.raises(ValueError, match="cycle"):
        await store.update(first.id, blocked_by=(second.id,))
    with pytest.raises(ValueError, match="itself"):
        await store.update(first.id, blocked_by=(first.id,))
