"""TUI boundary adapter for the worktree manager."""

from __future__ import annotations

from collections.abc import Callable

from codewright.command import WorktreeSummary
from codewright.worktree import ExitAction, ExitOptions, Manager


class WorktreeAdapter:
    def __init__(self, manager: Manager, set_cwd: Callable[[str], None]) -> None:
        self._manager = manager
        self._set_cwd = set_cwd

    def _summary(self, name: str) -> WorktreeSummary:
        item = self._manager.get(name)
        if item is None:
            raise ValueError("Worktree 不存在")
        session = self._manager.current_session()
        return WorktreeSummary(
            item.name,
            item.path,
            item.branch,
            item.manual,
            session is not None and session.worktree_name == item.name,
        )

    async def create(self, name: str) -> WorktreeSummary:
        item = await self._manager.create(name, manual=True)
        return self._summary(item.name)

    def list(self) -> tuple[WorktreeSummary, ...]:
        return tuple(self._summary(item.name) for item in self._manager.list())

    async def enter(self, name: str) -> str:
        session = await self._manager.enter(name)
        self._set_cwd(session.worktree_path)
        return session.worktree_path

    async def exit(self, *, remove: bool, discard: bool) -> str:
        session = self._manager.current_session()
        if session is None:
            raise ValueError("当前没有 Worktree session")
        report = await self._manager.exit(
            session.worktree_name,
            ExitAction.REMOVE if remove else ExitAction.KEEP,
            ExitOptions(discard_changes=discard),
        )
        self._set_cwd(report.restore_cwd)
        return report.restore_cwd

    async def remove(self, name: str, *, discard: bool) -> None:
        await self._manager.remove(name, ExitOptions(discard_changes=discard))


__all__ = ["WorktreeAdapter"]
