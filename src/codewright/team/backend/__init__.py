"""Runtime backend contracts for Agent Team members."""

from __future__ import annotations

from typing import Protocol

from codewright.team.types import BackendType, RuntimeHandle, SpawnRequest, SpawnResult


class Backend(Protocol):
    type: BackendType

    async def spawn(self, request: SpawnRequest) -> SpawnResult: ...

    async def wake(self, handle: RuntimeHandle) -> None: ...

    async def is_alive(self, handle: RuntimeHandle) -> bool: ...

    async def kill(self, handle: RuntimeHandle) -> None: ...


__all__ = ["Backend"]
