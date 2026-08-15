"""Capability-probed adapter boundary for optional iTerm2 pane control."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from codewright.team.types import RuntimeHandle, SpawnRequest, SpawnResult

type Runner = Callable[[Sequence[str]], Awaitable[tuple[int, str, str]]]


class ItermController:
    def __init__(self, runner: Runner, executable: str = "it2") -> None:
        self._runner = runner
        self._executable = executable

    async def probe(self) -> bool:
        code, stdout, stderr = await self._runner((self._executable, "--help"))
        help_text = f"{stdout}\n{stderr}".casefold()
        return code == 0 and all(word in help_text for word in ("split", "send", "close"))

    async def spawn(self, request: SpawnRequest) -> SpawnResult:
        del request
        raise RuntimeError("iTerm2 spawn is unavailable without a verified adapter contract")

    async def wake(self, handle: RuntimeHandle) -> None:
        del handle
        raise RuntimeError("iTerm2 wake is unavailable")

    async def is_alive(self, handle: RuntimeHandle) -> bool:
        del handle
        return False

    async def kill(self, handle: RuntimeHandle) -> None:
        del handle


__all__ = ["ItermController"]
