"""Injection-safe tmux backend for autonomous Team member processes."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence

from codewright.team.types import BackendType, RuntimeHandle, SpawnRequest, SpawnResult

type Runner = Callable[[Sequence[str]], Awaitable[tuple[int, str, str]]]


async def subprocess_runner(argv: Sequence[str]) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    return process.returncode or 0, stdout.decode().strip(), stderr.decode().strip()


class TmuxBackend:
    type = BackendType.TMUX

    def __init__(
        self,
        *,
        runner: Runner = subprocess_runner,
        environ: Mapping[str, str] | None = None,
        executable: str = "tmux",
    ) -> None:
        self._runner = runner
        self._environ = os.environ if environ is None else environ
        self._executable = executable

    async def spawn(self, request: SpawnRequest) -> SpawnResult:
        member_argv = (
            sys.executable,
            "-m",
            "codewright",
            "--team-member",
            request.team_slug,
            request.member_name,
        )
        if self._environ.get("TMUX"):
            argv = (
                self._executable,
                "split-window",
                "-d",
                "-P",
                "-F",
                "#{pane_id}",
                "--",
                *member_argv,
            )
        else:
            session = f"codewright-team-{request.team_slug}"
            argv = (
                self._executable,
                "new-session",
                "-d",
                "-s",
                session,
                "-P",
                "-F",
                "#{pane_id}",
                "--",
                *member_argv,
            )
        code, pane_id, stderr = await self._runner(argv)
        if code or not pane_id:
            raise RuntimeError(f"tmux spawn failed: {stderr or 'missing pane id'}")
        return SpawnResult(
            agent_id=f"agent-{request.team_slug}-{request.member_name}",
            pane_id=pane_id.splitlines()[-1],
        )

    async def wake(self, handle: RuntimeHandle) -> None:
        await self._checked((self._executable, "send-keys", "-t", handle.pane_id, "Enter"))

    async def is_alive(self, handle: RuntimeHandle) -> bool:
        code, stdout, _ = await self._runner(
            (self._executable, "list-panes", "-a", "-F", "#{pane_id}")
        )
        return code == 0 and handle.pane_id in stdout.splitlines()

    async def kill(self, handle: RuntimeHandle) -> None:
        await self._checked((self._executable, "kill-pane", "-t", handle.pane_id))

    async def _checked(self, argv: Sequence[str]) -> None:
        code, _, stderr = await self._runner(argv)
        if code:
            raise RuntimeError(f"tmux command failed: {stderr}")


__all__ = ["TmuxBackend", "subprocess_runner"]
