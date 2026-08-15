"""Bounded, asynchronous collection of non-sensitive runtime context."""

import asyncio
import platform as platform_module
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path

GIT_TIMEOUT_SECONDS = 2.0
MAX_GIT_STATUS_LINES = 20


@dataclass(frozen=True, slots=True)
class Environment:
    """Runtime facts that may change independently of the stable system prompt."""

    working_dir: str
    platform: str
    date: str
    git_status: str
    version: str
    model: str

    def render(self) -> str:
        """Render available facts in a deterministic order."""
        values = (
            ("Working directory", self.working_dir),
            ("Platform", self.platform),
            ("Date", self.date),
            ("Git status", self.git_status),
            ("Codewright version", self.version),
            ("Model", self.model),
        )
        lines = ["Environment:"]
        lines.extend(f"{label}: {value}" for label, value in values if value)
        return "\n".join(lines)


async def gather_environment(version: str, model: str) -> Environment:
    """Collect bounded runtime context without reading environment variables."""
    try:
        working_dir = str(Path.cwd())
    except OSError:
        working_dir = ""

    git_status = await _gather_git_status(working_dir or None)
    return Environment(
        working_dir=working_dir,
        platform=platform_module.system().lower(),
        date=date.today().isoformat(),
        git_status=git_status,
        version=version,
        model=model,
    )


async def _gather_git_status(working_dir: str | None) -> str:
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "status",
            "--porcelain",
            cwd=working_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(
            process.communicate(),
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        if process is not None:
            await _terminate_and_reap(process)
        return ""
    except asyncio.CancelledError:
        if process is not None:
            await _terminate_and_reap(process)
        raise
    except OSError:
        return ""

    if process.returncode != 0:
        return ""
    lines = stdout.decode("utf-8", errors="replace").splitlines()
    if not lines:
        return "clean"
    visible = lines[:MAX_GIT_STATUS_LINES]
    if len(lines) > MAX_GIT_STATUS_LINES:
        visible.append(f"... {len(lines) - MAX_GIT_STATUS_LINES} more changes")
    return "\n".join(visible)


async def _terminate_and_reap(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    with suppress(ProcessLookupError, OSError):
        await process.communicate()


__all__ = ["Environment", "gather_environment"]
