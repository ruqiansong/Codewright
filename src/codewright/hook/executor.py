"""Failure-open executors for lifecycle Hook actions."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from codewright.hook.rule import ActionType, HttpAction, Payload, Rule, ShellAction


@dataclass(slots=True)
class ExecutionResult:
    """One normalized action outcome consumed by the Hook Engine."""

    blocked: bool = False
    reason: str = ""
    prompt: str = ""
    err: Exception | None = None


class Executor:
    """Execute validated actions while containing ordinary Hook failures."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_http_client = http_client is None

    async def run(
        self,
        rule: Rule,
        payload: Payload,
        *,
        blocking: bool,
    ) -> ExecutionResult:
        """Dispatch one action; propagate cancellation but contain other errors."""
        try:
            action = rule.action
            if action.type is ActionType.SHELL:
                if action.shell is None:
                    raise ValueError("shell action payload is missing")
                return await self._run_shell(
                    action.shell,
                    payload,
                    blocking=blocking,
                    timeout_s=rule.timeout_s,
                )
            if action.type is ActionType.PROMPT:
                if action.prompt is None:
                    raise ValueError("prompt action payload is missing")
                return ExecutionResult(prompt=action.prompt.text)
            if action.type is ActionType.HTTP:
                if action.http is None:
                    raise ValueError("http action payload is missing")
                return await self._run_http(
                    action.http,
                    payload,
                    blocking=blocking,
                    timeout_s=rule.timeout_s,
                )
            if action.type is ActionType.SUBAGENT:
                if action.subagent is None:
                    raise ValueError("subagent action payload is missing")
                print(
                    f"[hook subagent] not yet implemented, skipped: {action.subagent.agent_name}",
                    file=sys.stderr,
                )
                return ExecutionResult()
            raise ValueError(f"unknown action type: {action.type}")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return ExecutionResult(err=error)

    async def _run_shell(
        self,
        action: ShellAction,
        payload: Payload,
        *,
        blocking: bool,
        timeout_s: float,
    ) -> ExecutionResult:
        cwd = payload.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            raise ValueError("hook payload cwd must be a non-empty string")
        if not Path(cwd).is_dir():
            raise ValueError(f"hook payload cwd is not a directory: {cwd}")
        process = await asyncio.create_subprocess_shell(
            action.command,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(_marshal_sorted(payload)),
                timeout=timeout_s,
            )
        except asyncio.CancelledError:
            await _kill_process(process)
            raise
        except TimeoutError:
            await _kill_process(process)
            return ExecutionResult(err=TimeoutError(f"shell timed out after {timeout_s:g}s"))

        stderr_text = stderr.decode("utf-8", errors="replace")
        stdout_text = stdout.decode("utf-8", errors="replace")
        if blocking and process.returncode == 2:
            return ExecutionResult(
                blocked=True,
                reason=(stderr_text or stdout_text).rstrip("\n"),
            )
        if process.returncode == 0:
            if stderr_text:
                print(stderr_text, end="", file=sys.stderr)
            return ExecutionResult()
        detail = stderr_text.rstrip("\n") or stdout_text.rstrip("\n")
        suffix = f": {detail}" if detail else ""
        return ExecutionResult(
            err=RuntimeError(f"exit {process.returncode}{suffix}"),
        )

    async def _run_http(
        self,
        action: HttpAction,
        payload: Payload,
        *,
        blocking: bool,
        timeout_s: float,
    ) -> ExecutionResult:
        body = (
            _marshal_sorted(payload)
            if action.body is None
            else action.body.format_map(payload).encode("utf-8")
        )
        response = await self._http_client.request(
            action.method or "POST",
            action.url,
            content=body,
            headers=action.headers,
            timeout=timeout_s,
        )
        if not response.is_success:
            return ExecutionResult(err=RuntimeError(f"HTTP status {response.status_code}"))
        if not blocking:
            return ExecutionResult()
        try:
            decoded = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return ExecutionResult(err=ValueError("invalid HTTP decision JSON"))
        if not isinstance(decoded, dict) or decoded.get("decision") != "block":
            return ExecutionResult()
        reason = decoded.get("reason")
        if not isinstance(reason, str):
            return ExecutionResult(err=ValueError("HTTP block decision requires string reason"))
        return ExecutionResult(blocked=True, reason=reason)

    async def aclose(self) -> None:
        """Close only the HTTP client owned by this Executor."""
        if self._owns_http_client:
            await self._http_client.aclose()


def _marshal_sorted(payload: Payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    """Kill the shell and its child process group, then reap the shell."""
    if process.returncode is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
    await process.wait()


__all__ = ["ExecutionResult", "Executor"]
