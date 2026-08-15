"""Asynchronous shell command execution with bounded output and safe cleanup."""

import asyncio
import json
import os
import signal
from collections.abc import Mapping
from pathlib import Path

from codewright.tool.ctx import cwd_from_ctx
from codewright.tool.models import Result

MAX_OUTPUT_CHARS = 30_000
_TRUNCATION_MARKER = "\n[truncated]\n"


class BashTool:
    """Run one shell command in the configured working directory."""

    name = "bash"
    read_only = False
    description = (
        "Run a shell command in the current working directory and return its output. "
        "Prefer read_file, glob, and grep for reading files, finding files, and searching content."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {"command": {"type": "string", "description": "Shell command to execute."}},
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, working_directory: Path | None = None) -> None:
        self._working_directory = working_directory or Path.cwd()

    async def execute(self, arguments_json: str) -> Result:
        """Validate and execute a shell command without blocking the event loop."""
        command_or_error = _parse_command(arguments_json)
        if isinstance(command_or_error, Result):
            return command_or_error

        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_shell(
                command_or_error,
                cwd=cwd_from_ctx() or self._working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
            stdout_bytes, stderr_bytes = await process.communicate()
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
        except asyncio.CancelledError:
            if process is not None:
                await _kill_and_reap(process)
            raise
        except OSError:
            return Result(
                content="Could not start the shell command.",
                is_error=True,
                error_code="command_start_failed",
            )

        stdout, stdout_truncated = _truncate_middle(stdout, MAX_OUTPUT_CHARS // 2)
        stderr, stderr_truncated = _truncate_middle(stderr, MAX_OUTPUT_CHARS // 2)
        content = _format_result(process.returncode, stdout, stderr)
        content, content_truncated = _truncate_middle(content, MAX_OUTPUT_CHARS)
        truncated = stdout_truncated or stderr_truncated or content_truncated
        is_error = process.returncode != 0
        return Result(
            content=content,
            is_error=is_error,
            error_code="command_failed" if is_error else None,
            truncated=truncated,
            metadata={
                "exit_code": process.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )


def _parse_command(arguments_json: str) -> str | Result:
    try:
        arguments = json.loads(arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return _invalid_arguments("Arguments must be a valid JSON object.")
    if not isinstance(arguments, dict):
        return _invalid_arguments("Arguments must be a JSON object.")
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return _invalid_arguments("command must be a non-empty string.")
    return command


async def _kill_and_reap(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError:
        pass
    await process.wait()


def _format_result(exit_code: int | None, stdout: str, stderr: str) -> str:
    stdout_display = stdout if stdout else "(empty)"
    stderr_display = stderr if stderr else "(empty)"
    return f"Exit code: {exit_code}\nstdout:\n{stdout_display}\nstderr:\n{stderr_display}"


def _truncate_middle(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    remaining = limit - len(_TRUNCATION_MARKER)
    prefix_length = remaining // 2
    suffix_length = remaining - prefix_length
    return text[:prefix_length] + _TRUNCATION_MARKER + text[-suffix_length:], True


def _invalid_arguments(message: str) -> Result:
    return Result(content=message, is_error=True, error_code="invalid_arguments")
