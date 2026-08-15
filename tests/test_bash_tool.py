"""Behavior tests for asynchronous shell execution."""

import json
import shlex
import sys
from pathlib import Path

import pytest

from codewright.tool import Registry
from codewright.tool.bash import MAX_OUTPUT_CHARS, BashTool


def command(script: str) -> str:
    executable = shlex.quote(sys.executable)
    return json.dumps({"command": f"{executable} -c {shlex.quote(script)}"})


def test_bash_description_prefers_specialized_tools() -> None:
    assert "Prefer read_file, glob, and grep" in BashTool.description


@pytest.mark.asyncio
async def test_bash_returns_stdout_stderr_and_exit_code(tmp_path: Path) -> None:
    result = await BashTool(tmp_path).execute(
        command("import sys; print('out'); print('err', file=sys.stderr)")
    )

    assert result.is_error is False
    assert result.metadata["exit_code"] == 0
    assert result.metadata["stdout"] == "out\n"
    assert result.metadata["stderr"] == "err\n"
    assert "Exit code: 0" in result.content


@pytest.mark.asyncio
async def test_bash_nonzero_exit_is_structured_error(tmp_path: Path) -> None:
    result = await BashTool(tmp_path).execute(command("import sys; print('bad'); sys.exit(7)"))

    assert result.is_error is True
    assert result.error_code == "command_failed"
    assert result.metadata["exit_code"] == 7
    assert result.metadata["stdout"] == "bad\n"


@pytest.mark.asyncio
async def test_bash_is_terminated_by_registry_timeout(tmp_path: Path) -> None:
    registry = Registry(default_timeout=0.05)
    registry.register(BashTool(tmp_path))

    result = await registry.execute("bash", command("import time; time.sleep(5)"))

    assert result.is_error is True
    assert result.error_code == "tool_timeout"


@pytest.mark.asyncio
async def test_bash_truncates_long_output(tmp_path: Path) -> None:
    result = await BashTool(tmp_path).execute(command("print('x' * 40000)"))

    assert result.truncated is True
    assert "[truncated]" in result.content
    assert len(result.content) <= MAX_OUTPUT_CHARS


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", ["{}", "not-json", json.dumps({"command": "  "})])
async def test_bash_rejects_invalid_command(payload: str, tmp_path: Path) -> None:
    result = await BashTool(tmp_path).execute(payload)

    assert result.is_error is True
    assert result.error_code == "invalid_arguments"
