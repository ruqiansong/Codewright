"""Tests for MCP connection management and lifecycle ownership."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mcp.types as mtypes
import pytest

import codewright.mcp.manager as manager_module
from codewright.mcp import Config, McpTool, ServerConfig, new_manager

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURE_SERVER = PROJECT_ROOT / "tests" / "fixtures" / "mcp_test_server.py"
SYNTHETIC_SECRET = "manager-secret-not-real"


async def wait_for_port(port: int) -> None:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            await asyncio.sleep(0.05)
            continue
        writer.close()
        await writer.wait_closed()
        return
    raise AssertionError(f"HTTP fixture did not open port {port}")


def unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def get_tool(manager: manager_module.Manager, name: str) -> McpTool:
    tool = next((item for item in manager.tools() if item.name == name), None)
    assert tool is not None
    return tool


@pytest.mark.asyncio
async def test_empty_config_returns_empty_closeable_manager() -> None:
    manager = await new_manager(Config(), version="0.6.0")

    assert manager.tools() == []
    await manager.close()
    await manager.close()


@pytest.mark.asyncio
async def test_stdio_server_connects_injects_env_and_stops_process() -> None:
    manager = await new_manager(
        Config(
            servers={
                "fixture": ServerConfig(
                    type="stdio",
                    command=sys.executable,
                    args=[str(FIXTURE_SERVER), "--transport", "stdio"],
                    env={"CODEWRIGHT_MCP_TEST_VALUE": SYNTHETIC_SECRET},
                )
            }
        ),
        version="0.6.0",
    )
    try:
        assert [tool.name for tool in manager.tools()] == [
            "mcp__fixture__echo",
            "mcp__fixture__process_info",
            "mcp__fixture__request_header",
        ]
        info = await get_tool(manager, "mcp__fixture__process_info").execute(
            json.dumps({"variable": "CODEWRIGHT_MCP_TEST_VALUE"})
        )
        pid_text, value = info.content.split("|", maxsplit=1)
        pid = int(pid_text)
        assert value == SYNTHETIC_SECRET
        assert process_exists(pid)
    finally:
        await manager.close()

    for _ in range(50):
        if not process_exists(pid):
            break
        await asyncio.sleep(0.02)
    assert not process_exists(pid)


@pytest.mark.asyncio
async def test_http_server_connects_and_receives_configured_header() -> None:
    port = unused_port()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(FIXTURE_SERVER),
        "--transport",
        "streamable-http",
        "--port",
        str(port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    manager: manager_module.Manager | None = None
    try:
        await wait_for_port(port)
        manager = await new_manager(
            Config(
                servers={
                    "remote": ServerConfig(
                        type="http",
                        url=f"http://127.0.0.1:{port}/mcp",
                        headers={"Authorization": f"Bearer {SYNTHETIC_SECRET}"},
                    )
                }
            ),
            version="0.6.0",
        )
        result = await get_tool(manager, "mcp__remote__request_header").execute(
            json.dumps({"name": "authorization"})
        )

        assert result.is_error is False
        assert result.content == f"Bearer {SYNTHETIC_SECRET}"
    finally:
        if manager is not None:
            await manager.close()
        if process.returncode is None:
            process.terminate()
        await process.wait()


@pytest.mark.asyncio
async def test_failed_server_is_isolated_and_does_not_leak_secret(
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = await new_manager(
        Config(
            servers={
                "broken": ServerConfig(
                    type="stdio",
                    command=f"/no/such/{SYNTHETIC_SECRET}",
                ),
                "working": ServerConfig(
                    type="stdio",
                    command=sys.executable,
                    args=[str(FIXTURE_SERVER), "--transport", "stdio"],
                ),
            }
        ),
        version="0.6.0",
    )
    try:
        assert any(tool.name == "mcp__working__echo" for tool in manager.tools())
        captured = capsys.readouterr()
        assert "connect server broken failed" in captured.err
        assert SYNTHETIC_SECRET not in captured.err
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_connect_timeout_is_bounded_and_runner_finishes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def hang(*args: object, **kwargs: object) -> tuple[Any, Any]:
        del args, kwargs
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(manager_module, "_enter_transport", hang)
    monkeypatch.setattr(manager_module, "connect_timeout", 0.01)
    manager = await new_manager(
        Config(servers={"slow": ServerConfig(type="stdio", command="unused")}),
        version="0.6.0",
    )
    await asyncio.sleep(0)

    assert manager.tools() == []
    assert all(runner.task.done() for runner in manager._runners)
    assert "timed out after 0.01s" in capsys.readouterr().err
    await manager.close()


class NoopSession:
    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> mtypes.CallToolResult:
        del name, arguments
        return mtypes.CallToolResult(content=[])


@pytest.mark.asyncio
async def test_duplicate_tool_keeps_later_definition(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = McpTool(
        "mcp__demo__echo",
        "echo",
        "first",
        {"type": "object"},
        False,
        NoopSession(),
    )
    second = McpTool(
        "mcp__demo__echo",
        "echo",
        "second",
        {"type": "object"},
        False,
        NoopSession(),
    )

    async def fake_runner(
        name: str,
        server: ServerConfig,
        version: str,
        ready: asyncio.Future[tuple[McpTool, ...]],
        stop: asyncio.Event,
    ) -> None:
        del name, server, version
        ready.set_result((first, second))
        await stop.wait()

    monkeypatch.setattr(manager_module, "_server_runner", fake_runner)
    manager = await new_manager(
        Config(servers={"demo": ServerConfig(type="stdio", command="unused")}),
        version="0.6.0",
    )
    try:
        assert manager.tools() == [second]
        assert "duplicate tool mcp__demo__echo" in capsys.readouterr().err
    finally:
        await manager.close()


class TrackingContext(AbstractAsyncContextManager[tuple[object, object]]):
    def __init__(self, events: list[tuple[str, asyncio.Task[Any] | None]], *, block: bool) -> None:
        self.events = events
        self.block = block

    async def __aenter__(self) -> tuple[object, object]:
        self.events.append(("transport-enter", asyncio.current_task()))
        return object(), object()

    async def __aexit__(self, *exc: object) -> None:
        del exc
        self.events.append(("transport-exit", asyncio.current_task()))
        if self.block:
            await asyncio.Event().wait()


class FakeSession(AbstractAsyncContextManager["FakeSession"]):
    events: list[tuple[str, asyncio.Task[Any] | None]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    async def __aenter__(self) -> FakeSession:
        self.events.append(("session-enter", asyncio.current_task()))
        return self

    async def __aexit__(self, *exc: object) -> None:
        del exc
        self.events.append(("session-exit", asyncio.current_task()))

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> SimpleNamespace:
        return SimpleNamespace(tools=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("block_close", [False, True])
async def test_contexts_exit_in_runner_task_and_close_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    block_close: bool,
) -> None:
    events: list[tuple[str, asyncio.Task[Any] | None]] = []
    FakeSession.events = events

    async def fake_enter(stack: Any, server: ServerConfig) -> tuple[object, object]:
        del server
        return await stack.enter_async_context(TrackingContext(events, block=block_close))

    monkeypatch.setattr(manager_module, "_enter_transport", fake_enter)
    monkeypatch.setattr(manager_module, "ClientSession", FakeSession)
    monkeypatch.setattr(manager_module, "close_timeout", 0.01)
    manager = await new_manager(
        Config(servers={"demo": ServerConfig(type="stdio", command="unused")}),
        version="0.6.0",
    )

    await manager.close()
    await manager.close()

    entered = next(task for event, task in events if event == "transport-enter")
    assert all(task is entered for _event, task in events)
    assert all(runner.task.done() for runner in manager._runners)
    captured = capsys.readouterr()
    if block_close:
        assert "close server demo timed out" in captured.err
    else:
        assert "close server demo timed out" not in captured.err
