"""Concurrent MCP server connection and lifecycle management."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

import httpx
import mcp.types as mtypes
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from codewright.mcp.config import Config, ServerConfig
from codewright.mcp.tool import McpTool, adapt_tool

connect_timeout: float = 30.0
close_timeout: float = 5.0

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Runner:
    name: str
    task: asyncio.Task[None]
    stop: asyncio.Event


class Manager:
    """Own MCP server runners and their stable adapted-tool snapshot."""

    def __init__(self) -> None:
        self._runners: list[_Runner] = []
        self._tools: dict[str, McpTool] = {}
        self._closed = False

    def tools(self) -> list[McpTool]:
        """Return a stable copy of all successfully adapted MCP tools."""
        return [self._tools[name] for name in sorted(self._tools)]

    async def close(self) -> None:
        """Signal all runners and reclaim every task without blocking forever."""
        if self._closed:
            return
        self._closed = True
        for runner in self._runners:
            runner.stop.set()

        tasks = [runner.task for runner in self._runners]
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, timeout=close_timeout + 0.1)
        if pending:
            _warn("close timeout; cancelling unfinished server runners")
            for task in pending:
                task.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)


async def new_manager(cfg: Config, version: str) -> Manager:
    """Connect all configured servers concurrently and isolate their failures."""
    manager = Manager()
    if not cfg.servers:
        return manager

    loop = asyncio.get_running_loop()
    ready_futures: list[asyncio.Future[tuple[McpTool, ...]]] = []
    for name, server in cfg.servers.items():
        ready: asyncio.Future[tuple[McpTool, ...]] = loop.create_future()
        stop = asyncio.Event()
        task = asyncio.create_task(
            _server_runner(name, server, version, ready, stop),
            name=f"codewright-mcp-{_task_label(name)}",
        )
        manager._runners.append(_Runner(name=name, task=task, stop=stop))
        ready_futures.append(ready)

    try:
        discovered = await asyncio.gather(*ready_futures)
    except asyncio.CancelledError:
        for runner in manager._runners:
            runner.stop.set()
            runner.task.cancel()
        await asyncio.gather(
            *(runner.task for runner in manager._runners),
            return_exceptions=True,
        )
        manager._closed = True
        raise

    for tools in discovered:
        for tool in tools:
            if tool.full_name in manager._tools:
                _warn(f"duplicate tool {_safe(tool.full_name)}; keeping later definition")
            manager._tools[tool.full_name] = tool
    return manager


async def _server_runner(
    name: str,
    server: ServerConfig,
    version: str,
    ready: asyncio.Future[tuple[McpTool, ...]],
    stop: asyncio.Event,
) -> None:
    stack = AsyncExitStack()
    try:
        try:
            async with asyncio.timeout(connect_timeout):
                read_stream, write_stream = await _enter_transport(stack, server)
                session = await stack.enter_async_context(
                    ClientSession(
                        read_stream,
                        write_stream,
                        client_info=mtypes.Implementation(name="codewright", version=version),
                    )
                )
                await session.initialize()
                listed = await session.list_tools()
                tools = tuple(
                    adapted
                    for remote in listed.tools
                    if (adapted := adapt_tool(name, remote, session)) is not None
                )
        except TimeoutError:
            _warn(f"connect server {_safe(name)} timed out after {connect_timeout:g}s")
            _set_ready(ready, ())
            return
        except asyncio.CancelledError:
            _set_ready(ready, ())
            raise
        except Exception as error:
            _warn(f"connect server {_safe(name)} failed: {_safe(type(error).__name__)}")
            _set_ready(ready, ())
            return

        logger.info("MCP server connected server=%s tools=%d", _safe(name), len(tools))
        _set_ready(ready, tools)
        await stop.wait()
    finally:
        _set_ready(ready, ())
        await _close_stack(stack, name)


async def _enter_transport(
    stack: AsyncExitStack,
    server: ServerConfig,
) -> tuple[Any, Any]:
    if server.type == "stdio":
        parameters = StdioServerParameters(
            command=server.command,
            args=list(server.args),
            env={**os.environ, **server.env},
        )
        transport = await stack.enter_async_context(stdio_client(parameters))
    else:
        client = await stack.enter_async_context(
            httpx.AsyncClient(headers=server.headers, timeout=None)
        )
        transport = await stack.enter_async_context(
            streamable_http_client(server.url, http_client=client)
        )
    return transport[0], transport[1]


async def _close_stack(stack: AsyncExitStack, name: str) -> None:
    try:
        async with asyncio.timeout(close_timeout):
            await stack.aclose()
    except TimeoutError:
        _warn(f"close server {_safe(name)} timed out after {close_timeout:g}s")
    except asyncio.CancelledError:
        raise
    except Exception as error:
        _warn(f"close server {_safe(name)} failed: {_safe(type(error).__name__)}")


def _set_ready(
    future: asyncio.Future[tuple[McpTool, ...]],
    tools: tuple[McpTool, ...],
) -> None:
    if not future.done():
        future.set_result(tools)


def _warn(message: str) -> None:
    print(f"[mcp] warn: {message}", file=sys.stderr)


def _safe(value: object, limit: int = 160) -> str:
    compact = str(value).replace("\n", " ").replace("\r", " ")
    return compact if len(compact) <= limit else compact[:limit] + "…"


def _task_label(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)[:48]


__all__ = ["Manager", "new_manager"]
