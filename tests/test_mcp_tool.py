"""Tests for adapting and executing MCP tools."""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import mcp.types as mtypes
import pytest

import codewright.mcp.tool as mcp_tool_module
from codewright.mcp import McpTool, adapt_tool
from codewright.tool import Registry

SYNTHETIC_SECRET = "mcp-call-secret-not-real"


class StubSession:
    def __init__(
        self,
        result: mtypes.CallToolResult | None = None,
        *,
        error: Exception | None = None,
        block: bool = False,
    ) -> None:
        self.result = result or mtypes.CallToolResult(
            content=[mtypes.TextContent(type="text", text="ok")]
        )
        self.error = error
        self.block = block
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> mtypes.CallToolResult:
        self.calls.append((name, arguments))
        if self.block:
            await asyncio.Event().wait()
        if self.error is not None:
            raise self.error
        return self.result


def remote_tool(**overrides: Any) -> mtypes.Tool:
    values: dict[str, Any] = {
        "name": "echo",
        "description": "Echo text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
    }
    values.update(overrides)
    return mtypes.Tool(**values)


def test_adapt_tool_maps_definition_and_read_only_annotation() -> None:
    session = StubSession()
    tool = adapt_tool(
        "demo",
        remote_tool(annotations=mtypes.ToolAnnotations(readOnlyHint=True)),
        session,
    )

    assert isinstance(tool, McpTool)
    assert tool.name == "mcp__demo__echo"
    assert tool.remote_name == "echo"
    assert tool.description == "Echo text."
    assert tool.parameters["type"] == "object"
    assert tool.read_only is True


def test_adapt_tool_uses_safe_fallbacks() -> None:
    tool = adapt_tool("demo", remote_tool(description=" ", inputSchema={}), StubSession())

    assert isinstance(tool, McpTool)
    assert tool.description == "Tool echo from MCP server demo."
    assert tool.parameters == {"type": "object"}
    assert tool.read_only is False


@pytest.mark.parametrize(
    ("server", "name"),
    [
        ("bad.server", "echo"),
        ("demo", "bad@tool"),
        ("", "echo"),
        ("demo", "x" * 60),
    ],
)
def test_adapt_tool_rejects_invalid_or_oversized_names(
    capsys: pytest.CaptureFixture[str], server: str, name: str
) -> None:
    tool = SimpleNamespace(
        name=name,
        description="description",
        inputSchema={"type": "object"},
        annotations=None,
    )

    assert adapt_tool(server, tool, StubSession()) is None  # type: ignore[arg-type]
    assert "skip tool" in capsys.readouterr().err


@pytest.mark.parametrize("schema", ["bad", {"type": "string"}])
def test_adapt_tool_rejects_non_object_schema(
    schema: object, capsys: pytest.CaptureFixture[str]
) -> None:
    tool = SimpleNamespace(
        name="echo",
        description="description",
        inputSchema=schema,
        annotations=None,
    )

    assert adapt_tool("demo", tool, StubSession()) is None  # type: ignore[arg-type]
    assert "input schema must be an object" in capsys.readouterr().err


def test_read_only_requires_literal_true() -> None:
    tool = SimpleNamespace(
        name="echo",
        description="description",
        inputSchema={"type": "object"},
        annotations=SimpleNamespace(readOnlyHint="yes"),
    )

    adapted = adapt_tool("demo", tool, StubSession())  # type: ignore[arg-type]

    assert isinstance(adapted, McpTool)
    assert adapted.read_only is False


@pytest.mark.asyncio
async def test_execute_parses_arguments_and_joins_text_blocks() -> None:
    session = StubSession(
        mtypes.CallToolResult(
            content=[
                mtypes.TextContent(type="text", text="first"),
                mtypes.TextContent(type="text", text="second"),
            ]
        )
    )
    tool = adapt_tool("demo", remote_tool(), session)
    assert isinstance(tool, McpTool)

    result = await tool.execute('{"text":"hello"}')

    assert result.content == "first\nsecond"
    assert result.is_error is False
    assert result.error_code is None
    assert session.calls == [("echo", {"text": "hello"})]


@pytest.mark.asyncio
async def test_execute_maps_remote_error() -> None:
    session = StubSession(
        mtypes.CallToolResult(
            content=[mtypes.TextContent(type="text", text="remote rejected")],
            isError=True,
        )
    )
    tool = adapt_tool("demo", remote_tool(), session)
    assert isinstance(tool, McpTool)

    result = await tool.execute("{}")

    assert result.content == "remote rejected"
    assert result.is_error is True
    assert result.error_code == "mcp_remote_error"


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["not-json", "[]", "null"])
async def test_execute_rejects_invalid_arguments(arguments: str) -> None:
    tool = adapt_tool("demo", remote_tool(), StubSession())
    assert isinstance(tool, McpTool)

    result = await tool.execute(arguments)

    assert result.is_error is True
    assert result.error_code == "invalid_arguments"


@pytest.mark.asyncio
async def test_execute_hides_protocol_exception_details(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = StubSession(error=RuntimeError(f"transport failed {SYNTHETIC_SECRET}"))
    tool = adapt_tool("demo", remote_tool(), session)
    assert isinstance(tool, McpTool)

    with caplog.at_level(logging.WARNING):
        result = await tool.execute("{}")

    assert result.is_error is True
    assert result.error_code == "mcp_call_failed"
    assert SYNTHETIC_SECRET not in result.content
    assert SYNTHETIC_SECRET not in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_registry_applies_timeout_to_mcp_tool() -> None:
    tool = adapt_tool("demo", remote_tool(), StubSession(block=True))
    assert isinstance(tool, McpTool)
    registry = Registry(default_timeout=0.01)
    registry.register(tool)

    result = await registry.execute(tool.name, "{}")

    assert result.is_error is True
    assert result.error_code == "tool_timeout"


@pytest.mark.asyncio
async def test_execute_propagates_cancellation() -> None:
    tool = adapt_tool("demo", remote_tool(), StubSession(block=True))
    assert isinstance(tool, McpTool)
    task = asyncio.create_task(tool.execute("{}"))
    await asyncio.sleep(0)

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_non_text_blocks_are_dropped_and_warned_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mcp_tool_module._non_text_warn_once.clear()
    session = StubSession(
        mtypes.CallToolResult(
            content=[
                mtypes.TextContent(type="text", text="visible"),
                mtypes.ImageContent(type="image", data="aGVsbG8=", mimeType="image/png"),
            ]
        )
    )
    tool = adapt_tool("demo", remote_tool(), session)
    assert isinstance(tool, McpTool)

    first = await tool.execute("{}")
    second = await tool.execute("{}")

    assert first.content == "visible"
    assert second.content == "visible"
    assert first.metadata["non_text_blocks_dropped"] == 1
    assert capsys.readouterr().err.count("non-text content blocks") == 1


@pytest.mark.asyncio
async def test_result_is_bounded_and_marked_truncated() -> None:
    oversized = "x" * (mcp_tool_module.MAX_MCP_RESULT_CHARS + 100)
    session = StubSession(
        mtypes.CallToolResult(content=[mtypes.TextContent(type="text", text=oversized)])
    )
    tool = adapt_tool("demo", remote_tool(), session)
    assert isinstance(tool, McpTool)

    result = await tool.execute("{}")

    assert len(result.content) <= mcp_tool_module.MAX_MCP_RESULT_CHARS
    assert result.content.endswith("[truncated]")
    assert result.truncated is True
