"""Tests for ordered registration and protected tool execution."""

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codewright.tool import Registry, Result, Tool, new_default_registry


@dataclass(slots=True)
class FakeTool:
    name: str = "fake"
    description: str = "Return the received arguments."
    parameters: Mapping[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only: bool = True
    delay: float = 0.0
    failure: Exception | None = None
    arguments: list[str] = field(default_factory=list, init=False)

    async def execute(self, arguments_json: str) -> Result:
        self.arguments.append(arguments_json)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failure is not None:
            raise self.failure
        return Result(arguments_json)


@dataclass(slots=True)
class TimeoutTool(FakeTool):
    execution_timeout: float | None = None


def test_registry_is_empty_and_fake_tool_satisfies_protocol() -> None:
    registry = Registry()
    tool = FakeTool()

    assert isinstance(tool, Tool)
    assert registry.definitions() == ()
    assert registry.get("missing") is None
    assert registry.count() == 0


def test_registry_preserves_definition_order_and_rejects_duplicates() -> None:
    registry = Registry()
    first = FakeTool(name="first")
    second = FakeTool(name="second")

    registry.register(first)
    registry.register(second)

    assert registry.count() == 2
    assert registry.get("first") is first
    assert [definition.name for definition in registry.definitions()] == ["first", "second"]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeTool(name="first"))


def test_default_registry_contains_exactly_six_ordered_tools(tmp_path: Path) -> None:
    registry = new_default_registry(working_directory=tmp_path)
    expected = ["read_file", "write_file", "edit_file", "bash", "glob", "grep"]

    assert [definition.name for definition in registry.definitions()] == expected
    assert all(definition.input_schema["type"] == "object" for definition in registry.definitions())
    assert all(registry.get(name) is not None for name in expected)
    assert [definition.name for definition in registry.read_only_definitions()] == [
        "read_file",
        "glob",
        "grep",
    ]
    assert registry.is_read_only("read_file") is True
    assert registry.is_read_only("write_file") is False
    assert registry.is_read_only("missing") is False


@pytest.mark.asyncio
async def test_registry_normalizes_empty_arguments_and_returns_result() -> None:
    registry = Registry()
    tool = FakeTool()
    registry.register(tool)

    result = await registry.execute("fake", "   ")

    assert result == Result("{}")
    assert tool.arguments == ["{}"]


@pytest.mark.asyncio
async def test_registry_returns_unknown_tool_result() -> None:
    result = await Registry().execute("missing", "{}")

    assert result.is_error is True
    assert result.error_code == "unknown_tool"


@pytest.mark.asyncio
async def test_registry_converts_timeout_to_structured_result() -> None:
    registry = Registry(default_timeout=0.01)
    registry.register(FakeTool(delay=1.0))

    result = await registry.execute("fake", "{}")

    assert result.is_error is True
    assert result.error_code == "tool_timeout"
    assert result.metadata["timeout_seconds"] == 0.01


@pytest.mark.asyncio
async def test_tool_can_disable_registry_default_timeout() -> None:
    registry = Registry(default_timeout=0.001)
    tool = TimeoutTool(delay=0.01, execution_timeout=None)
    registry.register(tool)

    assert await registry.execute("fake", "{}") == Result("{}")


@pytest.mark.asyncio
async def test_tool_timeout_is_used_and_explicit_timeout_takes_priority() -> None:
    registry = Registry(default_timeout=1.0)
    registry.register(TimeoutTool(delay=0.03, execution_timeout=0.001))

    timed_out = await registry.execute("fake", "{}")
    completed = await registry.execute("fake", "{}", timeout=0.1)

    assert timed_out.error_code == "tool_timeout"
    assert timed_out.metadata["timeout_seconds"] == 0.001
    assert completed == Result("{}")


@pytest.mark.asyncio
async def test_invalid_optional_tool_timeout_is_rejected_without_execution() -> None:
    registry = Registry()
    tool = TimeoutTool(execution_timeout=0)
    registry.register(tool)

    result = await registry.execute("fake", "{}")

    assert result.error_code == "invalid_timeout"
    assert tool.arguments == []


@pytest.mark.asyncio
async def test_registry_hides_unexpected_exception_details() -> None:
    secret = "synthetic-secret-value"
    registry = Registry()
    registry.register(FakeTool(failure=RuntimeError(secret)))

    result = await registry.execute("fake", "{}")

    assert result.is_error is True
    assert result.error_code == "internal_tool_error"
    assert secret not in result.content


@pytest.mark.asyncio
async def test_registry_propagates_external_cancellation() -> None:
    registry = Registry()
    registry.register(FakeTool(delay=1.0))

    task = asyncio.create_task(registry.execute("fake", "{}"))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_registry_propagates_cancellation_when_tool_timeout_is_disabled() -> None:
    registry = Registry(default_timeout=0.001)
    registry.register(TimeoutTool(delay=1.0, execution_timeout=None))

    task = asyncio.create_task(registry.execute("fake", "{}"))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
