"""Tool protocol, registry, timeout boundary, and result-size helpers."""

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from codewright.llm import ToolDefinition
from codewright.tool.models import Result

DEFAULT_TIMEOUT = 30.0
TRUNCATION_MARKER = "\n[truncated]"


@runtime_checkable
class Tool(Protocol):
    """Protocol implemented by every vendor-neutral Codewright tool."""

    @property
    def name(self) -> str:
        """Return the unique model-facing tool name."""
        ...

    @property
    def description(self) -> str:
        """Return the stable model-facing tool description."""
        ...

    @property
    def parameters(self) -> Mapping[str, object]:
        """Return the tool's JSON object Schema."""
        ...

    @property
    def read_only(self) -> bool:
        """Return whether concurrent execution is free of intended side effects."""
        ...

    async def execute(self, arguments_json: str) -> Result:
        """Execute one invocation represented by a complete JSON string."""
        ...


def truncate_text(
    text: str,
    *,
    max_chars: int | None = None,
    max_lines: int | None = None,
) -> tuple[str, bool]:
    """Bound text deterministically and append a visible truncation marker."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if max_chars is not None and max_chars < len(TRUNCATION_MARKER):
        raise ValueError(f"max_chars must be at least {len(TRUNCATION_MARKER)}")
    if max_lines is not None and max_lines < 1:
        raise ValueError("max_lines must be at least 1")

    truncated = False
    bounded = text
    if max_lines is not None:
        lines = bounded.splitlines(keepends=True)
        if len(lines) > max_lines:
            bounded = "".join(lines[:max_lines]).rstrip("\r\n")
            truncated = True

    marker_length = len(TRUNCATION_MARKER) if truncated else 0
    if max_chars is not None and len(bounded) + marker_length > max_chars:
        bounded = bounded[: max_chars - len(TRUNCATION_MARKER)]
        truncated = True

    if truncated:
        bounded = bounded.rstrip("\r\n") + TRUNCATION_MARKER
    return bounded, truncated


class Registry:
    """Ordered tool registry and the final tool execution safety boundary."""

    def __init__(self, *, default_timeout: float = DEFAULT_TIMEOUT) -> None:
        if (
            not isinstance(default_timeout, (int, float))
            or isinstance(default_timeout, bool)
            or default_timeout <= 0
        ):
            raise ValueError("default_timeout must be a positive number")
        self._default_timeout = float(default_timeout)
        self._order: list[str] = []
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register one valid tool without replacing an existing name."""
        if not isinstance(tool, Tool):
            raise TypeError("tool must satisfy the Tool protocol")
        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ValueError("tool name must not be empty")
        if tool.name != tool.name.strip():
            raise ValueError("tool name must not have surrounding whitespace")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")

        # Validate all model-facing definition data at startup, not request time.
        ToolDefinition(tool.name, tool.description, tool.parameters)
        self._tools[tool.name] = tool
        self._order.append(tool.name)

    def get(self, name: str) -> Tool | None:
        """Return a registered tool, or None for an unknown name."""
        return self._tools.get(name)

    def count(self) -> int:
        """Return the number of registered tools in constant time."""
        return len(self._tools)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return immutable definitions in registration order."""
        return tuple(
            ToolDefinition(name, self._tools[name].description, self._tools[name].parameters)
            for name in self._order
        )

    def read_only_definitions(self) -> tuple[ToolDefinition, ...]:
        """Return immutable read-only definitions in registration order."""
        return tuple(
            ToolDefinition(name, self._tools[name].description, self._tools[name].parameters)
            for name in self._order
            if self._tools[name].read_only
        )

    def is_read_only(self, name: str) -> bool:
        """Return False for unknown or potentially side-effecting tools."""
        tool = self.get(name)
        return tool is not None and tool.read_only

    async def execute(
        self,
        name: str,
        arguments_json: str,
        *,
        timeout: float | None = None,
    ) -> Result:
        """Execute a tool with normalization, timeout, and exception protection."""
        tool = self.get(name)
        if tool is None:
            return Result(
                content=f"Unknown tool: {name}",
                is_error=True,
                error_code="unknown_tool",
            )
        if not isinstance(arguments_json, str):
            return Result(
                content="Tool arguments must be a JSON string.",
                is_error=True,
                error_code="invalid_arguments",
            )

        normalized_arguments = arguments_json if arguments_json.strip() else "{}"
        selected_timeout = (
            getattr(tool, "execution_timeout", self._default_timeout)
            if timeout is None
            else timeout
        )
        if selected_timeout is not None and (
            not isinstance(selected_timeout, (int, float))
            or isinstance(selected_timeout, bool)
            or selected_timeout <= 0
        ):
            return Result(
                content="Tool timeout must be a positive number.",
                is_error=True,
                error_code="invalid_timeout",
            )

        try:
            if selected_timeout is None:
                return await tool.execute(normalized_arguments)
            return await asyncio.wait_for(
                tool.execute(normalized_arguments), timeout=float(selected_timeout)
            )
        except TimeoutError:
            if selected_timeout is None:
                return Result(
                    content="Tool execution failed unexpectedly.",
                    is_error=True,
                    error_code="internal_tool_error",
                )
            timeout_seconds = float(selected_timeout)
            return Result(
                content=f"Tool execution timed out after {timeout_seconds:g} seconds.",
                is_error=True,
                error_code="tool_timeout",
                metadata={"timeout_seconds": timeout_seconds},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return Result(
                content="Tool execution failed unexpectedly.",
                is_error=True,
                error_code="internal_tool_error",
            )


def new_default_registry(
    *,
    working_directory: Path | None = None,
    default_timeout: float = DEFAULT_TIMEOUT,
) -> Registry:
    """Build the fixed V0.2 tool set in its stable model-facing order."""
    from codewright.tool.bash import BashTool
    from codewright.tool.edit_file import EditFileTool
    from codewright.tool.glob_tool import GlobTool
    from codewright.tool.grep_tool import GrepTool
    from codewright.tool.read_file import ReadFileTool
    from codewright.tool.write_file import WriteFileTool

    if working_directory is not None and not isinstance(working_directory, Path):
        raise TypeError("working_directory must be a Path or None")
    search_root = (working_directory or Path.cwd()).resolve()
    registry = Registry(default_timeout=default_timeout)
    for tool in (
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        BashTool(working_directory),
        GlobTool(search_root),
        GrepTool(search_root),
    ):
        registry.register(tool)
    return registry
