"""Adapt remote MCP tools to Codewright's vendor-neutral tool protocol."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import mcp.types as mtypes

from codewright.tool import Result, truncate_text

MAX_MCP_RESULT_CHARS = 65_536
MAX_TOOL_NAME_LENGTH = 64
_VALID_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")
_non_text_warn_once: set[str] = set()
logger = logging.getLogger(__name__)


class CallerSession(Protocol):
    """Small session surface used by an adapted MCP tool."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> mtypes.CallToolResult: ...


@dataclass(frozen=True, slots=True)
class McpTool:
    """One remote MCP tool exposed through Codewright's Tool protocol."""

    full_name: str
    remote_name: str
    description: str
    parameters: dict[str, Any]
    read_only: bool
    caller: CallerSession

    @property
    def name(self) -> str:
        return self.full_name

    async def execute(self, arguments_json: str) -> Result:
        """Validate arguments, call the remote tool, and normalize its result."""
        arguments = _parse_arguments(arguments_json)
        if isinstance(arguments, Result):
            return arguments

        try:
            remote_result = await self.caller.call_tool(
                self.remote_name,
                arguments=arguments or None,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(
                "MCP tool call failed tool=%s error=%s",
                self.full_name,
                type(error).__name__,
            )
            return Result(
                content="MCP tool call failed.",
                is_error=True,
                error_code="mcp_call_failed",
            )

        texts: list[str] = []
        dropped = 0
        for block in remote_result.content:
            if isinstance(block, mtypes.TextContent):
                texts.append(block.text)
            else:
                dropped += 1
        if dropped and self.full_name not in _non_text_warn_once:
            _non_text_warn_once.add(self.full_name)
            print(
                f"[mcp] warn: tool {self.full_name} returned non-text content blocks (dropped)",
                file=sys.stderr,
            )

        content = "\n".join(texts) or "MCP tool completed without text output."
        bounded, truncated = truncate_text(content, max_chars=MAX_MCP_RESULT_CHARS)
        is_error = bool(remote_result.isError)
        return Result(
            content=bounded,
            is_error=is_error,
            error_code="mcp_remote_error" if is_error else None,
            truncated=truncated,
            metadata={
                "tool": self.full_name,
                "non_text_blocks_dropped": dropped,
            },
        )


def adapt_tool(
    server_name: str,
    remote_tool: mtypes.Tool,
    session: CallerSession,
) -> McpTool | None:
    """Validate and adapt one model-facing MCP tool definition."""
    remote_name = getattr(remote_tool, "name", None)
    if not _valid_component(server_name) or not _valid_component(remote_name):
        _warn_skip(server_name, remote_name, "name contains illegal characters")
        return None
    assert isinstance(remote_name, str)
    full_name = f"mcp__{server_name}__{remote_name}"
    if len(full_name) > MAX_TOOL_NAME_LENGTH:
        _warn_skip(server_name, remote_name, "name exceeds 64 characters")
        return None

    raw_schema = getattr(remote_tool, "inputSchema", None)
    if not raw_schema:
        parameters: dict[str, Any] = {"type": "object"}
    elif not isinstance(raw_schema, Mapping) or raw_schema.get("type") != "object":
        _warn_skip(server_name, remote_name, "input schema must be an object")
        return None
    else:
        parameters = dict(raw_schema)

    raw_description = getattr(remote_tool, "description", None)
    description = (
        raw_description.strip()
        if isinstance(raw_description, str) and raw_description.strip()
        else f"Tool {remote_name} from MCP server {server_name}."
    )
    annotations = getattr(remote_tool, "annotations", None)
    read_only = getattr(annotations, "readOnlyHint", None) is True
    return McpTool(
        full_name=full_name,
        remote_name=remote_name,
        description=description,
        parameters=parameters,
        read_only=read_only,
        caller=session,
    )


def _parse_arguments(arguments_json: str) -> dict[str, Any] | Result:
    if not isinstance(arguments_json, str):
        return _invalid_arguments("MCP tool arguments must be a JSON string.")
    try:
        value = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return _invalid_arguments("MCP tool arguments must be valid JSON.")
    if not isinstance(value, dict):
        return _invalid_arguments("MCP tool arguments must be a JSON object.")
    return value


def _invalid_arguments(message: str) -> Result:
    return Result(content=message, is_error=True, error_code="invalid_arguments")


def _valid_component(value: object) -> bool:
    return isinstance(value, str) and _VALID_COMPONENT.fullmatch(value) is not None


def _warn_skip(server_name: object, tool_name: object, reason: str) -> None:
    label = _safe(f"mcp__{server_name}__{tool_name}")
    print(f"[mcp] warn: skip tool {label}: {reason}", file=sys.stderr)


def _safe(value: object, limit: int = 160) -> str:
    compact = str(value).replace("\n", " ").replace("\r", " ")
    return compact if len(compact) <= limit else compact[:limit] + "…"


__all__ = ["CallerSession", "McpTool", "adapt_tool"]
