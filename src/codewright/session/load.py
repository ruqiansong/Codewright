"""Defensive JSONL loading for resumable Codewright sessions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codewright.llm import Message, MessageRole, ToolCall, ToolResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedSession:
    """Validated non-system history plus recovery metadata."""

    messages: list[Message]
    last_message_ts: int | None
    model: str


def load_session(session_dir: str) -> LoadedSession:
    """Load valid history after the final compaction marker."""
    if not isinstance(session_dir, str) or not session_dir.strip():
        raise ValueError("session_dir must be a non-empty string")
    path = Path(session_dir) / "conversation.jsonl"
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("conversation.jsonl does not exist")

    parsed: list[dict[str, Any]] = []
    model = ""
    last_marker = -1
    try:
        with path.open("rb") as stream:
            for raw_line in stream:
                try:
                    line = raw_line.decode("utf-8")
                    value = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(value, dict):
                    continue
                raw_model = value.get("model")
                if not model and isinstance(raw_model, str):
                    model = raw_model.strip()
                if value.get("type") == "compact":
                    last_marker = len(parsed)
                parsed.append(value)
    except OSError:
        raise

    messages: list[Message] = []
    last_message_ts: int | None = None
    for value in parsed[last_marker + 1 :]:
        if value.get("type") == "compact":
            continue
        try:
            message = _parse_message(value)
            timestamp = _parse_timestamp(value.get("ts"))
        except (TypeError, ValueError, KeyError):
            logger.warning("Invalid session record skipped")
            continue
        messages.append(message)
        last_message_ts = timestamp

    truncated = _truncate_orphaned_tool_calls(messages)
    if len(truncated) != len(messages):
        last_message_ts = _last_retained_timestamp(parsed[last_marker + 1 :], len(truncated))
    return LoadedSession(truncated, last_message_ts, model or "unknown")


def _parse_message(value: dict[str, Any]) -> Message:
    role = MessageRole(value["role"])
    if role is MessageRole.SYSTEM:
        raise ValueError("persisted system messages are not accepted")
    content = value.get("content", "")
    if not isinstance(content, str):
        raise TypeError("content must be a string")
    if role is MessageRole.ASSISTANT:
        return Message(role, content, tool_calls=_parse_tool_calls(value.get("tool_calls")))
    if role is MessageRole.TOOL:
        return Message(role, "", tool_results=_parse_tool_results(value.get("tool_results")))
    return Message(role, content)


def _parse_tool_calls(value: object) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError("tool_calls must be a list")
    calls: list[ToolCall] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise TypeError("tool call must be an object")
        calls.append(ToolCall(raw["id"], raw["name"], raw.get("arguments_json", "{}")))
    return tuple(calls)


def _parse_tool_results(value: object) -> tuple[ToolResult, ...]:
    if not isinstance(value, list):
        raise TypeError("tool_results must be a list")
    results: list[ToolResult] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise TypeError("tool result must be an object")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("tool result metadata must be an object")
        results.append(
            ToolResult(
                tool_call_id=raw["tool_call_id"],
                tool_name=raw["tool_name"],
                content=raw["content"],
                is_error=raw.get("is_error", False),
                error_code=raw.get("error_code"),
                truncated=raw.get("truncated", False),
                metadata=metadata,
            )
        )
    return tuple(results)


def _parse_timestamp(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("ts must be a non-negative integer")
    return value


def _truncate_orphaned_tool_calls(messages: list[Message]) -> list[Message]:
    """Drop an assistant tool request when it is the unpaired history tail."""
    output = list(messages)
    if output and output[-1].role is MessageRole.ASSISTANT and output[-1].tool_calls:
        output.pop()
    return output


def _last_retained_timestamp(values: list[dict[str, Any]], retained: int) -> int | None:
    valid = 0
    timestamp: int | None = None
    for value in values:
        if value.get("type") == "compact":
            continue
        try:
            _parse_message(value)
            current = _parse_timestamp(value.get("ts"))
        except (TypeError, ValueError, KeyError):
            continue
        if valid >= retained:
            break
        timestamp = current
        valid += 1
    return timestamp
