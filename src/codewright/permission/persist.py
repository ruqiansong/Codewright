"""Exact project-local allow-rule generation and atomic persistence."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from codewright.llm import ToolCall
from codewright.permission.matcher import ExactMatcher
from codewright.permission.rule import Rule, is_mcp_tool_name
from codewright.permission.sandbox import eval_symlinks_or_ancestor, sandbox_ok
from codewright.permission.settings import (
    Settings,
    extract_target,
    friendly_name,
    load_settings,
    search_pattern_safe,
)

if TYPE_CHECKING:
    from codewright.permission.engine import Engine

_BUILTIN_TOOLS = frozenset(
    {
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "Agent",
        "TaskList",
        "TaskGet",
        "TaskStop",
        "SendMessage",
    }
)


def rule_for(root: Path, call: ToolCall) -> tuple[Rule, str, bool]:
    """Build one exact in-memory rule and its stable YAML representation."""
    invalid = Rule("", None, True)
    is_mcp = is_mcp_tool_name(call.name)
    if call.name not in _BUILTIN_TOOLS and not is_mcp:
        return invalid, "", False
    target, is_file, ok = extract_target(call)
    if not ok or not search_pattern_safe(call):
        return invalid, "", False
    if is_mcp:
        return Rule(call.name, None, True, call.name), call.name, True

    exact_target = target
    if is_file:
        if not sandbox_ok(root, target):
            return invalid, "", False
        try:
            requested = Path(target).expanduser() if target else root
            if not requested.is_absolute():
                requested = root / requested
            exact_target = eval_symlinks_or_ancestor(requested).relative_to(root).as_posix()
            exact_target = exact_target or "."
        except (OSError, RuntimeError, ValueError):
            return invalid, "", False

    public_name = friendly_name(call.name)
    escaped = _escape_pattern(exact_target)
    return (
        Rule(public_name, ExactMatcher(exact_target), True, escaped),
        f"{public_name}({escaped})",
        True,
    )


def persist_local_allow(engine: Engine, call: ToolCall) -> None:
    """Atomically append an exact allow rule and update the live local layer."""
    rule, serialized, ok = rule_for(engine.root, call)
    if not ok:
        raise ValueError("Cannot create a safe persistent permission rule.")

    settings = load_settings(engine.local_path)
    if serialized not in settings.permissions.allow:
        settings.permissions.allow.append(serialized)
        _atomic_write(engine.local_path, settings)
    if rule not in engine.local.allow:
        engine.local.allow.append(rule)


def _escape_pattern(value: str) -> str:
    result: list[str] = []
    for character in value:
        if character in {"\\", "*", "(", ")"}:
            result.append("\\")
        result.append(character)
    return "".join(result)


def _atomic_write(path: Path, settings: Settings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "default_mode": settings.default_mode,
        "permissions": {
            "allow": settings.permissions.allow,
            "deny": settings.permissions.deny,
        },
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


__all__ = ["persist_local_allow", "rule_for"]
