"""Permission settings loading and built-in tool argument classification."""

import json
import re
import sys
from collections.abc import Set
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeGuard

import yaml

from codewright.llm import ToolCall
from codewright.permission.models import Category
from codewright.permission.rule import RuleSet, is_mcp_tool_name, parse_rule_detailed

_FRIENDLY_NAMES = {
    "bash": "Bash",
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "glob": "Glob",
    "grep": "Grep",
    "load_skill": "LoadSkill",
    "install_skill": "InstallSkill",
    "Agent": "Agent",
    "TaskList": "TaskList",
    "TaskGet": "TaskGet",
    "TaskStop": "TaskStop",
    "SendMessage": "SendMessage",
}
_READ_TOOLS = frozenset({"read_file", "glob", "grep", "load_skill", "TaskList", "TaskGet"})
_WRITE_TOOLS = frozenset({"write_file", "edit_file"})
_EXEC_TOOLS = frozenset({"bash", "install_skill", "Agent", "TaskStop", "SendMessage"})
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[/\\]")


class SettingsError(ValueError):
    """Raised when one permission settings file is unreadable or invalid."""


@dataclass(slots=True)
class PermissionsBlock:
    """Allow and deny rule strings from one settings layer."""

    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Settings:
    """Validated content of one permission settings file."""

    default_mode: str = ""
    permissions: PermissionsBlock = field(default_factory=PermissionsBlock)


def load_settings(path: str | Path) -> Settings:
    """Load one YAML settings file; a missing file is an empty layer."""
    settings_path = Path(path)
    try:
        with settings_path.open(encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except FileNotFoundError:
        return Settings()
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise SettingsError(f"Unable to load permission settings: {settings_path}") from error

    if raw is None:
        return Settings()
    if not isinstance(raw, dict):
        raise SettingsError("Permission settings root must be a YAML mapping.")

    default_mode = raw.get("default_mode", "")
    if not isinstance(default_mode, str):
        raise SettingsError("default_mode must be a string.")
    raw_permissions = raw.get("permissions", {})
    if not isinstance(raw_permissions, dict):
        raise SettingsError("permissions must be a YAML mapping.")
    allow = _string_list(raw_permissions.get("allow", []), field_name="permissions.allow")
    deny = _string_list(raw_permissions.get("deny", []), field_name="permissions.deny")
    return Settings(default_mode.strip(), PermissionsBlock(allow, deny))


def to_rule_set(settings: Settings) -> RuleSet:
    """Convert valid rule strings and safely skip malformed individual entries."""
    rules = RuleSet()
    for value in settings.permissions.allow:
        parsed, error = parse_rule_detailed(value, allow=True)
        if parsed is not None:
            rules.allow.append(parsed)
        else:
            print(f"rule {value!r} parse failed: {error}", file=sys.stderr)
    for value in settings.permissions.deny:
        parsed, error = parse_rule_detailed(value)
        if parsed is not None:
            rules.deny.append(parsed)
        else:
            print(f"rule {value!r} parse failed: {error}", file=sys.stderr)
    return rules


def friendly_name(internal: str) -> str:
    """Map one internal built-in tool name to its public rule name."""
    if not isinstance(internal, str):
        raise TypeError("internal must be a string")
    return _FRIENDLY_NAMES.get(internal, internal)


def categorize(internal: str, read_only: bool) -> Category:
    """Return the security category of one built-in or valid MCP tool."""
    if not isinstance(read_only, bool):
        raise TypeError("read_only must be a boolean")
    if internal in _READ_TOOLS:
        return Category.READ
    if internal in _WRITE_TOOLS:
        return Category.WRITE
    if internal in _EXEC_TOOLS:
        return Category.EXEC
    if is_mcp_tool_name(internal):
        return Category.READ if read_only else Category.EXEC
    raise ValueError(f"Unknown built-in tool: {internal}")


def extract_target(call: ToolCall) -> tuple[str, bool, bool]:
    """Extract and validate the permission target from one built-in tool call."""
    arguments = _arguments(call)
    if arguments is None:
        return "", False, False

    if is_mcp_tool_name(call.name):
        return "", False, True

    if call.name == "TaskList":
        return ("", False, True) if not arguments else ("", False, False)
    if call.name in {"TaskGet", "TaskStop"}:
        task_id = arguments.get("task_id")
        return (task_id, False, True) if _non_empty_string(task_id) else ("", False, False)
    if call.name == "SendMessage":
        name = arguments.get("name")
        message = arguments.get("message")
        if _non_empty_string(name) and _non_empty_string(message):
            return name, False, True
        return "", False, False
    if call.name == "Agent":
        prompt = arguments.get("prompt")
        description = arguments.get("description")
        if _non_empty_string(prompt) and _non_empty_string(description):
            return "", False, True
        return "", False, False

    if call.name == "load_skill":
        name = arguments.get("name")
        return (name, False, True) if _non_empty_string(name) else ("", False, False)

    if call.name == "install_skill":
        url = arguments.get("url")
        return (url, False, True) if _non_empty_string(url) else ("", False, False)

    if call.name == "read_file":
        return _file_target(arguments, required_strings=("path",))
    if call.name == "write_file":
        return _file_target(
            arguments, required_strings=("path", "content"), allow_empty={"content"}
        )
    if call.name == "edit_file":
        return _file_target(
            arguments,
            required_strings=("path", "old_string", "new_string"),
            allow_empty={"new_string"},
        )
    if call.name == "glob":
        if not _non_empty_string(arguments.get("pattern")):
            return "", True, False
        return _optional_search_path(arguments)
    if call.name == "grep":
        if not _non_empty_string(arguments.get("pattern")):
            return "", True, False
        file_glob = arguments.get("glob", "**/*")
        if not _non_empty_string(file_glob):
            return "", True, False
        return _optional_search_path(arguments)
    if call.name == "bash":
        command = arguments.get("command")
        if not _non_empty_string(command):
            return "", False, False
        return command, False, True
    return "", False, False


def search_pattern_safe(call: ToolCall) -> bool:
    """Reject absolute and parent-traversing Glob/Grep file patterns."""
    if call.name not in {"glob", "grep"}:
        return True
    arguments = _arguments(call)
    if arguments is None:
        return False
    key = "pattern" if call.name == "glob" else "glob"
    default = None if call.name == "glob" else "**/*"
    pattern = arguments.get(key, default)
    if not _non_empty_string(pattern):
        return False
    return path_pattern_safe(pattern)


def path_pattern_safe(pattern: str) -> bool:
    """Return whether a file glob is relative and has no parent traversal."""
    if not _non_empty_string(pattern):
        return False
    normalized = pattern.replace("\\", "/")
    if normalized.startswith("/") or _WINDOWS_ABSOLUTE.match(pattern):
        return False
    return ".." not in normalized.split("/")


def _arguments(call: ToolCall) -> dict[str, object] | None:
    try:
        value = json.loads(call.arguments_json)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _file_target(
    arguments: dict[str, object],
    *,
    required_strings: tuple[str, ...],
    allow_empty: Set[str] = frozenset(),
) -> tuple[str, bool, bool]:
    for field_name in required_strings:
        value = arguments.get(field_name)
        if not isinstance(value, str) or (field_name not in allow_empty and not value.strip()):
            return "", True, False
    return str(arguments["path"]), True, True


def _optional_search_path(arguments: dict[str, object]) -> tuple[str, bool, bool]:
    path = arguments.get("path", ".")
    if not isinstance(path, str):
        return "", True, False
    if not path.strip():
        return ".", True, True
    return path, True, True


def _non_empty_string(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: object, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SettingsError(f"{field_name} must be a list of strings.")
    return list(value)


__all__ = [
    "PermissionsBlock",
    "Settings",
    "SettingsError",
    "categorize",
    "extract_target",
    "friendly_name",
    "load_settings",
    "path_pattern_safe",
    "search_pattern_safe",
    "to_rule_set",
]
