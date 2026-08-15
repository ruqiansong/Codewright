"""Strict Markdown and YAML-frontmatter parsing for subagent definitions."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from codewright.permission import Mode, parse_mode
from codewright.subagent.definition import DEFAULT_MAX_TURNS, Definition, Source

UTF8_BOM = b"\xef\xbb\xbf"
AGENT_NAME_REGEX = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_ALLOWED_FIELDS = frozenset(
    {
        "name",
        "description",
        "tools",
        "disallowedTools",
        "model",
        "maxTurns",
        "permissionMode",
        "background",
        "planModeRequired",
        "isolation",
    }
)


class DefinitionParseError(ValueError):
    """Raised when one subagent definition cannot be parsed safely."""


def parse_frontmatter_and_body(data: bytes) -> tuple[dict[str, object], str]:
    """Decode UTF-8 bytes and split one strict frontmatter document."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if data.startswith(UTF8_BOM):
        data = data[len(UTF8_BOM) :]
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DefinitionParseError("subagent definition must be valid UTF-8") from error

    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise DefinitionParseError("subagent definition must start with YAML frontmatter")
    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.rstrip("\r\n") == "---"),
        None,
    )
    if closing_index is None:
        raise DefinitionParseError("subagent frontmatter is not closed")

    try:
        loaded = yaml.safe_load("".join(lines[1:closing_index]))
    except yaml.YAMLError as error:
        raise DefinitionParseError("subagent frontmatter contains invalid YAML") from error
    if not isinstance(loaded, Mapping):
        raise DefinitionParseError("subagent frontmatter must be a mapping")
    if not all(isinstance(key, str) for key in loaded):
        raise DefinitionParseError("subagent frontmatter keys must be strings")
    metadata = dict(loaded)
    if set(metadata) - _ALLOWED_FIELDS:
        raise DefinitionParseError("subagent frontmatter contains unknown fields")

    body = "".join(lines[closing_index + 1 :]).strip()
    if not body:
        raise DefinitionParseError("subagent body must not be empty")
    return metadata, body


def parse_definition(data: bytes, file_path: str, source: Source) -> Definition:
    """Parse one in-memory definition and attach stable source metadata."""
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")
    if not isinstance(source, Source):
        raise TypeError("source must be a Source")
    metadata, body = parse_frontmatter_and_body(data)

    name = metadata.get("name")
    if not isinstance(name, str) or AGENT_NAME_REGEX.fullmatch(name) is None:
        raise DefinitionParseError("subagent name must match ^[a-z][a-z0-9-]{0,31}$")

    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip():
        raise DefinitionParseError("subagent description must be a non-empty string")
    if description != description.strip() or "\n" in description or "\r" in description:
        raise DefinitionParseError("subagent description must be one trimmed line")

    tools = _parse_tool_list(metadata.get("tools"), field_name="tools")
    disallowed_tools = _parse_tool_list(
        metadata.get("disallowedTools"),
        field_name="disallowedTools",
    )
    model = _parse_model(metadata.get("model"))
    max_turns = _parse_max_turns(metadata.get("maxTurns"))
    permission_mode, dont_ask = _parse_permission_mode(metadata.get("permissionMode"), file_path)
    background = metadata.get("background", False)
    if not isinstance(background, bool):
        raise DefinitionParseError("subagent background must be a boolean")
    plan_mode_required = metadata.get("planModeRequired", False)
    if not isinstance(plan_mode_required, bool):
        raise DefinitionParseError("subagent planModeRequired must be a boolean")
    isolation = _parse_isolation(metadata.get("isolation"), file_path)

    return Definition(
        name=name,
        description=description,
        tools=tools,
        disallowed_tools=disallowed_tools,
        model=model,
        max_turns=max_turns,
        permission_mode=permission_mode,
        dont_ask=dont_ask,
        background=background,
        plan_mode_required=plan_mode_required,
        system_prompt=body,
        file_path=file_path,
        source=source,
        isolation=isolation,
    )


def parse_file(path: str | Path, source: Source) -> Definition:
    """Read a regular non-symlink file and parse one definition."""
    selected = Path(path)
    try:
        if selected.is_symlink() or not selected.is_file():
            raise DefinitionParseError("subagent source must be a regular non-symbolic-link file")
        resolved = selected.resolve(strict=True)
        data = selected.read_bytes()
    except DefinitionParseError:
        raise
    except OSError as error:
        raise DefinitionParseError("subagent source could not be read") from error
    return parse_definition(data, str(resolved), source)


def _parse_tool_list(value: object, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise DefinitionParseError(f"subagent {field_name} must be a list of strings")
    if any(not isinstance(item, str) or not item or item != item.strip() for item in value):
        raise DefinitionParseError(f"subagent {field_name} must contain trimmed non-empty strings")
    if len(set(value)) != len(value):
        raise DefinitionParseError(f"subagent {field_name} must not contain duplicates")
    return tuple(value)


def _parse_model(value: object) -> str:
    if value is None:
        return "inherit"
    if not isinstance(value, str) or not value.strip():
        raise DefinitionParseError("subagent model must be a non-empty string")
    if value != value.strip():
        raise DefinitionParseError("subagent model must be trimmed")
    return value


def _parse_max_turns(value: object) -> int:
    if value is None:
        return DEFAULT_MAX_TURNS
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DefinitionParseError("subagent maxTurns must be a positive integer")
    return value


def _parse_permission_mode(value: object, file_path: str) -> tuple[Mode, bool]:
    if value is None:
        return Mode.DEFAULT, False
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DefinitionParseError("subagent permissionMode must be a trimmed string")
    if value.casefold() == "dontask":
        return Mode.DEFAULT, True
    mode, ok = parse_mode(value)
    if ok:
        return mode, False
    print(
        f'subagent "{file_path}": unknown permissionMode "{value}", defaulting to default',
        file=sys.stderr,
    )
    return Mode.DEFAULT, False


def _parse_isolation(value: object, file_path: str) -> str:
    if value is None:
        return ""
    if value == "worktree":
        return "worktree"
    print(
        f'subagent "{file_path}": invalid isolation, disabling isolation',
        file=sys.stderr,
    )
    return ""


__all__ = [
    "AGENT_NAME_REGEX",
    "DefinitionParseError",
    "UTF8_BOM",
    "parse_definition",
    "parse_file",
    "parse_frontmatter_and_body",
]
