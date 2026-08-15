"""Layered MCP server configuration loading and validation."""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

type ServerType = Literal["stdio", "http"]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_KNOWN_FIELDS = frozenset({"type", "command", "args", "env", "url", "headers"})


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """One normalized and validated MCP server definition."""

    type: ServerType
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Config:
    """The effective MCP configuration after merging both layers."""

    servers: dict[str, ServerConfig] = field(default_factory=dict)


@dataclass(slots=True)
class _RawServer:
    type: str | None = None
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)


def load_config(root: str | Path) -> Config:
    """Load user and project MCP layers, degrading invalid entries safely."""
    try:
        user_path = Path.home() / ".codewright" / "config.yaml"
    except (OSError, RuntimeError):
        _warn("load user configuration failed: home_unavailable")
        user_servers: dict[str, _RawServer] = {}
    else:
        user_servers = _load_file(user_path)

    project_path = Path(root) / ".codewright.yaml"
    project_servers = _load_file(project_path)

    for name, server in user_servers.items():
        _apply_expansion(name, server)
    for name, server in project_servers.items():
        _apply_expansion(name, server)

    effective: dict[str, ServerConfig] = {}
    for name, server in _merge_servers(user_servers, project_servers).items():
        validated = _validate_server(name, server)
        if validated is not None:
            effective[name] = validated
    return Config(servers=effective)


def _load_file(path: Path) -> dict[str, _RawServer]:
    try:
        if not path.exists():
            return {}
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        _warn(f"load {_safe(path)} failed: invalid_yaml")
        return {}
    except OSError:
        _warn(f"load {_safe(path)} failed: read_error")
        return {}

    if not isinstance(raw, Mapping):
        _warn(f"load {_safe(path)} failed: root_must_be_mapping")
        return {}
    section = raw.get("mcp_servers")
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        _warn(f"load {_safe(path)} failed: mcp_servers_must_be_mapping")
        return {}

    servers: dict[str, _RawServer] = {}
    for raw_name, value in section.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            _warn("skip server <invalid>: name_must_be_non_empty_string")
            continue
        name = raw_name
        parsed = _parse_server(name, value)
        if parsed is not None:
            servers[name] = parsed
    return servers


def _parse_server(name: str, value: object) -> _RawServer | None:
    if not isinstance(value, Mapping):
        _skip(name, "definition_must_be_mapping")
        return None

    unknown = sorted(str(key) for key in value if key not in _KNOWN_FIELDS)
    if unknown:
        _warn(f"server {_safe(name)} ignored unknown fields: {', '.join(unknown)}")

    server_type = value.get("type")
    command = value.get("command")
    url = value.get("url")
    if server_type is not None and not isinstance(server_type, str):
        _skip(name, "type_must_be_string")
        return None
    if command is not None and not isinstance(command, str):
        _skip(name, "command_must_be_string")
        return None
    if url is not None and not isinstance(url, str):
        _skip(name, "url_must_be_string")
        return None

    args = _string_list(value.get("args", []))
    if args is None:
        _skip(name, "args_must_be_string_list")
        return None
    env = _string_map(value.get("env", {}))
    if env is None:
        _skip(name, "env_must_be_string_map")
        return None
    headers = _string_map(value.get("headers", {}))
    if headers is None:
        _skip(name, "headers_must_be_string_map")
        return None

    return _RawServer(
        type=server_type,
        command=command,
        args=args,
        env=env,
        url=url,
        headers=headers,
    )


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return list(value)


def _string_map(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        return None
    return dict(value)


def _expand_vars(value: str) -> tuple[str, list[str]]:
    undefined: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            undefined.append(name)
            return ""
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value), undefined


def _apply_expansion(name: str, server: _RawServer) -> None:
    undefined: set[str] = set()
    for values in (server.env, server.headers):
        for key, value in values.items():
            expanded, missing = _expand_vars(value)
            values[key] = expanded
            undefined.update(missing)
    for variable in sorted(undefined):
        _warn(f"undefined env var ${{{variable}}} referenced by server {_safe(name)}")


def _merge_servers(
    user: Mapping[str, _RawServer], project: Mapping[str, _RawServer]
) -> dict[str, _RawServer]:
    merged = dict(user)
    merged.update(project)
    return merged


def _validate_server(name: str, server: _RawServer) -> ServerConfig | None:
    if server.type not in {"stdio", "http"}:
        _skip(name, "type_must_be_stdio_or_http")
        return None
    if server.type == "stdio" and (server.command is None or not server.command.strip()):
        _skip(name, "stdio_requires_command")
        return None
    if server.type == "http" and (server.url is None or not server.url.strip()):
        _skip(name, "http_requires_url")
        return None
    server_type: ServerType = "stdio" if server.type == "stdio" else "http"
    return ServerConfig(
        type=server_type,
        command=server.command or "",
        args=list(server.args),
        env=dict(server.env),
        url=server.url or "",
        headers=dict(server.headers),
    )


def _skip(name: str, reason: str) -> None:
    _warn(f"skip server {_safe(name)}: {reason}")


def _warn(message: str) -> None:
    print(f"[mcp] warn: {message}", file=sys.stderr)


def _safe(value: object, limit: int = 160) -> str:
    compact = str(value).replace("\n", " ").replace("\r", " ")
    return compact if len(compact) <= limit else compact[:limit] + "…"


__all__ = ["Config", "ServerConfig", "load_config"]
