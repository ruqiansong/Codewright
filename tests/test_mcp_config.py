"""Tests for layered MCP server configuration."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from codewright.mcp import load_config

SYNTHETIC_SECRET = "mcp-test-secret-not-real"
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLE_CONFIG = PROJECT_ROOT / "docs" / "ch06" / "mcp-servers.example.yaml"


def write_layer(path: Path, servers: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"mcp_servers": servers}), encoding="utf-8")
    return path


def set_home(monkeypatch: pytest.MonkeyPatch, home: Path) -> Path:
    monkeypatch.setattr(Path, "home", lambda: home)
    return home / ".codewright" / "config.yaml"


def stdio_server(**overrides: Any) -> dict[str, object]:
    value: dict[str, object] = {
        "type": "stdio",
        "command": "python3.12",
        "args": ["-m", "example"],
    }
    value.update(overrides)
    return value


def test_missing_layers_return_empty_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_home(monkeypatch, tmp_path / "home")

    config = load_config(tmp_path / "project")

    assert config.servers == {}


def test_layers_merge_and_project_replaces_whole_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_path = set_home(monkeypatch, tmp_path / "home")
    root = tmp_path / "project"
    root.mkdir()
    write_layer(
        user_path,
        {
            "shared": stdio_server(args=["user"], env={"USER_ONLY": "yes"}),
            "user-only": stdio_server(command="user-command"),
        },
    )
    write_layer(
        root / ".codewright.yaml",
        {
            "shared": {"type": "http", "url": "https://example.test/mcp"},
            "project-only": stdio_server(command="project-command"),
        },
    )

    config = load_config(root)

    assert list(config.servers) == ["shared", "user-only", "project-only"]
    assert config.servers["shared"].type == "http"
    assert config.servers["shared"].command == ""
    assert config.servers["shared"].env == {}
    assert config.servers["user-only"].command == "user-command"
    assert config.servers["project-only"].command == "project-command"


def test_invalid_user_yaml_is_safe_and_project_still_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    user_path = set_home(monkeypatch, tmp_path / "home")
    user_path.parent.mkdir(parents=True)
    user_path.write_text(f"mcp_servers: [\n  {SYNTHETIC_SECRET}", encoding="utf-8")
    root = tmp_path / "project"
    root.mkdir()
    write_layer(root / ".codewright.yaml", {"working": stdio_server()})

    config = load_config(root)

    captured = capsys.readouterr()
    assert list(config.servers) == ["working"]
    assert "invalid_yaml" in captured.err
    assert SYNTHETIC_SECRET not in captured.err
    assert "Traceback" not in captured.err


def test_environment_expansion_is_limited_to_env_and_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    set_home(monkeypatch, tmp_path / "home")
    monkeypatch.setenv("DEFINED_TOKEN", SYNTHETIC_SECRET)
    monkeypatch.delenv("MISSING_TOKEN", raising=False)
    root = tmp_path / "project"
    root.mkdir()
    write_layer(
        root / ".codewright.yaml",
        {
            "local": {
                "type": "stdio",
                "command": "${DEFINED_TOKEN}",
                "args": ["${DEFINED_TOKEN}"],
                "env": {
                    "TOKEN": "Bearer ${DEFINED_TOKEN}",
                    "MISSING_A": "${MISSING_TOKEN}",
                    "MISSING_B": "prefix-${MISSING_TOKEN}",
                },
            },
            "remote": {
                "type": "http",
                "url": "https://example.test/mcp",
                "headers": {"Authorization": "Bearer ${DEFINED_TOKEN}"},
            },
        },
    )

    config = load_config(root)

    captured = capsys.readouterr()
    local = config.servers["local"]
    assert local.command == "${DEFINED_TOKEN}"
    assert local.args == ["${DEFINED_TOKEN}"]
    assert local.env["TOKEN"] == f"Bearer {SYNTHETIC_SECRET}"
    assert local.env["MISSING_A"] == ""
    assert local.env["MISSING_B"] == "prefix-"
    assert config.servers["remote"].headers["Authorization"] == (f"Bearer {SYNTHETIC_SECRET}")
    assert captured.err.count("${MISSING_TOKEN}") == 1
    assert SYNTHETIC_SECRET not in captured.err


@pytest.mark.parametrize(
    ("definition", "reason"),
    [
        ({}, "type_must_be_stdio_or_http"),
        ({"type": "other"}, "type_must_be_stdio_or_http"),
        ({"type": "stdio"}, "stdio_requires_command"),
        ({"type": "http"}, "http_requires_url"),
        ({"type": "stdio", "command": 42}, "command_must_be_string"),
        (stdio_server(args="bad"), "args_must_be_string_list"),
        (stdio_server(env={"TOKEN": 42}), "env_must_be_string_map"),
        (
            {"type": "http", "url": "https://example.test", "headers": []},
            "headers_must_be_string_map",
        ),
    ],
)
def test_invalid_server_is_skipped_without_affecting_valid_server(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    definition: object,
    reason: str,
) -> None:
    set_home(monkeypatch, tmp_path / "home")
    root = tmp_path / "project"
    root.mkdir()
    write_layer(
        root / ".codewright.yaml",
        {"invalid": definition, "valid": stdio_server()},
    )

    config = load_config(root)

    assert list(config.servers) == ["valid"]
    assert reason in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    [
        [],
        {"mcp_servers": []},
        {"mcp_servers": {"bad": []}},
        {"mcp_servers": {1: stdio_server()}},
    ],
)
def test_invalid_mapping_shapes_degrade_safely(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: object,
) -> None:
    set_home(monkeypatch, tmp_path / "home")
    root = tmp_path / "project"
    root.mkdir()
    (root / ".codewright.yaml").write_text(yaml.safe_dump(content), encoding="utf-8")

    config = load_config(root)

    assert config.servers == {}
    assert "[mcp] warn:" in capsys.readouterr().err


def test_unknown_fields_are_ignored_with_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    set_home(monkeypatch, tmp_path / "home")
    root = tmp_path / "project"
    root.mkdir()
    write_layer(root / ".codewright.yaml", {"demo": stdio_server(extra="ignored")})

    config = load_config(root)

    assert "demo" in config.servers
    assert "ignored unknown fields: extra" in capsys.readouterr().err


def test_ch06_example_defines_three_secret_free_servers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_home(monkeypatch, tmp_path / "home")
    monkeypatch.setenv("EXAMPLE_MCP_TOKEN", "in-memory-test-token")
    root = tmp_path / "project"
    root.mkdir()
    (root / ".codewright.yaml").write_text(
        EXAMPLE_CONFIG.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    config = load_config(root)

    assert list(config.servers) == ["demo", "local-sqlite", "example-http"]
    source = EXAMPLE_CONFIG.read_text(encoding="utf-8")
    assert "${EXAMPLE_MCP_TOKEN}" in source
    assert "in-memory-test-token" not in source
