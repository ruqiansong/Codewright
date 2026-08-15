"""Tests for strict subagent definition parsing."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from codewright.permission import Mode
from codewright.subagent import (
    Definition,
    DefinitionParseError,
    Source,
    parse_definition,
    parse_file,
    parse_frontmatter_and_body,
)


def definition_bytes(frontmatter: str, body: str = "System instructions") -> bytes:
    return f"---\n{frontmatter}\n---\n{body}\n".encode()


def test_definition_fields_and_frozen_contract() -> None:
    assert [field.name for field in fields(Definition)] == [
        "name",
        "description",
        "tools",
        "disallowed_tools",
        "model",
        "max_turns",
        "permission_mode",
        "dont_ask",
        "background",
        "plan_mode_required",
        "system_prompt",
        "file_path",
        "source",
        "isolation",
    ]
    definition = Definition(
        name="worker",
        description="Worker",
        system_prompt="Do work",
    )
    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"  # type: ignore[misc]


def test_parse_definition_applies_defaults() -> None:
    definition = parse_definition(
        definition_bytes("name: worker\ndescription: A worker"),
        "project:worker.md",
        Source.PROJECT,
    )

    assert definition == Definition(
        name="worker",
        description="A worker",
        model="inherit",
        max_turns=25,
        permission_mode=Mode.DEFAULT,
        system_prompt="System instructions",
        file_path="project:worker.md",
        source=Source.PROJECT,
    )


def test_parse_definition_accepts_all_fields_and_utf8_bom() -> None:
    data = b"\xef\xbb\xbf" + definition_bytes(
        "\n".join(
            (
                "name: auto-bash",
                "description: Automatic bash worker",
                "tools: [bash, read_file]",
                "disallowedTools: [write_file, edit_file]",
                "model: secondary",
                "maxTurns: 7",
                "permissionMode: dontAsk",
                "background: true",
                "planModeRequired: true",
            )
        ),
        "First line\n\nSecond line",
    )

    definition = parse_definition(data, "agent.md", Source.USER)

    assert definition.name == "auto-bash"
    assert definition.tools == ("bash", "read_file")
    assert definition.disallowed_tools == ("write_file", "edit_file")
    assert definition.model == "secondary"
    assert definition.max_turns == 7
    assert definition.permission_mode is Mode.DEFAULT
    assert definition.dont_ask is True
    assert definition.background is True
    assert definition.plan_mode_required is True
    assert definition.system_prompt == "First line\n\nSecond line"


def test_unknown_permission_mode_warns_and_falls_back(
    capsys: pytest.CaptureFixture[str],
) -> None:
    definition = parse_definition(
        definition_bytes("name: worker\ndescription: Worker\npermissionMode: surprising"),
        "safe-path.md",
        Source.PROJECT,
    )

    assert definition.permission_mode is Mode.DEFAULT
    assert definition.dont_ask is False
    warning = capsys.readouterr().err
    assert 'unknown permissionMode "surprising"' in warning
    assert "defaulting to default" in warning


def test_model_is_a_provider_name_not_a_vendor_alias() -> None:
    definition = parse_definition(
        definition_bytes("name: worker\ndescription: Worker\nmodel: local-provider"),
        "worker.md",
        Source.PROJECT,
    )

    assert definition.model == "local-provider"


@pytest.mark.parametrize(
    ("frontmatter", "message"),
    [
        ("description: missing name", "name"),
        ("name: Worker\ndescription: uppercase", "name"),
        ("name: under_score\ndescription: underscore", "name"),
        ("name: worker", "description"),
        ("name: worker\ndescription: ' padded '", "description"),
        ("name: worker\ndescription: worker\ntools: bash", "tools"),
        ("name: worker\ndescription: worker\ntools: [bash, bash]", "duplicates"),
        ("name: worker\ndescription: worker\ndisallowedTools: ['']", "non-empty"),
        ("name: worker\ndescription: worker\nmodel: ''", "model"),
        ("name: worker\ndescription: worker\nmaxTurns: 0", "maxTurns"),
        ("name: worker\ndescription: worker\nmaxTurns: -1", "maxTurns"),
        ("name: worker\ndescription: worker\nmaxTurns: true", "maxTurns"),
        ("name: worker\ndescription: worker\nmaxTurns: '5'", "maxTurns"),
        ("name: worker\ndescription: worker\nbackground: 'yes'", "background"),
        ("name: worker\ndescription: worker\nplanModeRequired: 'yes'", "planModeRequired"),
        ("name: worker\ndescription: worker\nunknown: value", "unknown"),
    ],
)
def test_parse_definition_rejects_invalid_metadata(
    frontmatter: str,
    message: str,
) -> None:
    with pytest.raises(DefinitionParseError, match=message):
        parse_definition(definition_bytes(frontmatter), "bad.md", Source.PROJECT)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"name: worker\n---\nbody", "start"),
        (b"---\nname: worker\nbody", "not closed"),
        (b"---\nname: [\n---\nbody", "invalid YAML"),
        (b"---\n- item\n---\nbody", "mapping"),
        (definition_bytes("name: worker\ndescription: Worker", "   "), "body"),
        (b"\xff\xfe", "UTF-8"),
    ],
)
def test_parse_frontmatter_rejects_malformed_data(data: bytes, message: str) -> None:
    with pytest.raises(DefinitionParseError, match=message):
        parse_frontmatter_and_body(data)


def test_parse_file_records_absolute_path(tmp_path: Path) -> None:
    path = tmp_path / "worker.md"
    path.write_bytes(definition_bytes("name: worker\ndescription: Worker"))

    definition = parse_file(path, Source.USER)

    assert definition.file_path == str(path.resolve())
    assert definition.source is Source.USER


def test_parse_file_rejects_missing_directory_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(DefinitionParseError, match="regular"):
        parse_file(tmp_path / "missing.md", Source.USER)
    with pytest.raises(DefinitionParseError, match="regular"):
        parse_file(tmp_path, Source.USER)

    target = tmp_path / "target.md"
    target.write_bytes(definition_bytes("name: worker\ndescription: Worker"))
    link = tmp_path / "link.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(DefinitionParseError, match="symbolic"):
        parse_file(link, Source.USER)


def test_parse_definition_rejects_wrong_api_types() -> None:
    data = definition_bytes("name: worker\ndescription: Worker")
    with pytest.raises(TypeError, match="file_path"):
        parse_definition(data, object(), Source.USER)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="source"):
        parse_definition(data, "worker.md", "user")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bytes"):
        parse_frontmatter_and_body("text")  # type: ignore[arg-type]
