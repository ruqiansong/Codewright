"""Tests for permission settings and built-in tool argument mapping."""

from pathlib import Path

import pytest

from codewright.llm import ToolCall
from codewright.permission.models import Category
from codewright.permission.settings import (
    SettingsError,
    categorize,
    extract_target,
    friendly_name,
    load_settings,
    search_pattern_safe,
    to_rule_set,
)


def call(name: str, arguments: str = "{}") -> ToolCall:
    return ToolCall("call-1", name, arguments)


def test_missing_settings_file_loads_empty(tmp_path: Path) -> None:
    settings = load_settings(tmp_path / "missing.yaml")
    assert settings.default_mode == ""
    assert settings.permissions.allow == []
    assert settings.permissions.deny == []


def test_load_settings_validates_yaml_shape(tmp_path: Path) -> None:
    valid = tmp_path / "valid.yaml"
    valid.write_text(
        "default_mode: plan\npermissions:\n  allow:\n    - Read(src/**)\n"
        "  deny:\n    - Bash(rm *)\n",
        encoding="utf-8",
    )
    settings = load_settings(valid)
    assert settings.default_mode == "plan"
    assert settings.permissions.allow == ["Read(src/**)"]
    assert settings.permissions.deny == ["Bash(rm *)"]

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("permissions: [", encoding="utf-8")
    with pytest.raises(SettingsError, match="Unable to load"):
        load_settings(invalid)

    wrong_shape = tmp_path / "wrong.yaml"
    wrong_shape.write_text("permissions:\n  allow: Read\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="list of strings"):
        load_settings(wrong_shape)


def test_to_rule_set_reports_and_skips_invalid_individual_rules(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        "permissions:\n  allow: [Read(src/**), Broken(*)]\n  deny: [Bash(rm *), 'Edit(']\n",
        encoding="utf-8",
    )
    rules = to_rule_set(load_settings(path))
    assert [(rule.tool, rule.allow) for rule in rules.allow] == [("Read", True)]
    assert [(rule.tool, rule.allow) for rule in rules.deny] == [("Bash", False)]
    stderr = capsys.readouterr().err
    assert "rule 'Broken(*)' parse failed" in stderr
    assert "rule 'Edit(' parse failed" in stderr


def test_to_rule_set_reports_bad_regex_without_rejecting_other_rules(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text(
        "permissions:\n  allow: ['Bash(~[invalid)', 'Bash(=git status)']\n",
        encoding="utf-8",
    )

    rules = to_rule_set(load_settings(path))

    assert len(rules.allow) == 1
    assert rules.allow[0].matcher is not None
    assert rules.allow[0].matcher.match("git status")
    assert "invalid regex" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("internal", "friendly", "category"),
    [
        ("bash", "Bash", Category.EXEC),
        ("read_file", "Read", Category.READ),
        ("write_file", "Write", Category.WRITE),
        ("edit_file", "Edit", Category.WRITE),
        ("glob", "Glob", Category.READ),
        ("grep", "Grep", Category.READ),
        ("load_skill", "LoadSkill", Category.READ),
        ("install_skill", "InstallSkill", Category.EXEC),
        ("Agent", "Agent", Category.EXEC),
        ("TaskList", "TaskList", Category.READ),
        ("TaskGet", "TaskGet", Category.READ),
        ("TaskStop", "TaskStop", Category.EXEC),
        ("SendMessage", "SendMessage", Category.EXEC),
    ],
)
def test_builtin_tool_mappings(internal: str, friendly: str, category: Category) -> None:
    assert friendly_name(internal) == friendly
    assert categorize(internal, read_only=category is Category.READ) is category


def test_unknown_tool_is_not_silently_categorized() -> None:
    assert friendly_name("custom") == "custom"
    with pytest.raises(ValueError, match="Unknown"):
        categorize("custom", read_only=True)


@pytest.mark.parametrize(
    ("tool_call", "expected"),
    [
        (call("read_file", '{"path":"README.md"}'), ("README.md", True, True)),
        (
            call("write_file", '{"path":"a.txt","content":""}'),
            ("a.txt", True, True),
        ),
        (
            call(
                "edit_file",
                '{"path":"a.txt","old_string":"old","new_string":""}',
            ),
            ("a.txt", True, True),
        ),
        (call("glob", '{"pattern":"**/*.py"}'), (".", True, True)),
        (call("glob", '{"pattern":"**/*.py","path":""}'), (".", True, True)),
        (
            call("grep", '{"pattern":"needle","path":"src"}'),
            ("src", True, True),
        ),
        (call("bash", '{"command":"git status"}'), ("git status", False, True)),
        (call("load_skill", '{"name":"review"}'), ("review", False, True)),
        (
            call("install_skill", '{"url":"https://skills.sh/o/r/s"}'),
            ("https://skills.sh/o/r/s", False, True),
        ),
        (call("TaskList", "{}"), ("", False, True)),
        (call("TaskGet", '{"task_id":"task-1"}'), ("task-1", False, True)),
        (call("TaskStop", '{"task_id":"task-1"}'), ("task-1", False, True)),
        (
            call("SendMessage", '{"name":"worker","message":"more"}'),
            ("worker", False, True),
        ),
        (
            call("Agent", '{"prompt":"work","description":"review"}'),
            ("", False, True),
        ),
    ],
)
def test_extract_target_for_builtin_tools(
    tool_call: ToolCall, expected: tuple[str, bool, bool]
) -> None:
    assert extract_target(tool_call) == expected


@pytest.mark.parametrize(
    "tool_call",
    [
        call("read_file", "not-json"),
        call("read_file", "[]"),
        call("read_file", "{}"),
        call("write_file", '{"path":"a.txt"}'),
        call("edit_file", '{"path":"a.txt","old_string":"old"}'),
        call("glob", '{"pattern":""}'),
        call("grep", '{"pattern":42}'),
        call("bash", '{"command":null}'),
        call("unknown", "{}"),
    ],
)
def test_extract_target_rejects_malformed_arguments(tool_call: ToolCall) -> None:
    assert extract_target(tool_call)[2] is False


@pytest.mark.parametrize(
    ("tool_call", "safe"),
    [
        (call("glob", '{"pattern":"**/*.py"}'), True),
        (call("glob", '{"pattern":"../*"}'), False),
        (call("glob", '{"pattern":"/tmp/*"}'), False),
        (call("glob", '{"pattern":"C:\\\\tmp\\\\*"}'), False),
        (call("grep", '{"pattern":"x"}'), True),
        (call("grep", '{"pattern":"x","glob":"../*.py"}'), False),
        (call("bash", '{"command":"pwd"}'), True),
    ],
)
def test_search_pattern_safety(tool_call: ToolCall, safe: bool) -> None:
    assert search_pattern_safe(tool_call) is safe
