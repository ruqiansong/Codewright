"""Tests for layered lifecycle Hook configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from codewright.hook import ActionType, CombineMode, Event, load
from codewright.permission.matcher import GlobMatcher, NotMatcher, RegexMatcher


def write_hooks(root: Path, hooks: object) -> Path:
    path = root / ".codewright" / "hooks.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"hooks": hooks}, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_loads_valid_rules_actions_conditions_and_durations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    project = tmp_path / "project"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    source = write_hooks(
        project,
        [
            {
                "name": "python-write",
                "event": "PostToolUse",
                "if": {
                    "all_of": [
                        {
                            "field": "tool_input.path",
                            "match": {"type": "glob", "value": "**/*.py"},
                        },
                        {
                            "field": "is_error",
                            "match": {
                                "type": "not",
                                "inner": {"type": "exact", "value": "True"},
                            },
                        },
                    ]
                },
                "action": {"type": "shell", "command": "ruff format"},
                "only_once": True,
                "async": True,
                "timeout": "2.5m",
            },
            {
                "name": "notify",
                "event": "Stop",
                "if": {
                    "any_of": [{"field": "event", "match": {"type": "regex", "value": "^Stop$"}}]
                },
                "action": {
                    "type": "http",
                    "url": "https://example.test/hook",
                    "method": "PUT",
                    "headers": {"X-Hook": "yes"},
                    "body": "{event}",
                },
                "timeout": "1h",
            },
        ],
    )

    engine = load(project)
    try:
        assert engine.sources == [str(source)]
        assert [rule.name for rule in engine.rules] == ["python-write", "notify"]
        first, second = engine.rules
        assert first.event is Event.POST_TOOL_USE
        assert first.action.type is ActionType.SHELL
        assert first.only_once and first.asyncio_mode and first.timeout_s == 150
        assert first.condition is not None and first.condition.mode is CombineMode.ALL_OF
        assert isinstance(first.condition.atoms[0].matcher, GlobMatcher)
        assert not first.condition.atoms[0].matcher.is_command
        assert isinstance(first.condition.atoms[1].matcher, NotMatcher)
        assert second.timeout_s == 3600
        assert second.condition is not None
        assert isinstance(second.condition.atoms[0].matcher, RegexMatcher)
        assert second.action.http is not None
        assert second.action.http.headers == {"X-Hook": "yes"}
    finally:
        await engine.aclose()


@pytest.mark.asyncio
async def test_invalid_entries_are_logged_and_valid_sibling_survives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    write_hooks(
        tmp_path,
        [
            {"name": "", "event": "Stop", "action": {"type": "prompt", "text": "x"}},
            {"name": "event", "event": "Nope", "action": {"type": "prompt", "text": "x"}},
            {"name": "action", "event": "Stop", "action": {"type": "nope"}},
            "not-a-mapping",
            {"name": "valid", "event": "Stop", "action": {"type": "prompt", "text": "ok"}},
        ],
    )

    engine = load(tmp_path)
    try:
        assert [rule.name for rule in engine.rules] == ["valid"]
    finally:
        await engine.aclose()
    errors = capsys.readouterr().err
    assert "name must be a non-empty string" in errors
    assert 'unknown event "Nope"' in errors
    assert "unknown action type 'nope'" in errors
    assert "definition must be a mapping" in errors


@pytest.mark.asyncio
async def test_condition_and_matcher_validation_skips_rules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    bad_conditions = [
        {"all_of": [], "any_of": []},
        {"all_of": []},
        {"all_of": [{"field": "x", "match": {"type": "regex", "value": "["}}]},
        {"all_of": [{"field": "x", "match": {"type": "not"}}]},
        {"all_of": [{"field": "", "match": {"type": "exact", "value": "x"}}]},
    ]
    write_hooks(
        tmp_path,
        [
            {
                "name": f"bad-{index}",
                "event": "Stop",
                "if": condition,
                "action": {"type": "prompt", "text": "x"},
            }
            for index, condition in enumerate(bad_conditions)
        ],
    )

    engine = load(tmp_path)
    try:
        assert engine.rules == []
    finally:
        await engine.aclose()
    errors = capsys.readouterr().err
    assert "exactly one of all_of or any_of" in errors
    assert "must be a non-empty list" in errors
    assert "invalid regex" in errors
    assert "not matcher must contain" in errors
    assert "field must be a non-empty string" in errors


@pytest.mark.asyncio
async def test_async_boolean_action_and_timeout_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    entries = [
        ("blocking", "PreToolUse", {"type": "shell", "command": "true"}, True, "1s"),
        ("prompt", "Stop", {"type": "prompt", "text": "x"}, True, "1s"),
        ("subagent", "Stop", {"type": "subagent", "agent_name": "x", "prompt": "y"}, True, "1s"),
        ("bool", "Stop", {"type": "shell", "command": "true"}, 1, "1s"),
        ("numeric-timeout", "Stop", {"type": "shell", "command": "true"}, False, 2),
        ("zero", "Stop", {"type": "shell", "command": "true"}, False, "0s"),
        ("negative", "Stop", {"type": "shell", "command": "true"}, False, "-1s"),
    ]
    write_hooks(
        tmp_path,
        [
            {"name": name, "event": event, "action": action, "async": mode, "timeout": timeout}
            for name, event, action, mode, timeout in entries
        ],
    )

    engine = load(tmp_path)
    try:
        assert engine.rules == []
    finally:
        await engine.aclose()
    errors = capsys.readouterr().err
    assert "async not allowed for blocking events" in errors
    assert errors.count("async only supports shell/http actions") == 2
    assert "async must be a boolean" in errors
    assert errors.count("timeout must be a positive duration string") == 2
    assert "timeout must be greater than zero" in errors


@pytest.mark.asyncio
async def test_project_precedes_user_and_duplicate_name_is_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    project_path = write_hooks(
        project,
        [{"name": "same", "event": "Stop", "action": {"type": "prompt", "text": "project"}}],
    )
    home_path = write_hooks(
        home,
        [
            {"name": "same", "event": "Stop", "action": {"type": "prompt", "text": "user"}},
            {"name": "user", "event": "Stop", "action": {"type": "prompt", "text": "user"}},
        ],
    )

    engine = load(project)
    try:
        assert engine.sources == [str(project_path), str(home_path)]
        assert [rule.name for rule in engine.rules] == ["same", "user"]
        assert engine.rules[0].action.prompt is not None
        assert engine.rules[0].action.prompt.text == "project"
    finally:
        await engine.aclose()
    assert 'hook "same": duplicate name, skipped' in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, expected",
    [
        ("hooks: [", "invalid YAML"),
        ("- not-a-mapping\n", "root must be a mapping"),
        ("hooks: nope\n", "hooks must be a list"),
    ],
)
async def test_invalid_file_is_logged_and_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: str,
    expected: str,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    path = tmp_path / ".codewright" / "hooks.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    engine = load(tmp_path)
    try:
        assert engine.rules == []
        assert engine.sources == []
    finally:
        await engine.aclose()
    assert expected in capsys.readouterr().err


@pytest.mark.asyncio
async def test_action_field_types_are_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    actions = [
        {"type": "shell", "command": 1},
        {"type": "prompt"},
        {"type": "http", "url": 1},
        {"type": "http", "url": "x", "method": 1},
        {"type": "http", "url": "x", "headers": {"X": 1}},
        {"type": "http", "url": "x", "body": 1},
        {"type": "subagent", "agent_name": "x"},
    ]
    write_hooks(
        tmp_path,
        [
            {"name": f"bad-{index}", "event": "Stop", "action": action}
            for index, action in enumerate(actions)
        ],
    )
    engine = load(tmp_path)
    try:
        assert engine.rules == []
    finally:
        await engine.aclose()
    errors = capsys.readouterr().err
    for field in ("command", "text", "url", "method", "headers", "body", "prompt"):
        assert field in errors
