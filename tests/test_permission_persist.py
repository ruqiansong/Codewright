"""Tests for safe exact allow-rule persistence."""

import json
from pathlib import Path

import pytest
import yaml

from codewright.llm import ToolCall
from codewright.permission.engine import new_engine
from codewright.permission.matcher import ExactMatcher
from codewright.permission.models import Decision, Mode
from codewright.permission.persist import rule_for


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall("call-1", name, json.dumps(arguments))


def test_rule_for_escapes_wildcards_and_backslashes(tmp_path: Path) -> None:
    tool_call = call("bash", command=r"echo * C:\temp")
    rule, serialized, ok = rule_for(tmp_path, tool_call)

    assert ok
    assert rule.tool == "Bash"
    assert rule.matcher == ExactMatcher(r"echo * C:\temp")
    assert r"\*" in serialized
    assert r"C:\\temp" in serialized


def test_persist_local_allow_is_atomic_idempotent_and_reloadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    engine = new_engine(root)
    tool_call = call("write_file", path="src/generated.py", content="value = 1")

    engine.persist_local_allow(tool_call)
    engine.persist_local_allow(tool_call)

    payload = yaml.safe_load(engine.local_path.read_text(encoding="utf-8"))
    assert payload["permissions"]["allow"] == ["Write(src/generated.py)"]
    reloaded = new_engine(root)
    assert reloaded.check(Mode.DEFAULT, tool_call, False) == (Decision.ALLOW, "")


def test_persist_refuses_unsafe_or_malformed_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    engine = new_engine(root)

    with pytest.raises(ValueError, match="safe persistent"):
        engine.persist_local_allow(call("read_file", path="../outside.txt"))
    assert not engine.local_path.exists()


def test_invalid_existing_local_settings_are_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    config_dir = root / ".codewright"
    config_dir.mkdir(parents=True)
    local_path = config_dir / "settings.local.yaml"
    local_path.write_text("permissions: [", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home)
    engine = new_engine(root)

    with pytest.raises(ValueError):
        engine.persist_local_allow(call("bash", command="git status"))
    assert local_path.read_text(encoding="utf-8") == "permissions: ["
