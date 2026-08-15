"""Tests for layered permission-engine decisions and configuration loading."""

import json
from pathlib import Path

import pytest

from codewright.llm import ToolCall
from codewright.permission import PermissionSetupError
from codewright.permission.engine import Engine, mode_fallback, new_engine
from codewright.permission.matcher import ExactMatcher, GlobMatcher
from codewright.permission.models import Category, Decision, Mode
from codewright.permission.rule import Rule, RuleSet


def call(name: str, **arguments: object) -> ToolCall:
    return ToolCall("call-1", name, json.dumps(arguments))


def bare_engine(root: Path) -> Engine:
    return Engine(
        root=root.resolve(),
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=root / ".codewright" / "settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


@pytest.mark.parametrize(
    ("mode", "category", "expected"),
    [
        (Mode.DEFAULT, Category.READ, Decision.ALLOW),
        (Mode.DEFAULT, Category.WRITE, Decision.ASK),
        (Mode.DEFAULT, Category.EXEC, Decision.ASK),
        (Mode.ACCEPT_EDITS, Category.WRITE, Decision.ALLOW),
        (Mode.ACCEPT_EDITS, Category.EXEC, Decision.ASK),
        (Mode.PLAN, Category.WRITE, Decision.ASK),
        (Mode.BYPASS, Category.WRITE, Decision.ALLOW),
        (Mode.BYPASS, Category.EXEC, Decision.ALLOW),
    ],
)
def test_mode_fallback_matrix(mode: Mode, category: Category, expected: Decision) -> None:
    assert mode_fallback(mode, category) is expected


@pytest.mark.parametrize("mode", list(Mode))
@pytest.mark.parametrize(
    "tool_call",
    [
        ToolCall("x", "unknown", "{}"),
        ToolCall("x", "read_file", "not-json"),
        ToolCall("x", "write_file", '{"path":"a.txt"}'),
        ToolCall("x", "bash", '{"command":42}'),
    ],
)
def test_unknown_and_invalid_calls_are_denied_in_every_mode(
    tmp_path: Path, mode: Mode, tool_call: ToolCall
) -> None:
    decision, reason = bare_engine(tmp_path).check(mode, tool_call, read_only=False)
    assert decision is Decision.DENY
    assert reason


def test_blacklist_and_sandbox_precede_explicit_allow_rules(tmp_path: Path) -> None:
    engine = bare_engine(tmp_path)
    engine.local.allow.extend([Rule("Bash", None, True), Rule("Read", None, True)])

    dangerous = engine.check(Mode.BYPASS, call("bash", command="rm -rf /"), False)
    escaped = engine.check(
        Mode.BYPASS,
        call("read_file", path=str(tmp_path.parent / "outside.txt")),
        True,
    )

    assert dangerous[0] is Decision.DENY
    assert "黑名单" in dangerous[1]
    assert escaped[0] is Decision.DENY
    assert "项目目录之外" in escaped[1]


def test_rules_use_local_project_user_precedence_and_same_layer_deny(
    tmp_path: Path,
) -> None:
    engine = bare_engine(tmp_path)
    engine.user.allow.append(Rule("Write", GlobMatcher("src/**", False), True, "src/**"))
    engine.project.deny.append(Rule("Write", GlobMatcher("src/**", False), False, "src/**"))
    engine.local.allow.append(Rule("Write", ExactMatcher("src/safe.py"), True, "=src/safe.py"))
    engine.local.deny.append(
        Rule("Write", ExactMatcher("src/blocked.py"), False, "=src/blocked.py")
    )

    safe = engine.check(
        Mode.DEFAULT,
        call("write_file", path="src/safe.py", content="ok"),
        False,
    )
    blocked = engine.check(
        Mode.BYPASS,
        call("write_file", path="src/blocked.py", content="no"),
        False,
    )
    project_denied = engine.check(
        Mode.BYPASS,
        call("write_file", path="src/other.py", content="no"),
        False,
    )

    assert safe == (Decision.ALLOW, "")
    assert blocked[0] is Decision.DENY
    assert project_denied[0] is Decision.DENY


def test_mode_fallback_runs_after_no_rule_match(tmp_path: Path) -> None:
    engine = bare_engine(tmp_path)
    read = engine.check(Mode.DEFAULT, call("read_file", path="README.md"), True)
    write = engine.check(
        Mode.DEFAULT,
        call("write_file", path="new.txt", content="content"),
        False,
    )

    assert read == (Decision.ALLOW, "")
    assert write[0] is Decision.ASK
    assert "default" in write[1]


def test_install_skill_is_known_side_effecting_tool_requiring_approval(tmp_path: Path) -> None:
    engine = bare_engine(tmp_path)
    install = call("install_skill", url="https://skills.sh/owner/repo/review")

    assert engine.check(Mode.DEFAULT, install, read_only=False)[0] is Decision.ASK
    assert engine.check(Mode.ACCEPT_EDITS, install, read_only=False)[0] is Decision.ASK
    assert engine.check(Mode.BYPASS, install, read_only=False)[0] is Decision.ALLOW


def test_search_pattern_is_checked_before_mode_bypass(tmp_path: Path) -> None:
    decision, reason = bare_engine(tmp_path).check(
        Mode.BYPASS,
        call("glob", pattern="../*"),
        True,
    )
    assert decision is Decision.DENY
    assert "搜索模式" in reason


def test_new_engine_loads_layers_and_uses_default_mode_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    home = tmp_path / "home"
    (home / ".codewright").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: home)
    (home / ".codewright" / "settings.yaml").write_text(
        "default_mode: plan\npermissions:\n  allow: [Read]\n",
        encoding="utf-8",
    )
    config_dir = root / ".codewright"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        "default_mode: acceptEdits\npermissions:\n  deny: [Bash]\n",
        encoding="utf-8",
    )
    (config_dir / "settings.local.yaml").write_text(
        "default_mode: invalid\npermissions:\n  allow: [Write(src/**)]\n",
        encoding="utf-8",
    )

    engine = new_engine(root)

    assert engine.root == root.resolve()
    assert engine.default_mode is Mode.ACCEPT_EDITS
    assert len(engine.user.allow) == 1
    assert len(engine.project.deny) == 1
    assert len(engine.local.allow) == 1


def test_new_engine_degrades_invalid_layer_but_requires_valid_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    config_dir = root / ".codewright"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text("permissions: [", encoding="utf-8")

    engine = new_engine(root)
    assert engine.project.allow == []
    assert engine.default_mode is Mode.DEFAULT

    with pytest.raises(PermissionSetupError):
        new_engine(tmp_path / "missing")
