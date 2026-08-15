"""Tests for MCP tools flowing through Codewright permissions."""

import json
from pathlib import Path

import pytest
import yaml

from codewright.llm import ToolCall
from codewright.permission.engine import Engine, new_engine
from codewright.permission.models import Category, Decision, Mode
from codewright.permission.rule import (
    Rule,
    RuleSet,
    is_mcp_tool_name,
    is_mcp_tool_selector,
    parse_rule,
)
from codewright.permission.settings import categorize, extract_target

MCP_READ = "mcp__github__get_issue"
MCP_WRITE = "mcp__github__create_issue"
SYNTHETIC_SECRET = "permission-mcp-secret-not-real"


def call(name: str, arguments: object | None = None) -> ToolCall:
    return ToolCall("call-1", name, json.dumps(arguments if arguments is not None else {}))


def bare_engine(root: Path) -> Engine:
    return Engine(
        root=root.resolve(),
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=root / ".codewright" / "settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


@pytest.mark.parametrize("name", [MCP_READ, "mcp__demo-server__tool_name", "mcp__a__b"])
def test_exact_mcp_tool_names_are_recognized(name: str) -> None:
    assert is_mcp_tool_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "custom",
        "mcp__github",
        "mcp____tool",
        "mcp__github__",
        "mcp__github__bad.tool",
        "mcp__github__" + "x" * 60,
    ],
)
def test_invalid_mcp_tool_names_are_rejected(name: str) -> None:
    assert not is_mcp_tool_name(name)


def test_mcp_rules_accept_exact_and_tool_wildcard_selectors() -> None:
    assert parse_rule(MCP_READ, allow=True) == (Rule(MCP_READ, None, True, MCP_READ), True)
    assert parse_rule("mcp__github__*", allow=True) == (
        Rule("mcp__github__*", None, True, "mcp__github__*"),
        True,
    )
    assert is_mcp_tool_selector("mcp__github__create_*")


@pytest.mark.parametrize(
    "value",
    [
        "mcp__*__get_issue",
        "mcp__github__tool(*)",
        "mcp__github__bad.tool",
        "mcp__github__" + "x" * 60,
    ],
)
def test_mcp_rules_reject_unsafe_selectors(value: str) -> None:
    _rule, valid = parse_rule(value, allow=True)
    assert valid is False


def test_rule_set_matches_mcp_selector_and_prioritizes_exact_deny() -> None:
    rules = RuleSet(
        allow=[Rule("mcp__github__*", None, True, "mcp__github__*")],
        deny=[Rule(MCP_WRITE, None, False, MCP_WRITE)],
    )

    assert rules.match(MCP_READ, "") == (Decision.ALLOW, True)
    assert rules.match(MCP_WRITE, "") == (Decision.DENY, True)
    assert rules.match("mcp__slack__get_issue", "") == (Decision.ALLOW, False)


def test_mcp_category_and_target_use_safe_defaults() -> None:
    assert categorize(MCP_READ, read_only=True) is Category.READ
    assert categorize(MCP_WRITE, read_only=False) is Category.EXEC
    assert extract_target(call(MCP_WRITE, {"title": "example"})) == ("", False, True)
    assert extract_target(ToolCall("bad", MCP_WRITE, "not-json")) == ("", False, False)


def test_mcp_mode_fallback_and_unknown_tool_denial(tmp_path: Path) -> None:
    engine = bare_engine(tmp_path)

    assert engine.check(Mode.DEFAULT, call(MCP_READ), True) == (Decision.ALLOW, "")
    assert engine.check(Mode.DEFAULT, call(MCP_WRITE), False)[0] is Decision.ASK
    assert engine.check(Mode.ACCEPT_EDITS, call(MCP_WRITE), False)[0] is Decision.ASK
    assert engine.check(Mode.PLAN, call(MCP_WRITE), False)[0] is Decision.ASK
    assert engine.check(Mode.BYPASS, call(MCP_WRITE), False) == (Decision.ALLOW, "")
    assert engine.check(Mode.BYPASS, call("unknown"), False)[0] is Decision.DENY
    assert engine.check(Mode.BYPASS, ToolCall("bad", MCP_WRITE, "[]"), False)[0] is (Decision.DENY)


def test_mcp_explicit_rules_apply_before_mode_fallback(tmp_path: Path) -> None:
    engine = bare_engine(tmp_path)
    engine.project.allow.append(Rule("mcp__github__*", None, True, "mcp__github__*"))
    engine.project.deny.append(Rule(MCP_WRITE, None, False, MCP_WRITE))

    assert engine.check(Mode.DEFAULT, call(MCP_READ), True) == (Decision.ALLOW, "")
    decision, reason = engine.check(Mode.BYPASS, call(MCP_WRITE), False)
    assert decision is Decision.DENY
    assert "deny" in reason


def test_mcp_permanent_allow_is_exact_secret_free_and_reloadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    engine = new_engine(root)
    tool_call = call(MCP_WRITE, {"token": SYNTHETIC_SECRET, "title": "example"})

    engine.persist_local_allow(tool_call)
    engine.persist_local_allow(tool_call)

    text = engine.local_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    assert payload["permissions"]["allow"] == [MCP_WRITE]
    assert SYNTHETIC_SECRET not in text
    reloaded = new_engine(root)
    assert reloaded.check(Mode.DEFAULT, tool_call, False) == (Decision.ALLOW, "")


def test_builtin_blacklist_still_precedes_bypass_with_mcp_support(tmp_path: Path) -> None:
    decision, reason = bare_engine(tmp_path).check(
        Mode.BYPASS,
        call("bash", {"command": "rm -rf /"}),
        read_only=False,
    )

    assert decision is Decision.DENY
    assert "黑名单" in reason
