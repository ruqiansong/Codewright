"""Tests for permission rule parsing and glob matching."""

import pytest

from codewright.permission.matcher import GlobMatcher
from codewright.permission.models import Decision
from codewright.permission.rule import (
    Rule,
    RuleSet,
    match_pattern,
    parse_rule,
    parse_rule_detailed,
)


def test_parse_rule_normalizes_tool_and_preserves_pattern() -> None:
    assert parse_rule(" bash(git *) ", allow=True) == (
        Rule("Bash", GlobMatcher("git *", True), True, "git *"),
        True,
    )
    assert parse_rule("Read") == (Rule("Read", None, False), True)


@pytest.mark.parametrize("value", ["", "Unknown(*)", "Read(", "Read)", "Read(src))", "Read((src)"])
def test_parse_rule_rejects_invalid_values(value: str) -> None:
    rule, valid = parse_rule(value)
    assert not valid
    assert rule.tool == ""


def test_parse_rule_accepts_balanced_parentheses_in_command() -> None:
    assert parse_rule("Bash(echo $(pwd))", allow=True) == (
        Rule(
            "Bash",
            GlobMatcher("echo $(pwd)", True),
            True,
            "echo $(pwd)",
        ),
        True,
    )


def test_command_patterns_match_whole_target() -> None:
    assert match_pattern("git *", "git status", path_mode=False)
    assert not match_pattern("git *", "npm install", path_mode=False)
    assert match_pattern(r"echo \*", "echo *", path_mode=False)
    assert not match_pattern(r"echo \*", "echo value", path_mode=False)


def test_path_patterns_distinguish_single_and_double_star() -> None:
    assert match_pattern("src/**", "src/a/b.py", path_mode=True)
    assert match_pattern("**/*.py", "main.py", path_mode=True)
    assert match_pattern("**/*.py", "src/main.py", path_mode=True)
    assert not match_pattern("src/*.py", "src/a/main.py", path_mode=True)
    assert not match_pattern("src/**", "docs/main.py", path_mode=True)


def test_rule_set_prioritizes_deny_over_allow() -> None:
    rules = RuleSet(
        allow=[Rule("Bash", GlobMatcher("git *", True), True, "git *")],
        deny=[
            Rule(
                "Bash",
                GlobMatcher("git push *", True),
                False,
                "git push *",
            )
        ],
    )

    assert rules.match("Bash", "git push origin main") == (Decision.DENY, True)
    assert rules.match("Bash", "git status") == (Decision.ALLOW, True)
    assert rules.match("Bash", "npm test") == (Decision.ALLOW, False)


@pytest.mark.parametrize(
    ("source", "hit", "miss"),
    [
        ("Bash(=git status)", "git status", "git status -s"),
        (r"Bash(~^npm.*)", "npm test", "pnpm test"),
        (r"Bash(!~^rm)", "ls -lh", "rm -rf ."),
        ("Write(**/*.py)", "src/main.py", "src/main.ts"),
    ],
)
def test_rule_matcher_syntax(source: str, hit: str, miss: str) -> None:
    rule, valid = parse_rule(source, allow=True)
    assert valid
    assert RuleSet(allow=[rule]).match(rule.tool, hit) == (Decision.ALLOW, True)
    assert RuleSet(allow=[rule]).match(rule.tool, miss) == (Decision.ALLOW, False)


def test_detailed_parser_reports_invalid_regex_without_breaking_compatibility() -> None:
    parsed, error = parse_rule_detailed("Bash(~[invalid)")
    assert parsed is None
    assert error is not None and "invalid regex" in error
    sentinel, valid = parse_rule("Bash(~[invalid)")
    assert not valid
    assert sentinel == Rule("", None, False)
