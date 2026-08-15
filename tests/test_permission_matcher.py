"""Tests for compiled exact, glob, regex, and inverse matchers."""

import pytest

from codewright.permission.matcher import (
    ExactMatcher,
    GlobMatcher,
    NotMatcher,
    RegexMatcher,
    compile_matcher,
    match_pattern,
)


@pytest.mark.parametrize(
    ("pattern", "is_command", "value", "expected"),
    [
        ("=git status", True, "git status", True),
        ("=git status", True, "git status -s", False),
        (r"~^npm (install|test)$", True, "npm install", True),
        (r"~^npm (install|test)$", True, "npm run dev", False),
        ("!=foo", False, "foo", False),
        ("!=foo", False, "bar", True),
        (r"!~^rm", True, "rm -rf .", False),
        (r"!~^rm", True, "ls -lh", True),
        ("!git *", True, "git status", False),
        ("!git *", True, "npm install", True),
        ("**/*.py", False, "main.py", True),
        ("**/*.py", False, "src/main.py", True),
        ("src/*.py", False, "src/nested/main.py", False),
    ],
    ids=[
        "exact-hit",
        "exact-miss",
        "regex-hit",
        "regex-miss",
        "not-exact-miss",
        "not-exact-hit",
        "not-regex-miss",
        "not-regex-hit",
        "not-glob-miss",
        "not-glob-hit",
        "path-root-double-star",
        "path-nested-double-star",
        "path-single-star-boundary",
    ],
)
def test_compile_matcher_cases(
    pattern: str,
    is_command: bool,
    value: str,
    expected: bool,
) -> None:
    assert compile_matcher(pattern, is_command=is_command).match(value) is expected


def test_compile_matcher_builds_distinct_immutable_types() -> None:
    assert isinstance(compile_matcher("=x", is_command=False), ExactMatcher)
    assert isinstance(compile_matcher("x*", is_command=False), GlobMatcher)
    assert isinstance(compile_matcher("~x", is_command=False), RegexMatcher)
    assert isinstance(compile_matcher("!=x", is_command=False), NotMatcher)


@pytest.mark.parametrize(
    ("pattern", "message"),
    [("", "empty matcher pattern"), ("~[invalid", "invalid regex")],
)
def test_compile_matcher_rejects_invalid_patterns(pattern: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compile_matcher(pattern, is_command=False)


def test_match_pattern_preserves_escaped_star_and_empty_legacy_behavior() -> None:
    assert match_pattern(r"echo \*", "echo *", path_mode=False)
    assert not match_pattern(r"echo \*", "echo value", path_mode=False)
    assert match_pattern("", "anything", path_mode=False)
