"""Compiled matchers shared by permission rules and lifecycle hooks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class Matcher(Protocol):
    """Match one normalized string value without mutating runtime state."""

    def match(self, value: str) -> bool:
        """Return whether *value* satisfies this matcher."""
        ...

    def __str__(self) -> str:
        """Return the stable user-facing matcher expression."""
        ...


@dataclass(frozen=True, slots=True)
class ExactMatcher:
    """Match only a value equal to the configured string."""

    value: str

    def match(self, value: str) -> bool:
        return value == self.value

    def __str__(self) -> str:
        return f"={self.value}"


@dataclass(frozen=True, slots=True)
class GlobMatcher:
    """Match Codewright's escaped-star glob language as a command or path."""

    pattern: str
    is_command: bool

    def match(self, value: str) -> bool:
        return match_pattern(self.pattern, value, path_mode=not self.is_command)

    def __str__(self) -> str:
        return self.pattern


@dataclass(frozen=True, slots=True)
class RegexMatcher:
    """Match when a compiled regular expression finds any substring."""

    src: str
    compiled: re.Pattern[str]

    def match(self, value: str) -> bool:
        return self.compiled.search(value) is not None

    def __str__(self) -> str:
        return f"~{self.src}"


@dataclass(frozen=True, slots=True)
class NotMatcher:
    """Invert the result of another matcher."""

    inner: Matcher

    def match(self, value: str) -> bool:
        return not self.inner.match(value)

    def __str__(self) -> str:
        return f"!{self.inner}"


def compile_matcher(pattern: str, *, is_command: bool) -> Matcher:
    """Compile one prefixed matcher expression or legacy glob pattern."""
    if not isinstance(pattern, str):
        raise TypeError("pattern must be a string")
    if not isinstance(is_command, bool):
        raise TypeError("is_command must be a boolean")
    if not pattern:
        raise ValueError("empty matcher pattern")

    prefix, remainder = pattern[0], pattern[1:]
    if prefix == "=":
        return ExactMatcher(remainder)
    if prefix == "~":
        try:
            compiled = re.compile(remainder)
        except re.error as error:
            raise ValueError(f"invalid regex: {error}") from error
        return RegexMatcher(remainder, compiled)
    if prefix == "!":
        return NotMatcher(compile_matcher(remainder, is_command=is_command))
    return GlobMatcher(pattern, is_command)


def match_pattern(
    pattern: str,
    target: str,
    *,
    path_mode: bool | None = None,
) -> bool:
    """Match the documented escaped-star glob language against a whole target."""
    if not isinstance(pattern, str) or not isinstance(target, str):
        raise TypeError("pattern and target must be strings")
    if not pattern:
        return True
    selected_path_mode = "/" in pattern or "/" in target if path_mode is None else path_mode
    expression = _glob_expression(pattern, path_mode=selected_path_mode)
    return re.fullmatch(expression, target) is not None


def _glob_expression(pattern: str, *, path_mode: bool) -> str:
    parts: list[str] = []
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\":
            index += 1
            if index == len(pattern):
                parts.append(re.escape("\\"))
                break
            parts.append(re.escape(pattern[index]))
        elif character == "*":
            is_double = index + 1 < len(pattern) and pattern[index + 1] == "*"
            if is_double:
                index += 1
                if path_mode and index + 1 < len(pattern) and pattern[index + 1] == "/":
                    index += 1
                    parts.append("(?:.*/)?")
                else:
                    parts.append(".*")
            else:
                parts.append("[^/]*" if path_mode else ".*")
        else:
            parts.append(re.escape(character))
        index += 1
    return "".join(parts)


__all__ = [
    "ExactMatcher",
    "GlobMatcher",
    "Matcher",
    "NotMatcher",
    "RegexMatcher",
    "compile_matcher",
    "match_pattern",
]
