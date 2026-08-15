"""Strict zero-argument slash-command input parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$", re.IGNORECASE)
_ASCII_WHITESPACE = frozenset(" \t\n\r\f\v")


@dataclass(frozen=True, slots=True)
class Invocation:
    """One parsed input with an optional raw zero-tokenized argument tail."""

    name: str = ""
    args: str = ""
    is_slash: bool = False
    valid: bool = False


def parse_invocation(input_text: str) -> Invocation:
    """Parse a slash name and preserve its trimmed, un-tokenized argument tail."""
    if not isinstance(input_text, str):
        raise TypeError("input_text must be a string")
    value = input_text.strip()
    if not value or not value.startswith("/"):
        return Invocation()

    candidate = value[1:]
    if not candidate or candidate.startswith("/") or candidate[0] in _ASCII_WHITESPACE:
        return Invocation(is_slash=True)

    split_at = next(
        (index for index, character in enumerate(candidate) if character in _ASCII_WHITESPACE),
        len(candidate),
    )
    name = candidate[:split_at]
    args = candidate[split_at:].strip() if split_at < len(candidate) else ""
    if _NAME_PATTERN.fullmatch(name) is None:
        return Invocation(is_slash=True)
    return Invocation(name.casefold(), args, is_slash=True, valid=True)


def parse(input_text: str) -> tuple[str, bool]:
    """Return the legacy zero-argument parse result used before CH10."""
    invocation = parse_invocation(input_text)
    name = invocation.name if invocation.valid and not invocation.args else ""
    return name, invocation.is_slash


__all__ = ["Invocation", "parse", "parse_invocation"]
