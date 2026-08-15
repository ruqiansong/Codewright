"""Tests for strict zero-argument slash-command parsing."""

import pytest

from codewright.command import Invocation, parse, parse_invocation


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ("", False)),
        ("   ", ("", False)),
        ("hello", ("", False)),
        ("/", ("", True)),
        ("/help", ("help", True)),
        ("  /HELP  ", ("help", True)),
        ("/help xx", ("", True)),
        ("/help  ", ("help", True)),
        ("//double", ("", True)),
        ("/ /help", ("", True)),
        ("/valid-name", ("valid-name", True)),
        ("/valid_name", ("valid_name", True)),
        ("/invalid.name", ("", True)),
    ],
)
def test_parse(value: str, expected: tuple[str, bool]) -> None:
    assert parse(value) == expected


def test_parse_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="input_text"):
        parse(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("hello", Invocation()),
        ("/", Invocation(is_slash=True)),
        ("//bad", Invocation(is_slash=True)),
        ("/ help", Invocation(is_slash=True)),
        ("/HELP", Invocation("help", "", True, True)),
        (" /skill info review ", Invocation("skill", "info review", True, True)),
        ("/test-skill  raw  args", Invocation("test-skill", "raw  args", True, True)),
        ("/invalid.name value", Invocation(is_slash=True)),
    ],
)
def test_parse_invocation_preserves_raw_argument_tail(value: str, expected: Invocation) -> None:
    assert parse_invocation(value) == expected


def test_legacy_parse_continues_to_reject_argument_tails() -> None:
    assert parse("/skill info review") == ("", True)
