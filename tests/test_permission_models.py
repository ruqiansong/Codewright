"""Tests for dependency-free permission types and public mode names."""

import pytest

from codewright.permission import (
    Category,
    Decision,
    Mode,
    Outcome,
    PermissionSetupError,
    parse_mode,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("default", Mode.DEFAULT),
        ("ACCEPTEDITS", Mode.ACCEPT_EDITS),
        (" plan ", Mode.PLAN),
        ("BypassPermissions", Mode.BYPASS),
    ],
)
def test_parse_mode_accepts_public_names_case_insensitively(
    value: str,
    expected: Mode,
) -> None:
    assert parse_mode(value) == (expected, True)


def test_parse_mode_returns_safe_fallback_for_unknown_name() -> None:
    assert parse_mode("unrecognized") == (Mode.DEFAULT, False)


def test_parse_mode_rejects_non_string_values() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        parse_mode(None)  # type: ignore[arg-type]


def test_mode_string_values_are_stable() -> None:
    assert [str(mode) for mode in Mode] == [
        "default",
        "acceptEdits",
        "plan",
        "bypassPermissions",
    ]


def test_public_permission_types_are_distinct_integer_enums() -> None:
    assert Decision.ALLOW != Decision.DENY
    assert Category.READ != Category.WRITE
    assert Outcome.ALLOW_ONCE != Outcome.ALLOW_FOREVER
    assert issubclass(PermissionSetupError, RuntimeError)
