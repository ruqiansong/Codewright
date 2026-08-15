"""Tests for vendor-neutral tool execution results."""

import pytest

from codewright.tool import Result, truncate_text


def test_result_defensively_copies_metadata() -> None:
    metadata: dict[str, object] = {"count": 1}
    result = Result("ok", metadata=metadata)

    metadata["count"] = 2

    assert result.metadata == {"count": 1}


def test_result_enforces_error_code_invariant() -> None:
    with pytest.raises(ValueError, match="must have an error_code"):
        Result("failed", is_error=True)
    with pytest.raises(ValueError, match="successful result"):
        Result("ok", error_code="unexpected")


def test_truncate_text_supports_line_and_character_limits() -> None:
    by_line, line_truncated = truncate_text("one\ntwo\nthree\n", max_lines=2)
    by_char, char_truncated = truncate_text("x" * 100, max_chars=24)
    unchanged, unchanged_truncated = truncate_text("short", max_chars=24, max_lines=2)

    assert by_line == "one\ntwo\n[truncated]"
    assert line_truncated is True
    assert len(by_char) == 24
    assert by_char.endswith("[truncated]")
    assert char_truncated is True
    assert (unchanged, unchanged_truncated) == ("short", False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_chars": 1}, "max_chars"),
        ({"max_lines": 0}, "max_lines"),
    ],
)
def test_truncate_text_rejects_invalid_limits(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        truncate_text("text", **kwargs)
