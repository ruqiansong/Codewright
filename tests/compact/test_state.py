"""Tests for persistent-session identity and path construction."""

import re
from datetime import datetime
from pathlib import Path

import pytest

from codewright.compact import (
    new_session_context,
    open_session_context,
    parse_session_time,
)


def test_new_session_context_uses_readable_id_and_nested_paths(tmp_path: Path) -> None:
    context = new_session_context(str(tmp_path))

    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{4}", context.session_id)
    expected_dir = tmp_path / ".codewright" / "sessions" / context.session_id
    assert Path(context.session_dir) == expected_dir
    assert Path(context.spill_dir) == expected_dir / "tool-results"
    assert not expected_dir.exists()


def test_parse_session_time_requires_complete_valid_id() -> None:
    assert parse_session_time("20260812-142305-a1b2") == datetime(2026, 8, 12, 14, 23, 5)

    for invalid in (
        "1717000000-abc12345",
        "20260812-142305-a1b2-extra",
        "20261312-142305-a1b2",
        "20260812-252305-a1b2",
        "20260812-142305-A1B2",
        "../../20260812-142305-a1b2",
    ):
        with pytest.raises(ValueError):
            parse_session_time(invalid)


def test_open_session_context_requires_existing_contained_directory(tmp_path: Path) -> None:
    session_id = "20260812-142305-a1b2"
    session_dir = tmp_path / ".codewright" / "sessions" / session_id
    session_dir.mkdir(parents=True)

    context = open_session_context(str(tmp_path), session_id)

    assert Path(context.session_dir) == session_dir
    assert Path(context.spill_dir) == session_dir / "tool-results"
    with pytest.raises(FileNotFoundError):
        open_session_context(str(tmp_path), "20260812-142306-a1b3")


def test_session_helpers_validate_argument_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        new_session_context(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        parse_session_time(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        open_session_context(123, "20260812-142305-a1b2")  # type: ignore[arg-type]
