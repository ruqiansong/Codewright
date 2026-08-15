"""Behavior tests for read, write, and exact edit tools."""

import json
from pathlib import Path

import pytest

from codewright.tool.edit_file import EditFileTool
from codewright.tool.read_file import MAX_LINES, ReadFileTool
from codewright.tool.write_file import WriteFileTool


def arguments(**values: object) -> str:
    return json.dumps(values)


def test_edit_file_description_requires_reading_before_editing() -> None:
    assert "read_file first" in EditFileTool.description
    assert "old_string is unique" in EditFileTool.description


@pytest.mark.asyncio
async def test_read_file_returns_numbered_utf8_content(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("alpha\nβeta\n", encoding="utf-8")

    result = await ReadFileTool().execute(arguments(path=str(path)))

    assert result.is_error is False
    assert result.content == "     1\talpha\n     2\tβeta"
    assert result.metadata["lines"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "error_code"),
    [
        ("not-json", "invalid_arguments"),
        ("[]", "invalid_arguments"),
        ("{}", "invalid_arguments"),
    ],
)
async def test_read_file_rejects_invalid_arguments(payload: str, error_code: str) -> None:
    result = await ReadFileTool().execute(payload)

    assert result.is_error is True
    assert result.error_code == error_code


@pytest.mark.asyncio
async def test_read_file_returns_structured_path_and_encoding_errors(tmp_path: Path) -> None:
    missing = await ReadFileTool().execute(arguments(path=str(tmp_path / "missing.txt")))
    directory = await ReadFileTool().execute(arguments(path=str(tmp_path)))
    binary_path = tmp_path / "binary.bin"
    binary_path.write_bytes(b"\xff\xfe")
    binary = await ReadFileTool().execute(arguments(path=str(binary_path)))

    assert missing.error_code == "file_not_found"
    assert directory.error_code == "not_a_file"
    assert binary.error_code == "encoding_error"
    assert all(result.is_error for result in (missing, directory, binary))


@pytest.mark.asyncio
async def test_read_file_truncates_large_files(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("".join(f"line {index}\n" for index in range(MAX_LINES + 1)), encoding="utf-8")

    result = await ReadFileTool().execute(arguments(path=str(path)))

    assert result.truncated is True
    assert result.content.endswith("[truncated]")
    assert result.metadata["lines"] == MAX_LINES


@pytest.mark.asyncio
async def test_write_file_creates_parents_overwrites_and_allows_empty_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a" / "b" / "file.txt"
    tool = WriteFileTool()

    first = await tool.execute(arguments(path=str(path), content="hello β"))
    second = await tool.execute(arguments(path=str(path), content=""))

    assert first.is_error is False
    assert first.metadata["bytes"] == len("hello β".encode())
    assert second.is_error is False
    assert path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_write_file_returns_structured_argument_and_path_errors(tmp_path: Path) -> None:
    missing_content = await WriteFileTool().execute(arguments(path="file.txt"))
    target_directory = await WriteFileTool().execute(arguments(path=str(tmp_path), content="x"))
    parent_file = tmp_path / "parent"
    parent_file.write_text("x", encoding="utf-8")
    parent_conflict = await WriteFileTool().execute(
        arguments(path=str(parent_file / "child.txt"), content="x")
    )

    assert missing_content.error_code == "invalid_arguments"
    assert target_directory.error_code == "not_a_file"
    assert parent_conflict.error_code == "write_failed"


@pytest.mark.asyncio
async def test_edit_file_replaces_exactly_one_match(tmp_path: Path) -> None:
    path = tmp_path / "edit.txt"
    path.write_text("before unique after", encoding="utf-8")

    result = await EditFileTool().execute(
        arguments(path=str(path), old_string="unique", new_string="updated")
    )

    assert result.is_error is False
    assert result.metadata["match_count"] == 1
    assert path.read_text(encoding="utf-8") == "before updated after"


@pytest.mark.asyncio
async def test_edit_file_zero_and_multiple_matches_do_not_change_file(tmp_path: Path) -> None:
    path = tmp_path / "edit.txt"
    original = "same / same"
    path.write_text(original, encoding="utf-8")
    tool = EditFileTool()

    missing = await tool.execute(
        arguments(path=str(path), old_string="absent", new_string="replacement")
    )
    multiple = await tool.execute(
        arguments(path=str(path), old_string="same", new_string="replacement")
    )

    assert missing.error_code == "match_not_found"
    assert missing.metadata["match_count"] == 0
    assert multiple.error_code == "match_not_unique"
    assert multiple.metadata["match_count"] == 2
    assert "2" in multiple.content
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.asyncio
async def test_edit_file_rejects_empty_old_string_and_missing_file(tmp_path: Path) -> None:
    invalid = await EditFileTool().execute(
        arguments(path="file.txt", old_string="", new_string="new")
    )
    missing = await EditFileTool().execute(
        arguments(path=str(tmp_path / "missing"), old_string="old", new_string="new")
    )

    assert invalid.error_code == "invalid_arguments"
    assert missing.error_code == "file_not_found"
