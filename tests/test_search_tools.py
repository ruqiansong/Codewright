"""Behavior tests for bounded glob and content search tools."""

import json
from pathlib import Path

import pytest

from codewright.tool.glob_tool import MAX_RESULTS as GLOB_MAX_RESULTS
from codewright.tool.glob_tool import GlobTool
from codewright.tool.grep_tool import MAX_RESULTS as GREP_MAX_RESULTS
from codewright.tool.grep_tool import GrepTool


def arguments(**values: object) -> str:
    return json.dumps(values)


@pytest.mark.asyncio
async def test_glob_finds_recursive_files_in_stable_order(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.py").write_text("", encoding="utf-8")
    (tmp_path / "nested" / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "nested" / "ignored.txt").write_text("", encoding="utf-8")

    result = await GlobTool(tmp_path).execute(arguments(pattern="**/*.py"))

    assert result.is_error is False
    assert result.content.splitlines() == sorted(result.content.splitlines())
    assert result.content.splitlines()[0].endswith("nested/a.py")
    assert result.content.splitlines()[1].endswith("z.py")


@pytest.mark.asyncio
async def test_glob_handles_no_match_invalid_root_and_result_limit(tmp_path: Path) -> None:
    tool = GlobTool(tmp_path)
    no_match = await tool.execute(arguments(pattern="*.missing"))
    missing_root = await tool.execute(arguments(pattern="*", path="missing"))
    for index in range(GLOB_MAX_RESULTS + 1):
        (tmp_path / f"{index:03d}.py").write_text("", encoding="utf-8")
    limited = await tool.execute(arguments(pattern="*.py"))

    assert no_match.is_error is False
    assert no_match.metadata["matches"] == 0
    assert missing_root.error_code == "path_not_found"
    assert limited.truncated is True
    assert len(limited.content.splitlines()) == GLOB_MAX_RESULTS + 1
    assert limited.content.endswith("[truncated]")


@pytest.mark.asyncio
async def test_grep_returns_file_line_and_content_with_filters(tmp_path: Path) -> None:
    (tmp_path / "keep.py").write_text("first\nneedle here\n", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("needle ignored\n", encoding="utf-8")

    result = await GrepTool(tmp_path).execute(arguments(pattern="needle", glob="**/*.py"))

    assert result.is_error is False
    assert result.content.endswith("keep.py:2:needle here")
    assert result.metadata["matches"] == 1


@pytest.mark.asyncio
async def test_grep_handles_invalid_regex_no_match_and_missing_root(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("text", encoding="utf-8")

    tool = GrepTool(tmp_path)
    invalid = await tool.execute(arguments(pattern="["))
    no_match = await tool.execute(arguments(pattern="absent"))
    missing = await tool.execute(arguments(pattern="text", path="missing"))

    assert invalid.error_code == "invalid_regex"
    assert no_match.is_error is False
    assert no_match.metadata["matches"] == 0
    assert missing.error_code == "path_not_found"


@pytest.mark.asyncio
async def test_grep_limits_matches_and_marks_truncation(tmp_path: Path) -> None:
    path = tmp_path / "many.txt"
    path.write_text("needle\n" * (GREP_MAX_RESULTS + 1), encoding="utf-8")

    result = await GrepTool(tmp_path).execute(arguments(pattern="needle", path=path.name))

    assert result.truncated is True
    assert result.metadata["matches"] == GREP_MAX_RESULTS
    assert result.content.endswith("[truncated]")


@pytest.mark.asyncio
async def test_grep_skips_non_utf8_files_without_failing_search(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\xffneedle")
    (tmp_path / "text.txt").write_text("needle", encoding="utf-8")

    result = await GrepTool(tmp_path).execute(arguments(pattern="needle"))

    assert result.is_error is False
    assert "text.txt:1:needle" in result.content
    assert result.metadata["skipped_files"] == 1


@pytest.mark.asyncio
async def test_search_tools_reject_parent_traversal_patterns(tmp_path: Path) -> None:
    glob_result = await GlobTool(tmp_path).execute(arguments(pattern="../*"))
    grep_result = await GrepTool(tmp_path).execute(arguments(pattern="secret", glob="../*"))

    assert glob_result.error_code == "path_outside_sandbox"
    assert grep_result.error_code == "path_outside_sandbox"


@pytest.mark.asyncio
async def test_search_tools_treat_empty_path_as_sandbox_root(tmp_path: Path) -> None:
    (tmp_path / "match.py").write_text("needle\n", encoding="utf-8")

    glob_result = await GlobTool(tmp_path).execute(arguments(pattern="*.py", path=""))
    grep_result = await GrepTool(tmp_path).execute(arguments(pattern="needle", path=""))

    assert glob_result.is_error is False
    assert glob_result.content.endswith("match.py")
    assert grep_result.is_error is False
    assert "match.py:1:needle" in grep_result.content


@pytest.mark.asyncio
async def test_search_tools_do_not_follow_file_symlinks_outside_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("secret_marker = True\n", encoding="utf-8")
    (root / "linked.py").symlink_to(outside)

    glob_result = await GlobTool(root).execute(arguments(pattern="**/*.py"))
    grep_result = await GrepTool(root).execute(arguments(pattern="secret_marker"))

    assert glob_result.error_code == "path_outside_sandbox"
    assert grep_result.error_code == "path_outside_sandbox"
    assert "secret_marker" not in grep_result.content
