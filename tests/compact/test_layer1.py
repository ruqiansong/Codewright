"""Tests for preventive tool-result offloading."""

import os
from pathlib import Path

import pytest

from codewright.compact.layer1 import build_preview, offload_and_snip, spill_single
from codewright.compact.state import ContentReplacementState, SessionContext
from codewright.llm import Message, MessageRole, ToolResult


def session(tmp_path: Path) -> SessionContext:
    return SessionContext("test-session", str(tmp_path / "tool-results"))


def tool_message(*contents: str) -> Message:
    results = tuple(
        ToolResult(f"call-{index}", "read_file", content) for index, content in enumerate(contents)
    )
    return Message(MessageRole.TOOL, "", tool_results=results)


@pytest.mark.parametrize("tool_use_id", ["../escape", "/tmp/escape", "a/b", "a\\b"])
def test_spill_single_contains_untrusted_ids(tmp_path: Path, tool_use_id: str) -> None:
    context = session(tmp_path)

    result_path = Path(spill_single(context, tool_use_id, "content"))

    assert result_path.parent == Path(context.spill_dir).resolve()
    assert result_path.name != tool_use_id
    assert result_path.read_text() == "content"


def test_spill_single_is_idempotent_and_rejects_conflicting_content(tmp_path: Path) -> None:
    context = session(tmp_path)
    result_path = Path(spill_single(context, "call-1", "content"))
    original_mtime = result_path.stat().st_mtime_ns

    assert spill_single(context, "call-1", "content") == str(result_path)
    assert result_path.stat().st_mtime_ns == original_mtime
    with pytest.raises(OSError, match="conflicts"):
        spill_single(context, "call-1", "different")
    assert result_path.read_text() == "content"


def test_spill_single_cleans_up_temporary_file_on_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = session(tmp_path)

    def fail_replace(source: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        raise OSError("publish failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="publish failed"):
        spill_single(context, "call-1", "content")
    assert list(Path(context.spill_dir).glob(".spill-*")) == []


def test_build_preview_is_stable() -> None:
    preview = build_preview(12, "first\nsecond", "/safe/path")

    assert preview == build_preview(12, "first\nsecond", "/safe/path")
    assert "original size: 12 bytes" in preview
    assert "[saved to] /safe/path" in preview
    assert "[head preview]" in preview
    assert "文件读取工具" in preview


def test_offload_single_large_result_and_reuse_preview(tmp_path: Path) -> None:
    context = session(tmp_path)
    state = ContentReplacementState()
    original = tool_message("行\n" * 30_000)

    first = offload_and_snip([original], state, context)
    second = offload_and_snip(first, state, context)

    preview = first[0].tool_results[0].content
    assert first == second
    assert preview != original.tool_results[0].content
    assert "original size:" in preview
    head = preview.split("[head preview]\n", 1)[1].split("\n完整内容", 1)[0]
    assert len(head.encode("utf-8")) <= 2_048
    assert len(head.splitlines()) <= 20
    spilled = list(Path(context.spill_dir).iterdir())
    assert len(spilled) == 1
    assert spilled[0].read_text() == original.tool_results[0].content


def test_offload_aggregate_replaces_minimum_largest_result(tmp_path: Path) -> None:
    context = session(tmp_path)
    original = tool_message(*(size * "x" for size in (49_000, 48_000, 47_000, 46_000, 45_000)))

    updated = offload_and_snip([original], ContentReplacementState(), context)[0]

    changed = [
        before.content != after.content
        for before, after in zip(original.tool_results, updated.tool_results, strict=True)
    ]
    assert changed == [True, False, False, False, False]
    assert sum(len(result.content.encode()) for result in updated.tool_results[1:]) == 186_000


def test_offload_failure_keeps_result_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = session(tmp_path)
    state = ContentReplacementState()
    original = tool_message("x" * 60_000)

    def fail_spill(*args: object, **kwargs: object) -> str:
        raise OSError("disk unavailable")

    monkeypatch.setattr("codewright.compact.layer1.spill_single", fail_spill)
    assert offload_and_snip([original], state, context) == [original]
    monkeypatch.undo()

    updated = offload_and_snip([original], state, context)
    assert updated != [original]
