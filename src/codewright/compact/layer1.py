"""Deterministic offloading and preview replacement for large tool results."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import replace
from pathlib import Path

from codewright.compact.const import (
    MESSAGE_AGGREGATE_LIMIT,
    PREVIEW_HEAD_BYTES,
    PREVIEW_HEAD_LINES,
    SINGLE_RESULT_LIMIT,
)
from codewright.compact.state import ContentReplacementState, SessionContext
from codewright.llm import Message, MessageRole


def _safe_result_name(tool_use_id: str) -> str:
    """Map an untrusted tool call id to a deterministic path-safe name."""
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        raise ValueError("tool_use_id must be a non-empty string")
    return hashlib.sha256(tool_use_id.encode("utf-8")).hexdigest()


def _spill_path(session: SessionContext, tool_use_id: str) -> Path:
    """Return a contained spill path for one untrusted tool call id."""
    spill_root = Path(session.spill_dir).resolve()
    target = (spill_root / _safe_result_name(tool_use_id)).resolve()
    if target.parent != spill_root:
        raise OSError("tool result path escaped the session spill directory")
    return target


def spill_single(session: SessionContext, tool_use_id: str, content: str) -> str:
    """Atomically persist one tool result and return its safe absolute path."""
    if not isinstance(session, SessionContext):
        raise TypeError("session must be a SessionContext")
    if not isinstance(content, str):
        raise TypeError("content must be a string")

    target = _spill_path(session, tool_use_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = content.encode("utf-8")
    if target.exists():
        if not target.is_file() or target.read_bytes() != encoded:
            raise OSError("tool result id conflicts with existing spill content")
        return str(target)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".spill-", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return str(target)


def _head_preview(content: str) -> str:
    """Return at most the configured number of lines and UTF-8 bytes."""
    line_limited = "".join(content.splitlines(keepends=True)[:PREVIEW_HEAD_LINES])
    encoded = line_limited.encode("utf-8")
    if len(encoded) <= PREVIEW_HEAD_BYTES:
        return line_limited
    return encoded[:PREVIEW_HEAD_BYTES].decode("utf-8", errors="ignore")


def build_preview(original_bytes: int, head: str, spill_path: str) -> str:
    """Build the stable prompt replacement for an offloaded tool result."""
    if not isinstance(original_bytes, int) or isinstance(original_bytes, bool):
        raise TypeError("original_bytes must be an integer")
    if original_bytes < 0:
        raise ValueError("original_bytes must not be negative")
    if not isinstance(head, str):
        raise TypeError("head must be a string")
    if not isinstance(spill_path, str) or not spill_path.strip():
        raise ValueError("spill_path must be a non-empty string")
    return "\n".join(
        (
            f"[content offloaded] original size: {original_bytes} bytes",
            f"[saved to] {spill_path}",
            "[head preview]",
            head,
            "完整内容已保存到上述路径，如需查看请用文件读取工具读取该路径，"
            "不要凭头部预览猜测全文。",
        )
    )


def _replacement_indexes(message: Message) -> set[int]:
    """Choose the minimum size-ordered results required by both limits."""
    sizes = [len(result.content.encode("utf-8")) for result in message.tool_results]
    selected = {index for index, size in enumerate(sizes) if size > SINGLE_RESULT_LIMIT}
    aggregate = sum(size for size in sizes if size <= SINGLE_RESULT_LIMIT)
    if aggregate <= MESSAGE_AGGREGATE_LIMIT:
        return selected
    candidates = sorted(
        (index for index, size in enumerate(sizes) if size <= SINGLE_RESULT_LIMIT),
        key=lambda index: sizes[index],
        reverse=True,
    )
    for index in candidates:
        if aggregate <= MESSAGE_AGGREGATE_LIMIT:
            break
        selected.add(index)
        aggregate -= sizes[index]
    return selected


def offload_and_snip(
    messages: list[Message],
    state: ContentReplacementState,
    session: SessionContext,
) -> list[Message]:
    """Return messages with oversized tool results replaced by stable previews."""
    if not isinstance(state, ContentReplacementState):
        raise TypeError("state must be a ContentReplacementState")
    if not isinstance(session, SessionContext):
        raise TypeError("session must be a SessionContext")

    output: list[Message] = []
    for message in messages:
        if not isinstance(message, Message):
            raise TypeError("messages must contain only Message values")
        if message.role is not MessageRole.TOOL:
            output.append(message)
            continue

        selected = _replacement_indexes(message)
        updated_results = []
        for index, result in enumerate(message.tool_results):
            original = result.content

            def decide(
                selected_for_offload: bool = index in selected,
                tool_call_id: str = result.tool_call_id,
                original_content: str = original,
            ) -> tuple[str, str]:
                if not selected_for_offload:
                    return "kept", ""
                try:
                    spill_path = spill_single(session, tool_call_id, original_content)
                except OSError:
                    return "skip", ""
                preview = build_preview(
                    len(original_content.encode("utf-8")),
                    _head_preview(original_content),
                    spill_path,
                )
                return "replaced", preview

            updated_content = state.decide_once(result.tool_call_id, original, decide)
            updated_results.append(
                result if updated_content == original else replace(result, content=updated_content)
            )

        updated_tuple = tuple(updated_results)
        output.append(
            message
            if updated_tuple == message.tool_results
            else replace(message, tool_results=updated_tuple)
        )
    return output
