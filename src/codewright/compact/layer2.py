"""Pure message-selection primitives used by LLM-backed compaction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from codewright.compact.const import (
    ESTIMATE_CHARS_PER_TOKEN,
    MANUAL_SAFETY_MARGIN,
    PTL_DROP_PERCENTAGE,
    PTL_RETRY_LIMIT,
    RECENT_KEEP_MESSAGES,
    RECENT_KEEP_TOKENS,
    SUMMARY_RESERVE,
)
from codewright.compact.recovery import build_recovery_attachment
from codewright.compact.summary_prompt import build_summary_prompt, extract_summary
from codewright.compact.token import message_chars
from codewright.llm import Message, MessageRole, PromptTooLongError

if TYPE_CHECKING:
    from codewright.compact.compact import ManageInput

_SUMMARY_BRIDGE = "（已加载上下文摘要与恢复信息。请继续。）"


def _paired_assistant_index(messages: Sequence[Message], tool_index: int) -> int | None:
    """Find the nearest assistant call batch matching a tool-result message."""
    result_ids = {result.tool_call_id for result in messages[tool_index].tool_results}
    for index in range(tool_index - 1, -1, -1):
        candidate = messages[index]
        if candidate.role is MessageRole.USER:
            return None
        if candidate.role is MessageRole.ASSISTANT:
            call_ids = {call.id for call in candidate.tool_calls}
            if result_ids.issubset(call_ids):
                return index
    return None


def pick_recent_tail(messages: Sequence[Message]) -> list[Message]:
    """Select a recent suffix without separating tool calls from their results."""
    if not messages:
        return []
    history_start = 1 if messages[0].role is MessageRole.SYSTEM else 0
    if history_start == len(messages):
        return []

    start_index = len(messages)
    byte_count = 0
    message_count = 0
    for index in range(len(messages) - 1, history_start - 1, -1):
        message = messages[index]
        if not isinstance(message, Message):
            raise TypeError("messages must contain only Message values")
        byte_count += message_chars((message,))
        message_count += 1
        start_index = index
        estimated_tokens = math.ceil(byte_count / ESTIMATE_CHARS_PER_TOKEN)
        if estimated_tokens >= RECENT_KEEP_TOKENS and message_count >= RECENT_KEEP_MESSAGES:
            break

    if messages[start_index].role is MessageRole.TOOL:
        paired_index = _paired_assistant_index(messages, start_index)
        if paired_index is not None:
            start_index = paired_index
        else:
            while start_index < len(messages) and messages[start_index].role is MessageRole.TOOL:
                start_index += 1
    return list(messages[start_index:])


def _join_after_summary(summary_and_recovery: Message, recent: Sequence[Message]) -> list[Message]:
    """Join a user summary to recent history with a protocol-valid boundary."""
    if summary_and_recovery.role is not MessageRole.USER:
        raise ValueError("summary_and_recovery must have the user role")
    cleaned = list(recent)
    while cleaned and cleaned[0].role in {MessageRole.SYSTEM, MessageRole.TOOL}:
        cleaned.pop(0)
    if not cleaned:
        return [summary_and_recovery]
    if cleaned[0].role is MessageRole.USER:
        bridge = Message(MessageRole.ASSISTANT, _SUMMARY_BRIDGE)
        return [summary_and_recovery, bridge, *cleaned]
    return [summary_and_recovery, *cleaned]


def group_by_user_turn(messages: Sequence[Message]) -> list[list[Message]]:
    """Group history into complete user submissions and their following exchanges."""
    groups: list[list[Message]] = []
    leading: list[Message] = []
    for message in messages:
        if not isinstance(message, Message):
            raise TypeError("messages must contain only Message values")
        if message.role is MessageRole.SYSTEM:
            continue
        if message.role is MessageRole.USER:
            if not groups:
                groups.append([*leading, message])
                leading = []
            else:
                groups.append([message])
        elif groups:
            groups[-1].append(message)
        else:
            leading.append(message)
    if leading:
        groups.append(leading)
    return groups


async def summarize_once(in_: ManageInput, messages: Sequence[Message]) -> str:
    """Request one tool-free summary without changing main-request usage anchors."""
    text_buffer: list[str] = []
    async for event in in_.provider.stream_chat(build_summary_prompt(list(messages)), tools=()):
        if event.error is not None:
            raise event.error
        if event.text:
            text_buffer.append(event.text)
    return extract_summary("".join(text_buffer))


def _retry_messages(
    system: Message | None,
    groups: Sequence[Sequence[Message]],
    dropped_groups: int,
) -> list[Message]:
    messages: list[Message] = []
    if system is not None:
        messages.append(system)
    if dropped_groups:
        messages.append(
            Message(
                MessageRole.ASSISTANT,
                f"[compaction notice] {dropped_groups} oldest user turn group(s) omitted.",
            )
        )
    for group in groups:
        messages.extend(group)
    return messages


def _summary_request_tokens(messages: Sequence[Message]) -> int:
    prompt = build_summary_prompt(list(messages))
    return math.ceil(message_chars(prompt) / ESTIMATE_CHARS_PER_TOKEN)


async def ptl_retry(
    in_: ManageInput,
    messages: Sequence[Message],
    first_err: PromptTooLongError | None,
) -> str:
    """Retry an oversized summary after dropping complete oldest user-turn groups."""
    system = messages[0] if messages and messages[0].role is MessageRole.SYSTEM else None
    remaining = group_by_user_turn(messages)
    if not remaining:
        if first_err is not None:
            raise first_err
        raise PromptTooLongError()

    dropped_groups = 0
    last_error = first_err
    retry_count = 0
    preflight_limit = in_.context_window - SUMMARY_RESERVE - MANUAL_SAFETY_MARGIN

    if first_err is None:
        while remaining:
            candidate = _retry_messages(system, remaining, dropped_groups)
            if preflight_limit > 0 and _summary_request_tokens(candidate) <= preflight_limit:
                break
            remaining.pop(0)
            dropped_groups += 1
    else:
        remaining.pop(0)
        dropped_groups += 1

    while remaining:
        candidate = _retry_messages(system, remaining, dropped_groups)
        try:
            return await summarize_once(in_, candidate)
        except PromptTooLongError as error:
            last_error = error
            retry_count += 1

        if retry_count < PTL_RETRY_LIMIT:
            drop_count = 1
        else:
            drop_count = max(1, math.ceil(len(remaining) * PTL_DROP_PERCENTAGE))
        drop_count = min(drop_count, len(remaining))
        del remaining[:drop_count]
        dropped_groups += drop_count

    if last_error is not None:
        raise last_error
    raise PromptTooLongError()


async def run_summary(in_: ManageInput) -> list[Message]:
    """Build summarized history with recovery material and a recent raw tail."""
    old_messages = list(in_.conv.messages())
    if not old_messages or old_messages[0].role is not MessageRole.SYSTEM:
        raise ValueError("conversation must contain one leading system message")
    recovery_snapshot = in_.recovery.snapshot()
    preflight_limit = in_.context_window - SUMMARY_RESERVE - MANUAL_SAFETY_MARGIN

    if preflight_limit <= 0 or _summary_request_tokens(old_messages) > preflight_limit:
        summary_text = await ptl_retry(in_, old_messages, None)
    else:
        try:
            summary_text = await summarize_once(in_, old_messages)
        except PromptTooLongError as error:
            summary_text = await ptl_retry(in_, old_messages, error)

    recovery_text = build_recovery_attachment(recovery_snapshot, in_.tool_defs)
    summary_and_recovery = Message(
        MessageRole.USER,
        f"## 历史会话摘要\n{summary_text}\n\n{recovery_text}",
    )
    joined = _join_after_summary(summary_and_recovery, pick_recent_tail(old_messages))
    return [old_messages[0], *joined]


def _estimate_complete_history(messages: Sequence[Message]) -> int:
    return math.ceil(message_chars(messages) / ESTIMATE_CHARS_PER_TOKEN)


async def auto_compact(in_: ManageInput) -> tuple[list[Message], int, int]:
    """Run automatic summary compaction and maintain its circuit breaker."""
    try:
        messages = await run_summary(in_)
    except Exception:
        in_.auto_tracking.record_failure()
        raise
    in_.auto_tracking.record_success()
    return messages, in_.estimated_token, _estimate_complete_history(messages)


async def force_compact(in_: ManageInput) -> tuple[list[Message], int, int]:
    """Run forced summary compaction without consulting automatic tracking."""
    messages = await run_summary(in_)
    return messages, in_.estimated_token, _estimate_complete_history(messages)
