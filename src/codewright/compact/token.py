"""Approximate context token accounting anchored to provider usage."""

from __future__ import annotations

import math
from collections.abc import Sequence

from codewright.compact.const import ESTIMATE_CHARS_PER_TOKEN
from codewright.llm import Message, TokenUsage


def usage_anchor(usage: TokenUsage) -> int:
    """Return the provider-normalized total without double-counting cache tokens."""
    if not isinstance(usage, TokenUsage):
        raise TypeError("usage must be a TokenUsage")
    return usage.total_tokens


def message_chars(messages: Sequence[Message]) -> int:
    """Return the UTF-8 byte footprint of protocol-neutral message content."""
    total = 0
    for message in messages:
        if not isinstance(message, Message):
            raise TypeError("messages must contain only Message values")
        total += len(message.content.encode("utf-8"))
        total += sum(len(call.arguments_json.encode("utf-8")) for call in message.tool_calls)
        total += sum(len(result.content.encode("utf-8")) for result in message.tool_results)
    return total


def estimate_tokens(
    anchor: int,
    all_messages: Sequence[Message],
    anchor_message_count: int,
) -> int:
    """Estimate tokens from a real usage anchor plus messages appended afterward."""
    if not isinstance(anchor, int) or isinstance(anchor, bool) or anchor < 0:
        raise ValueError("anchor must be a non-negative integer")
    if (
        not isinstance(anchor_message_count, int)
        or isinstance(anchor_message_count, bool)
        or anchor_message_count < 0
    ):
        raise ValueError("anchor_message_count must be a non-negative integer")

    start = min(anchor_message_count, len(all_messages))
    added_bytes = message_chars(all_messages[start:])
    return anchor + math.ceil(added_bytes / ESTIMATE_CHARS_PER_TOKEN)
