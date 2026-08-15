"""Completion values and the single consumer for Agent event streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codewright.llm import TokenUsage

if TYPE_CHECKING:
    from codewright.agent import Event


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Final text and accumulated usage for one completed Agent run."""

    text: str
    usage: TokenUsage


class MaxTurnsReached(RuntimeError):
    """Raised when a child Agent exhausts its configured iteration budget."""

    def __init__(self, last_text: str) -> None:
        if not isinstance(last_text, str):
            raise TypeError("last_text must be a string")
        self.last_text = last_text
        super().__init__("Subagent reached its maximum turn limit.")


async def consume_events(
    events: AsyncIterator[Event],
    event_sink: asyncio.Queue[Event | None] | None = None,
) -> CompletionResult:
    """Consume one Agent.run stream without duplicating orchestration logic."""
    text_parts: list[str] = []
    usage = TokenUsage(0, 0, 0)
    completed = False
    try:
        async for event in events:
            _offer_event(event_sink, event)
            if event.text:
                text_parts.append(event.text)
            if event.usage is not None:
                usage = add_usage(usage, event.usage)
            if event.error is not None:
                raise event.error
            if event.done:
                completed = True
                break
        if not completed:
            raise RuntimeError("Agent execution ended without completing.")
        return CompletionResult("".join(text_parts), usage)
    finally:
        _offer_event(event_sink, None)


def add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    """Add all vendor-neutral token accounting fields."""
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
        cache_write_tokens=left.cache_write_tokens + right.cache_write_tokens,
        cache_read_tokens=left.cache_read_tokens + right.cache_read_tokens,
    )


def _offer_event(
    event_sink: asyncio.Queue[Event | None] | None,
    event: Event | None,
) -> None:
    if event_sink is None:
        return
    try:
        event_sink.put_nowait(event)
    except asyncio.QueueFull:
        pass


__all__ = ["CompletionResult", "MaxTurnsReached", "add_usage", "consume_events"]
