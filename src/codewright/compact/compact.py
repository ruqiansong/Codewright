"""Public context-management orchestration."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from codewright.compact.const import AUTO_SAFETY_MARGIN, SUMMARY_RESERVE
from codewright.compact.layer1 import offload_and_snip
from codewright.compact.layer2 import auto_compact, force_compact
from codewright.compact.state import (
    CompactCircuitBreaker,
    ContentReplacementState,
    RecoveryState,
    SessionContext,
)
from codewright.compact.token import estimate_tokens
from codewright.conversation import Conversation
from codewright.llm import MessageRole, Provider, ToolDefinition

logger = logging.getLogger(__name__)


class TriggerKind(StrEnum):
    """Reason context management was requested."""

    AUTO = "auto"
    MANUAL = "manual"
    EMERGENCY = "emergency"


@dataclass(frozen=True, slots=True)
class ManageInput:
    """Dependencies and token state for one management pass."""

    conv: Conversation
    provider: Provider
    context_window: int
    tool_defs: Sequence[ToolDefinition]
    replacement: ContentReplacementState
    recovery: RecoveryState
    auto_tracking: CompactCircuitBreaker
    session: SessionContext
    usage_anchor: int
    anchor_msg_len: int
    estimated_token: int
    trigger: TriggerKind = TriggerKind.AUTO
    on_layer2_start: Callable[[int], Awaitable[None] | None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.conv, Conversation):
            raise TypeError("conv must be a Conversation")
        for field_name in ("context_window", "usage_anchor", "anchor_msg_len", "estimated_token"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.context_window == 0:
            raise ValueError("context_window must be positive")
        if not isinstance(self.trigger, TriggerKind):
            raise TypeError("trigger must be a TriggerKind")
        if not all(isinstance(definition, ToolDefinition) for definition in self.tool_defs):
            raise TypeError("tool_defs must contain only ToolDefinition values")


@dataclass(frozen=True, slots=True)
class ManageOutput:
    """Observable outcome of one context-management pass."""

    before_tokens: int
    after_tokens: int
    history_rewritten: bool
    layer2_started: bool


async def _run_layer1(in_: ManageInput) -> tuple[bool, bool, int]:
    old_messages = list(in_.conv.messages())
    if not any(message.role is MessageRole.TOOL for message in old_messages):
        current_tokens = estimate_tokens(
            in_.usage_anchor,
            old_messages,
            in_.anchor_msg_len,
        )
        return False, False, current_tokens
    rewritten = await asyncio.to_thread(
        offload_and_snip,
        old_messages,
        in_.replacement,
        in_.session,
    )
    changed = rewritten != old_messages
    if changed:
        in_.conv.replace_history(rewritten)
        anchor_end = min(in_.anchor_msg_len, len(old_messages), len(rewritten))
        anchor_rewritten = old_messages[:anchor_end] != rewritten[:anchor_end]
        if anchor_rewritten:
            return True, True, estimate_tokens(0, rewritten, 0)
        return (
            True,
            False,
            estimate_tokens(in_.usage_anchor, rewritten, in_.anchor_msg_len),
        )
    return False, False, estimate_tokens(in_.usage_anchor, old_messages, in_.anchor_msg_len)


async def manage_context(in_: ManageInput) -> ManageOutput:
    """Apply layer-one offloading and optional layer-two summarization."""
    before_tokens = in_.estimated_token

    if in_.trigger is TriggerKind.MANUAL:
        messages, _, after_tokens = await force_compact(in_)
        in_.conv.replace_history(messages)
        return ManageOutput(before_tokens, after_tokens, True, True)

    layer1_changed, anchor_rewritten, current_tokens = await _run_layer1(in_)
    layer2_input = replace(
        in_,
        usage_anchor=0 if anchor_rewritten else in_.usage_anchor,
        anchor_msg_len=0 if anchor_rewritten else in_.anchor_msg_len,
        estimated_token=current_tokens,
    )

    if in_.trigger is TriggerKind.EMERGENCY:
        messages, _, after_tokens = await force_compact(layer2_input)
        in_.conv.replace_history(messages)
        return ManageOutput(before_tokens, after_tokens, True, True)

    if in_.context_window <= SUMMARY_RESERVE + AUTO_SAFETY_MARGIN:
        logger.warning("Automatic compaction disabled for context window %s", in_.context_window)
        return ManageOutput(before_tokens, current_tokens, layer1_changed, False)

    threshold = in_.context_window - SUMMARY_RESERVE - AUTO_SAFETY_MARGIN
    if current_tokens < threshold or in_.auto_tracking.tripped():
        return ManageOutput(before_tokens, current_tokens, layer1_changed, False)

    if in_.on_layer2_start is not None:
        started = in_.on_layer2_start(current_tokens)
        if inspect.isawaitable(started):
            await started
    messages, _, after_tokens = await auto_compact(layer2_input)
    in_.conv.replace_history(messages)
    return ManageOutput(before_tokens, after_tokens, True, True)
