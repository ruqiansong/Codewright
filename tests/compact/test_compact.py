"""Tests for top-level context management orchestration."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from codewright.compact import (
    CompactCircuitBreaker,
    ContentReplacementState,
    ManageInput,
    RecoveryState,
    SessionContext,
    TriggerKind,
    manage_context,
)
from codewright.conversation import Conversation
from codewright.llm import (
    ChatResult,
    Message,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolDefinition,
)


class CompactProvider:
    def __init__(self, scripts: Sequence[Sequence[StreamEvent]]) -> None:
        self.scripts = list(scripts)
        self.calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-summary"

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        for event in self.scripts.pop(0):
            yield event


def make_input(
    tmp_path: Path,
    provider: CompactProvider,
    *,
    estimated_token: int = 1_000,
    usage_anchor: int = 0,
) -> ManageInput:
    conversation = Conversation("system")
    conversation.add_user("hello")
    messages = conversation.messages()
    return ManageInput(
        conv=conversation,
        provider=provider,
        context_window=200_000,
        tool_defs=(),
        replacement=ContentReplacementState(),
        recovery=RecoveryState(),
        auto_tracking=CompactCircuitBreaker(),
        session=SessionContext("test", str(tmp_path / "spill")),
        usage_anchor=usage_anchor,
        anchor_msg_len=len(messages) if usage_anchor else 0,
        estimated_token=estimated_token,
    )


def summary_script(text: str) -> list[StreamEvent]:
    return [StreamEvent.delta(f"<summary>{text}</summary>"), StreamEvent.completed()]


@pytest.fixture(autouse=True)
def synchronous_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid this CI Python build's default-executor shutdown defect."""

    async def run_immediately(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("codewright.compact.compact.asyncio.to_thread", run_immediately)


async def test_manage_context_auto_skips_layer2_below_threshold(tmp_path: Path) -> None:
    provider = CompactProvider([])
    in_ = make_input(tmp_path, provider)

    output = await manage_context(in_)

    assert output.layer2_started is False
    assert output.history_rewritten is False
    assert provider.calls == 0


async def test_manage_context_auto_triggers_at_threshold(tmp_path: Path) -> None:
    provider = CompactProvider([summary_script("automatic")])
    in_ = make_input(tmp_path, provider, estimated_token=170_000, usage_anchor=170_000)

    output = await manage_context(in_)

    assert output.layer2_started is True
    assert output.history_rewritten is True
    assert "automatic" in in_.conv.messages()[1].content


async def test_manage_context_manual_bypasses_threshold_and_breaker(tmp_path: Path) -> None:
    provider = CompactProvider([summary_script("manual")])
    in_ = make_input(tmp_path, provider)
    for _ in range(3):
        in_.auto_tracking.record_failure()

    output = await manage_context(replace(in_, trigger=TriggerKind.MANUAL))

    assert output.layer2_started is True
    assert "manual" in in_.conv.messages()[1].content


async def test_manage_context_emergency_forces_layer2(tmp_path: Path) -> None:
    provider = CompactProvider([summary_script("emergency")])
    in_ = make_input(tmp_path, provider)

    output = await manage_context(replace(in_, trigger=TriggerKind.EMERGENCY))

    assert output.layer2_started is True
    assert output.history_rewritten is True
    assert "emergency" in in_.conv.messages()[1].content
