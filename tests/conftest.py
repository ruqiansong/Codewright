"""Shared pytest fixtures for Codewright tests."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

from codewright.llm import (
    ChatResult,
    LLMError,
    Message,
    MessageRole,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolDefinition,
)


@pytest.fixture(autouse=True)
def synchronous_to_thread_on_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid this Python build's default-executor shutdown defect in tests."""

    async def run_immediately(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", run_immediately)


@dataclass(frozen=True, slots=True)
class ScriptedReply:
    """One deterministic response used by the end-to-end Fake Provider."""

    chunks: tuple[str, ...] = ()
    error: LLMError | None = None
    delay_seconds: float = 0.0
    pause_after_delta: int | None = None


class ScriptedProvider:
    """Configurable offline Provider supporting success, error, delay, and cancellation."""

    def __init__(self, replies: Sequence[ScriptedReply]) -> None:
        if not replies:
            raise ValueError("replies must not be empty")
        self._replies = tuple(replies)
        self._reply_index = 0
        self.requests: list[tuple[Message, ...]] = []
        self.chat_requests: list[tuple[Message, ...]] = []
        self.stream_requests: list[tuple[Message, ...]] = []
        self.request_contexts: list[RequestContext | None] = []
        self.paused = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.closed = False

    @property
    def provider_name(self) -> str:
        return "scripted"

    @property
    def model_name(self) -> str:
        return "scripted-model"

    def _next_reply(self, messages: Sequence[Message]) -> ScriptedReply:
        request = tuple(messages)
        self.requests.append(request)
        reply = self._replies[min(self._reply_index, len(self._replies) - 1)]
        self._reply_index += 1
        return reply

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del parameters, tools
        self.request_contexts.append(request_context)
        request = tuple(messages)
        self.chat_requests.append(request)
        reply = self._next_reply(request)
        if reply.delay_seconds:
            await asyncio.sleep(reply.delay_seconds)
        if reply.error is not None:
            raise reply.error
        return ChatResult(
            message=Message(MessageRole.ASSISTANT, "".join(reply.chunks)),
            model=self.model_name,
        )

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters, tools
        self.request_contexts.append(request_context)
        request = tuple(messages)
        self.stream_requests.append(request)
        reply = self._next_reply(request)
        try:
            for index, chunk in enumerate(reply.chunks):
                if reply.delay_seconds:
                    await asyncio.sleep(reply.delay_seconds)
                yield StreamEvent.delta(chunk)
                if reply.pause_after_delta == index:
                    self.paused.set()
                    await self.release.wait()
            if reply.error is not None:
                yield StreamEvent.failed(reply.error)
            else:
                yield StreamEvent.completed()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def valid_config_path() -> Path:
    """Return the checked-in, non-production configuration fixture."""
    return Path(__file__).parent / "fixtures" / "config.valid.yaml"
