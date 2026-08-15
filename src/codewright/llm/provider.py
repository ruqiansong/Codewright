"""Protocol implemented by every Codewright language model provider."""

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol, runtime_checkable

from codewright.llm.models import ChatResult, Message, RequestContext, StreamEvent, ToolDefinition

type RequestParameters = Mapping[str, object]


@runtime_checkable
class Provider(Protocol):
    """Vendor-neutral asynchronous language model provider contract."""

    @property
    def provider_name(self) -> str:
        """Return the configured provider name."""
        ...

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        ...

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        """Return one complete assistant response."""
        ...

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield text deltas followed by one terminal event."""
        ...
