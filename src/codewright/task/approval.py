"""FIFO routing for approval requests produced by background subagents."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from codewright.agent import ApprovalRequest
from codewright.permission import Outcome


@dataclass(frozen=True, slots=True)
class SubagentApproval:
    """One approval request paired with its display source."""

    source: str
    request: ApprovalRequest

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, str)
            or not self.source
            or self.source != self.source.strip()
        ):
            raise ValueError("source must be a non-empty trimmed string")
        if not isinstance(self.request, ApprovalRequest):
            raise TypeError("request must be an ApprovalRequest")


class SubagentApprovalBroker:
    """Maintain FIFO approval display order and settle requests on shutdown."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[SubagentApproval | None] = asyncio.Queue()
        self._pending: set[asyncio.Future[Outcome]] = set()
        self._closed = False
        self._lock = asyncio.Lock()

    def subscribe(self) -> asyncio.Queue[SubagentApproval | None]:
        """Return the stable queue consumed by the presentation layer."""
        return self._queue

    async def request(self, source: str, request: ApprovalRequest) -> Outcome:
        """Enqueue one request and wait for its existing response Future."""
        envelope = SubagentApproval(source, request)
        async with self._lock:
            if self._closed:
                if not request.respond.done():
                    request.respond.set_result(Outcome.DENY_ONCE)
                return Outcome.DENY_ONCE
            if request.respond.done():
                if request.respond.cancelled():
                    return Outcome.DENY_ONCE
                return request.respond.result()
            self._pending.add(request.respond)
            self._queue.put_nowait(envelope)
        try:
            return await request.respond
        except asyncio.CancelledError:
            if not request.respond.done():
                request.respond.cancel()
            raise
        finally:
            async with self._lock:
                self._pending.discard(request.respond)

    async def aclose(self) -> None:
        """Idempotently deny pending approvals and stop the queue consumer."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            pending = tuple(self._pending)
            self._pending.clear()
            for response in pending:
                if not response.done():
                    response.set_result(Outcome.DENY_ONCE)
            self._queue.put_nowait(None)


__all__ = ["SubagentApproval", "SubagentApprovalBroker"]
