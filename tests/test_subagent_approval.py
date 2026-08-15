"""Tests for FIFO subagent approval routing and shutdown cleanup."""

import asyncio

import pytest

from codewright.agent import ApprovalRequest
from codewright.permission import Outcome
from codewright.task import SubagentApprovalBroker


def _request(name: str) -> ApprovalRequest:
    future = asyncio.get_running_loop().create_future()
    return ApprovalRequest(name, "write_file", "{}", "confirm", future)


@pytest.mark.asyncio
async def test_broker_routes_concurrent_requests_fifo_and_returns_outcomes() -> None:
    broker = SubagentApprovalBroker()
    first = _request("first")
    second = _request("second")
    first_wait = asyncio.create_task(broker.request("alpha", first))
    second_wait = asyncio.create_task(broker.request("beta", second))
    await asyncio.sleep(0)

    first_envelope = await broker.subscribe().get()
    second_envelope = await broker.subscribe().get()
    assert first_envelope is not None and first_envelope.source == "alpha"
    assert second_envelope is not None and second_envelope.source == "beta"
    first.respond.set_result(Outcome.ALLOW_ONCE)
    second.respond.set_result(Outcome.DENY_ONCE)

    assert await first_wait is Outcome.ALLOW_ONCE
    assert await second_wait is Outcome.DENY_ONCE
    await broker.aclose()


@pytest.mark.asyncio
async def test_broker_caller_cancellation_cancels_response_future() -> None:
    broker = SubagentApprovalBroker()
    request = _request("cancel")
    waiting = asyncio.create_task(broker.request("worker", request))
    await asyncio.sleep(0)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert request.respond.cancelled()
    await broker.aclose()


@pytest.mark.asyncio
async def test_broker_close_denies_pending_and_is_idempotent() -> None:
    broker = SubagentApprovalBroker()
    request = _request("pending")
    waiting = asyncio.create_task(broker.request("worker", request))
    await asyncio.sleep(0)

    await broker.aclose()
    await broker.aclose()

    assert await waiting is Outcome.DENY_ONCE
    assert await broker.subscribe().get() is not None
    assert await broker.subscribe().get() is None
    closed = _request("closed")
    assert await broker.request("worker", closed) is Outcome.DENY_ONCE
    assert closed.respond.result() is Outcome.DENY_ONCE
