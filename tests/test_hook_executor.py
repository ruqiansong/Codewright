"""Unit tests for lifecycle Hook action execution."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from codewright.hook import (
    Action,
    ActionType,
    Event,
    Executor,
    HttpAction,
    PromptAction,
    Rule,
    ShellAction,
    SubagentAction,
)


def shell_rule(command: str, *, timeout: float = 2.0) -> Rule:
    return Rule(
        "shell-test",
        Event.PRE_TOOL_USE,
        Action(ActionType.SHELL, shell=ShellAction(command)),
        timeout_s=timeout,
    )


def http_rule(action: HttpAction, *, event: Event = Event.PRE_TOOL_USE) -> Rule:
    return Rule("http-test", event, Action(ActionType.HTTP, http=action))


@pytest.mark.asyncio
async def test_shell_exit_two_blocks_with_stderr_and_stable_stdin(tmp_path: Path) -> None:
    output = tmp_path / "payload.json"
    command = f"cat > {output.name}; echo blocked >&2; exit 2"
    executor = Executor()
    try:
        result = await executor.run(
            shell_rule(command),
            {"z": 1, "cwd": str(tmp_path), "a": 2},
            blocking=True,
        )
    finally:
        await executor.aclose()

    assert result.blocked
    assert result.reason == "blocked"
    assert output.read_text(encoding="utf-8") == '{"a":2,"cwd":"' + str(tmp_path) + '","z":1}'


@pytest.mark.asyncio
async def test_shell_success_forwards_stderr_discards_stdout_and_uses_cwd(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    executor = Executor()
    try:
        result = await executor.run(
            shell_rule("pwd; echo diagnostic >&2"),
            {"cwd": str(tmp_path)},
            blocking=False,
        )
    finally:
        await executor.aclose()

    captured = capsys.readouterr()
    assert result.err is None
    assert captured.out == ""
    assert captured.err == "diagnostic\n"


@pytest.mark.asyncio
async def test_shell_failure_timeout_and_invalid_cwd_are_contained(tmp_path: Path) -> None:
    executor = Executor()
    try:
        failed = await executor.run(
            shell_rule("echo bad >&2; exit 1"),
            {"cwd": str(tmp_path)},
            blocking=False,
        )
        timed_out = await executor.run(
            shell_rule("sleep 2", timeout=0.01),
            {"cwd": str(tmp_path)},
            blocking=False,
        )
        invalid = await executor.run(shell_rule("true"), {}, blocking=False)
    finally:
        await executor.aclose()

    assert isinstance(failed.err, RuntimeError)
    assert "exit 1: bad" in str(failed.err)
    assert isinstance(timed_out.err, TimeoutError)
    assert isinstance(invalid.err, ValueError)


@pytest.mark.asyncio
async def test_prompt_and_subagent_stub(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    executor = Executor()
    prompt = Rule(
        "prompt-test",
        Event.SESSION_START,
        Action(ActionType.PROMPT, prompt=PromptAction("use zh-CN")),
    )
    subagent = Rule(
        "subagent-test",
        Event.SESSION_START,
        Action(
            ActionType.SUBAGENT,
            subagent=SubagentAction("reviewer", "review this"),
        ),
    )
    try:
        prompt_result = await executor.run(prompt, {"cwd": str(tmp_path)}, blocking=False)
        subagent_result = await executor.run(subagent, {"cwd": str(tmp_path)}, blocking=False)
    finally:
        await executor.aclose()

    assert prompt_result.prompt == "use zh-CN"
    assert subagent_result.err is None
    assert "skipped: reviewer" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_http_block_template_success_and_failure_open() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/block":
            return httpx.Response(200, json={"decision": "block", "reason": "policy"})
        if path == "/bad-block":
            return httpx.Response(200, json={"decision": "block"})
        if path == "/invalid":
            return httpx.Response(200, text="not-json")
        if path == "/error":
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, text="accepted")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    executor = Executor(client)
    payload = {"event": "Stop", "cwd": "/workspace"}
    try:
        blocked = await executor.run(
            http_rule(HttpAction("https://hook.test/block")),
            payload,
            blocking=True,
        )
        templated = await executor.run(
            http_rule(
                HttpAction("https://hook.test/notify", body='{{"kind":"{event}"}}'),
                event=Event.STOP,
            ),
            payload,
            blocking=False,
        )
        template_failed = await executor.run(
            http_rule(
                HttpAction("https://hook.test/notify", body="{missing}"),
                event=Event.STOP,
            ),
            payload,
            blocking=False,
        )
        bad_block = await executor.run(
            http_rule(HttpAction("https://hook.test/bad-block")),
            payload,
            blocking=True,
        )
        invalid = await executor.run(
            http_rule(HttpAction("https://hook.test/invalid")),
            payload,
            blocking=True,
        )
        failed = await executor.run(
            http_rule(HttpAction("https://hook.test/error")),
            payload,
            blocking=False,
        )
        non_json = await executor.run(
            http_rule(HttpAction("https://hook.test/notify"), event=Event.STOP),
            payload,
            blocking=False,
        )
    finally:
        await executor.aclose()

    assert blocked.blocked and blocked.reason == "policy"
    assert templated.err is None
    assert json.loads(requests[1].content) == {"kind": "Stop"}
    assert isinstance(template_failed.err, KeyError)
    assert isinstance(bad_block.err, ValueError)
    assert isinstance(invalid.err, ValueError)
    assert isinstance(failed.err, RuntimeError)
    assert non_json.err is None
    assert requests[-1].content == b'{"cwd":"/workspace","event":"Stop"}'
    assert not client.is_closed
    await client.aclose()


@pytest.mark.asyncio
async def test_executor_propagates_cancellation(tmp_path: Path) -> None:
    executor = Executor()
    task = asyncio.create_task(
        executor.run(
            shell_rule("sleep 5", timeout=10),
            {"cwd": str(tmp_path)},
            blocking=False,
        )
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await executor.aclose()


@pytest.mark.asyncio
async def test_executor_closes_owned_http_client() -> None:
    executor = Executor()
    owned_client = executor._http_client
    await executor.aclose()
    assert owned_client.is_closed
