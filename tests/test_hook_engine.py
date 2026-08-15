"""Unit tests for Hook conditions, event parsing, and Engine dispatch."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from codewright.hook import (
    Action,
    ActionType,
    AtomCondition,
    CombineMode,
    Condition,
    Engine,
    Event,
    PromptAction,
    Rule,
    ShellAction,
    is_blocking,
    parse_event,
)
from codewright.hook.matcher import eval_condition, get_by_path
from codewright.permission.matcher import ExactMatcher, GlobMatcher


@dataclass
class Session:
    fired: set[str] = field(default_factory=set)

    def claim_hook_once(self, name: str) -> bool:
        if name in self.fired:
            return False
        self.fired.add(name)
        return True


def prompt_rule(name: str, text: str, *, once: bool = False) -> Rule:
    return Rule(
        name,
        Event.SESSION_START,
        Action(ActionType.PROMPT, prompt=PromptAction(text)),
        only_once=once,
    )


def shell_rule(
    name: str,
    event: Event,
    command: str,
    *,
    async_mode: bool = False,
    once: bool = False,
) -> Rule:
    return Rule(
        name,
        event,
        Action(ActionType.SHELL, shell=ShellAction(command)),
        only_once=once,
        asyncio_mode=async_mode,
        timeout_s=1,
    )


def test_events_parse_and_only_designated_events_block() -> None:
    assert len(Event) == 11
    assert parse_event("PreToolUse") is Event.PRE_TOOL_USE
    assert parse_event("unknown") is None
    assert is_blocking(Event.PRE_TOOL_USE)
    assert is_blocking(Event.USER_PROMPT_SUBMIT)
    assert not is_blocking(Event.STOP)


def test_nested_field_lookup_and_all_any_conditions() -> None:
    payload = {
        "tool_input": {"path": "src/main.py", "flags": ["a", "b"]},
        "is_error": False,
        "count": 2,
    }
    assert get_by_path(payload, "tool_input.path") == "src/main.py"
    assert get_by_path(payload, "tool_input.flags") == '["a","b"]'
    assert get_by_path(payload, "is_error") == "False"
    assert get_by_path(payload, "count") == "2"
    assert get_by_path(payload, "missing.path") == ""
    all_condition = Condition(
        CombineMode.ALL_OF,
        (
            AtomCondition("tool_input.path", GlobMatcher("**/*.py", False)),
            AtomCondition("is_error", ExactMatcher("False")),
        ),
    )
    any_condition = Condition(
        CombineMode.ANY_OF,
        (
            AtomCondition("missing", ExactMatcher("present")),
            AtomCondition("count", ExactMatcher("2")),
        ),
    )
    assert eval_condition(None, payload)
    assert eval_condition(all_condition, payload)
    assert eval_condition(any_condition, payload)


@pytest.mark.asyncio
async def test_dispatch_accumulates_prompts_in_declaration_order() -> None:
    engine = Engine(
        [prompt_rule("first", "one"), prompt_rule("second", "two")],
        ["project.yaml"],
    )
    try:
        result = await engine.dispatch(
            Event.SESSION_START,
            {"cwd": "."},
            Session(),
        )
    finally:
        await engine.aclose()

    assert result.injected_prompts == ["one", "two"]
    assert engine.sources == ["project.yaml"]
    assert [rule.name for rule in engine.rules] == ["first", "second"]
    engine.sources.append("mutated")
    engine.rules.clear()
    assert engine.sources == ["project.yaml"]
    assert [rule.name for rule in engine.rules] == ["first", "second"]


@pytest.mark.asyncio
async def test_blocking_dispatch_stops_after_first_block(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    engine = Engine(
        [
            shell_rule("blocker", Event.PRE_TOOL_USE, "echo denied >&2; exit 2"),
            shell_rule("later", Event.PRE_TOOL_USE, f"touch {marker.name}"),
        ],
        [],
    )
    try:
        result = await engine.dispatch(
            Event.PRE_TOOL_USE,
            {"cwd": str(tmp_path)},
            Session(),
        )
    finally:
        await engine.aclose()

    assert result.blocked
    assert result.reason == "denied"
    assert result.blocking_hook_name == "blocker"
    assert not marker.exists()


@pytest.mark.asyncio
async def test_exit_two_does_not_block_nonblocking_event(tmp_path: Path) -> None:
    engine = Engine([shell_rule("stop", Event.STOP, "exit 2")], [])
    try:
        result = await engine.dispatch(Event.STOP, {"cwd": str(tmp_path)}, Session())
    finally:
        await engine.aclose()
    assert not result.blocked


@pytest.mark.asyncio
async def test_only_once_claims_before_execution_and_resets_with_new_session(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    engine = Engine(
        [shell_rule("once", Event.STOP, "exit 1", once=True)],
        [],
    )
    first = Session()
    try:
        await engine.dispatch(Event.STOP, {"cwd": str(tmp_path)}, first)
        await engine.dispatch(Event.STOP, {"cwd": str(tmp_path)}, first)
        first_logs = capsys.readouterr().err
        await engine.dispatch(Event.STOP, {"cwd": str(tmp_path)}, Session())
        second_logs = capsys.readouterr().err
    finally:
        await engine.aclose()

    assert first_logs.count("[hook once]") == 1
    assert second_logs.count("[hook once]") == 1


@pytest.mark.asyncio
async def test_background_failure_is_logged_and_aclose_drains(tmp_path: Path, capsys) -> None:
    engine = Engine(
        [shell_rule("async-fail", Event.STOP, "exit 1", async_mode=True)],
        [],
    )
    result = await engine.dispatch(Event.STOP, {"cwd": str(tmp_path)}, Session())
    assert not result.blocked
    await engine.aclose()
    assert "[hook async-fail] Stop failed: exit 1" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_cancelled_aclose_cancels_background_and_propagates(tmp_path: Path) -> None:
    engine = Engine(
        [shell_rule("slow", Event.STOP, "sleep 5", async_mode=True)],
        [],
    )
    await engine.dispatch(Event.STOP, {"cwd": str(tmp_path)}, Session())
    closer = asyncio.create_task(engine.aclose())
    await asyncio.sleep(0.01)
    closer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closer
