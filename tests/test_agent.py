"""Offline behavior tests for the bounded Codewright ReAct loop."""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from codewright.agent import (
    MAX_ITERATIONS,
    MAX_SUMMARY_LINES,
    MAX_UNKNOWN_RUN,
    NOTICE_CANCELLED,
    NOTICE_MAX_ITER,
    NOTICE_STREAM_ERROR,
    NOTICE_UNKNOWN_TOOLS,
    PLAN_REMINDER_INTERVAL,
    Agent,
    ApprovalRequest,
    CompactPhase,
    Event,
    Mode,
    Phase,
)
from codewright.agent.runtime import SessionRuntime
from codewright.compact import (
    CompactCircuitBreaker,
    ContentReplacementState,
    RecoveryState,
    SessionContext,
    TriggerKind,
    manage_context,
)
from codewright.conversation import Conversation
from codewright.hook import Action as HookAction
from codewright.hook import ActionType as HookActionType
from codewright.hook import Engine as HookEngine
from codewright.hook import Event as HookEvent
from codewright.hook import ExecutionResult as HookExecutionResult
from codewright.hook import PromptAction as HookPromptAction
from codewright.hook import Rule as HookRule
from codewright.hook.executor import Executor as HookExecutor
from codewright.llm import (
    ChatResult,
    LLMServiceError,
    Message,
    MessageRole,
    PromptTooLongError,
    RequestContext,
    RequestParameters,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from codewright.memory import Manager
from codewright.permission import Decision, Engine, Outcome
from codewright.permission.matcher import ExactMatcher
from codewright.permission.rule import Rule, RuleSet
from codewright.prompt import PLAN_MODE_REMINDER, plan_reminder, render_skill_catalog
from codewright.skills import SkillLoader
from codewright.tool import LoadSkillTool, Registry, Result


class FakeProvider:
    """Return scripted streams and record every protocol-neutral request."""

    provider_name = "fake"
    model_name = "fake-model"

    def __init__(self, replies: Sequence[Sequence[StreamEvent] | Exception]) -> None:
        self._replies = tuple(replies)
        self.requests: list[tuple[Message, ...]] = []
        self.tool_definitions: list[tuple[ToolDefinition, ...]] = []
        self.request_contexts: list[RequestContext | None] = []

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del messages, parameters, tools, request_context
        raise AssertionError("This fake is configured for stream_chat")

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters
        index = len(self.requests)
        self.requests.append(tuple(messages))
        self.tool_definitions.append(tuple(tools))
        self.request_contexts.append(request_context)
        reply = self._replies[index]
        if isinstance(reply, Exception):
            raise reply
        for event in reply:
            yield event


class RepeatingToolProvider(FakeProvider):
    """Request one registered tool forever with unique call IDs."""

    def __init__(self) -> None:
        super().__init__(())

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters
        index = len(self.requests)
        self.requests.append(tuple(messages))
        self.tool_definitions.append(tuple(tools))
        self.request_contexts.append(request_context)
        yield StreamEvent.tool_calls_ready(
            (ToolCall(f"call-{index}", "read_file", '{"path":"README.md"}'),)
        )
        yield StreamEvent.completed()


class NonStreamingProvider(FakeProvider):
    def __init__(self, result: ChatResult) -> None:
        super().__init__(())
        self._result = result

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del parameters
        self.requests.append(tuple(messages))
        self.tool_definitions.append(tuple(tools))
        self.request_contexts.append(request_context)
        return self._result


class BlockingProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__(())
        self.started = asyncio.Event()

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters
        self.requests.append(tuple(messages))
        self.tool_definitions.append(tuple(tools))
        self.request_contexts.append(request_context)
        self.started.set()
        await asyncio.Event().wait()
        yield StreamEvent.completed()


class HookProbeExecutor(HookExecutor):
    """Record Hook execution and return deterministic outcomes without I/O."""

    def __init__(self, outcomes: Mapping[str, HookExecutionResult] | None = None) -> None:
        self.outcomes = dict(outcomes or {})
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def run(self, rule, payload, *, blocking):
        del blocking
        self.calls.append((rule.name, dict(payload)))
        return self.outcomes.get(rule.name, HookExecutionResult())

    async def aclose(self) -> None:
        return None


def hook_rule(name: str, event: HookEvent, text: str = "") -> HookRule:
    return HookRule(
        name,
        event,
        HookAction(HookActionType.PROMPT, prompt=HookPromptAction(text)),
    )


@dataclass(slots=True)
class Probe:
    active: int = 0
    peak: int = 0
    timeline: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FakeTool:
    name: str
    result: Result
    read_only: bool = True
    delay: float = 0.0
    probe: Probe | None = None
    description: str = "A deterministic test tool."
    parameters: Mapping[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    calls: list[str] = field(default_factory=list)
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def execute(self, arguments_json: str) -> Result:
        self.calls.append(arguments_json)
        self.started.set()
        if self.probe is not None:
            self.probe.active += 1
            self.probe.peak = max(self.probe.peak, self.probe.active)
            self.probe.timeline.append(f"start:{self.name}")
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return self.result
        finally:
            if self.probe is not None:
                self.probe.timeline.append(f"end:{self.name}")
                self.probe.active -= 1


def conversation() -> Conversation:
    value = Conversation("You are Codewright.")
    value.add_user("Complete this task")
    return value


def registry_with(*tools: FakeTool) -> Registry:
    registry = Registry()
    for tool in tools:
        registry.register(tool)
    return registry


def build_agent(
    provider: FakeProvider,
    registry: Registry,
    *,
    version: str = "dev",
) -> Agent:
    root = Path.cwd().resolve()
    engine = Engine(
        root=root,
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=root / ".codewright" / "settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )
    return Agent(provider, registry, engine, version=version)


def runtime_for(tmp_path: Path, *, context_window: int = 200_000) -> SessionRuntime:
    return SessionRuntime(
        replacement=ContentReplacementState(),
        recovery=RecoveryState(),
        auto_tracking=CompactCircuitBreaker(),
        session=SessionContext("test-session", str(tmp_path / "spill")),
        context_window=context_window,
    )


def test_runtime_reset_for_new_session_replaces_session_scoped_state(tmp_path: Path) -> None:
    runtime = runtime_for(tmp_path, context_window=123_456)
    old_replacement = runtime.replacement
    old_recovery = runtime.recovery
    old_tracking = runtime.auto_tracking
    old_active = runtime.active_skills
    runtime.active_skills.activate("review", "body", tmp_path.resolve())
    runtime.update_anchor(300, 7)
    runtime.increment_turn_count()
    new_session = SessionContext("new-session", str(tmp_path / "new-spill"))

    runtime.reset_for_new_session(new_session)

    assert runtime.session is new_session
    assert runtime.replacement is not old_replacement
    assert runtime.recovery is not old_recovery
    assert runtime.auto_tracking is not old_tracking
    assert runtime.active_skills is not old_active
    assert runtime.active_skills.names() == ()
    assert runtime.anchor_snapshot() == (0, 0, 123_456)
    assert runtime.turn_count == 0


def test_runtime_hook_state_is_atomic_consumed_and_reset(tmp_path: Path) -> None:
    runtime = runtime_for(tmp_path)
    runtime.append_reminders(["one", "", "two"])
    assert runtime.take_reminders() == ["one", "two"]
    assert runtime.take_reminders() == []
    assert runtime.claim_hook_once("once")
    assert not runtime.claim_hook_once("once")
    assert runtime.claim_session_end()
    assert not runtime.claim_session_end()

    runtime.append_reminders(["old"])
    runtime.reset_for_new_session(SessionContext("next", str(tmp_path / "next")))

    assert runtime.take_reminders() == []
    assert runtime.claim_hook_once("once")
    assert runtime.claim_session_end()


def test_agent_active_skill_methods_preserve_session_order(tmp_path: Path) -> None:
    agent = build_runtime_agent(FakeProvider([text_reply("done")]), Registry(), tmp_path)
    root = tmp_path.resolve()

    agent.activate_skill("first", "old", root)
    agent.activate_skill("second", "body", root)
    agent.activate_skill("first", "new", root)

    assert [entry.name for entry in agent.list_active_skills()] == ["first", "second"]
    assert [entry.body for entry in agent.list_active_skills()] == ["new", "body"]
    agent.clear_active_skills()
    assert agent.list_active_skills() == ()


def test_runtime_reset_for_new_session_rejects_invalid_context(tmp_path: Path) -> None:
    runtime = runtime_for(tmp_path)
    with pytest.raises(TypeError, match="SessionContext"):
        runtime.reset_for_new_session(object())  # type: ignore[arg-type]


def build_runtime_agent(
    provider: FakeProvider,
    registry: Registry,
    tmp_path: Path,
    *,
    runtime: SessionRuntime | None = None,
) -> Agent:
    return Agent(
        provider,
        registry,
        engine_for(tmp_path),
        runtime=runtime or runtime_for(tmp_path),
    )


def engine_for(root: Path) -> Engine:
    return Engine(
        root=root.resolve(),
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=root / ".codewright" / "settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


async def collect_with_outcome(
    agent: Agent,
    value: Conversation,
    outcome: Outcome,
    *,
    cancel_event: asyncio.Event | None = None,
) -> list[Event]:
    events: list[Event] = []
    async for event in agent.run(value, cancel_event=cancel_event):
        events.append(event)
        if event.approval is not None:
            event.approval.respond.set_result(outcome)
    return events


async def collect(
    agent: Agent,
    value: Conversation,
    *,
    stream: bool = True,
    mode: Mode = Mode.DEFAULT,
    cancel_event: asyncio.Event | None = None,
) -> list[Event]:
    return [
        event
        async for event in agent.run(
            value,
            stream=stream,
            mode=mode,
            cancel_event=cancel_event,
        )
    ]


def tool_reply(*calls: ToolCall) -> tuple[StreamEvent, ...]:
    return (StreamEvent.tool_calls_ready(tuple(calls)), StreamEvent.completed())


def text_reply(text: str) -> tuple[StreamEvent, ...]:
    return (StreamEvent.delta(text), StreamEvent.completed())


@pytest.fixture(autouse=True)
def synchronous_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid this CI Python build's default-executor shutdown defect."""

    async def run_immediately(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("codewright.compact.compact.asyncio.to_thread", run_immediately)
    monkeypatch.setattr("codewright.agent.asyncio.to_thread", run_immediately)


@pytest.mark.usefixtures("synchronous_to_thread")
async def test_agent_emergency_compacts_and_retries_once(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            (StreamEvent.failed(PromptTooLongError()),),
            text_reply("<summary>restored context</summary>"),
            text_reply("Recovered answer."),
        ]
    )
    value = conversation()
    probe = HookProbeExecutor()
    hooks = HookEngine(
        [
            hook_rule("pre-compact", HookEvent.PRE_COMPACT),
            hook_rule("post-compact", HookEvent.POST_COMPACT),
        ],
        [],
        executor=probe,
    )
    agent = build_runtime_agent(provider, Registry(), tmp_path)
    agent._hook_engine = hooks

    events = await collect(agent, value)
    await hooks.aclose()

    phases = [event.compact.phase for event in events if event.compact is not None]
    assert phases == [CompactPhase.BEFORE_EMERGENCY, CompactPhase.AFTER_EMERGENCY]
    assert value.messages()[-1].content == "Recovered answer."
    assert len(provider.requests) == 3
    assert [payload["trigger"] for _, payload in probe.calls] == [
        "emergency",
        "emergency",
    ]


@pytest.mark.usefixtures("synchronous_to_thread")
async def test_agent_emits_auto_compact_events_only_when_layer2_runs(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            text_reply("<summary>automatic summary</summary>"),
            text_reply("Final answer."),
        ]
    )
    runtime = runtime_for(tmp_path)
    runtime.update_anchor(170_000, 2)
    probe = HookProbeExecutor()
    hooks = HookEngine(
        [
            hook_rule("pre-compact", HookEvent.PRE_COMPACT),
            hook_rule("post-compact", HookEvent.POST_COMPACT),
        ],
        [],
        executor=probe,
    )

    agent = build_runtime_agent(provider, Registry(), tmp_path, runtime=runtime)
    agent._hook_engine = hooks
    events = await collect(agent, conversation())
    await hooks.aclose()

    compact_events = [event.compact for event in events if event.compact is not None]
    assert [event.phase for event in compact_events] == [
        CompactPhase.BEFORE_AUTO,
        CompactPhase.AFTER_AUTO,
    ]
    assert all(event.error_message == "" for event in compact_events)
    assert provider.requests[-1][-1].content == "Complete this task"
    assert [payload["trigger"] for _, payload in probe.calls] == ["auto", "auto"]


@pytest.mark.usefixtures("synchronous_to_thread")
async def test_agent_emergency_does_not_retry_second_ptl(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            (StreamEvent.failed(PromptTooLongError()),),
            text_reply("<summary>restored context</summary>"),
            (StreamEvent.failed(PromptTooLongError()),),
        ]
    )

    events = await collect(
        build_runtime_agent(provider, Registry(), tmp_path),
        conversation(),
    )

    assert isinstance(next(event.error for event in events if event.error), PromptTooLongError)
    assert len(provider.requests) == 3


@pytest.mark.usefixtures("synchronous_to_thread")
async def test_agent_emergency_stops_when_compacted_history_is_still_too_large(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FakeProvider(
        [
            (StreamEvent.failed(PromptTooLongError()),),
            text_reply("<summary>still large</summary>"),
        ]
    )
    agent = build_runtime_agent(provider, Registry(), tmp_path)
    real_manage_context = manage_context

    async def report_oversized_history(in_):
        output = await real_manage_context(in_)
        if in_.trigger is TriggerKind.EMERGENCY:
            return replace(output, after_tokens=in_.context_window)
        return output

    monkeypatch.setattr("codewright.agent.manage_context", report_oversized_history)

    events = await collect(agent, conversation())

    assert isinstance(next(event.error for event in events if event.error), PromptTooLongError)
    assert len(provider.requests) == 2


@pytest.mark.usefixtures("synchronous_to_thread")
async def test_agent_updates_usage_anchor_and_tracks_read_file(tmp_path: Path) -> None:
    target = tmp_path / "tracked.txt"
    target.write_text("clean file content", encoding="utf-8")
    usage = TokenUsage(input_tokens=8, output_tokens=2, total_tokens=10)
    provider = FakeProvider(
        [
            tool_reply(ToolCall("read-1", "read_file", f'{{"path":"{target}"}}')),
            (StreamEvent.delta("done"), StreamEvent.usage_report(usage), StreamEvent.completed()),
        ]
    )
    tool = FakeTool("read_file", Result("     1\tclean file content"))
    runtime = runtime_for(tmp_path)
    agent = build_runtime_agent(provider, registry_with(tool), tmp_path, runtime=runtime)

    await collect(agent, conversation())

    assert runtime.anchor_snapshot()[:2] == (10, 5)
    snapshot = runtime.recovery.snapshot()
    assert [(record.path, record.content) for record in snapshot] == [
        (str(target.resolve()), "clean file content")
    ]


@pytest.mark.usefixtures("synchronous_to_thread")
async def test_run_force_compact_bypasses_auto_breaker(tmp_path: Path) -> None:
    provider = FakeProvider([text_reply("<summary>manual summary</summary>")])
    runtime = runtime_for(tmp_path)
    for _ in range(3):
        runtime.auto_tracking.record_failure()
    agent = build_runtime_agent(provider, Registry(), tmp_path, runtime=runtime)
    probe = HookProbeExecutor()
    hooks = HookEngine(
        [
            hook_rule("pre-compact", HookEvent.PRE_COMPACT),
            hook_rule("post-compact", HookEvent.POST_COMPACT),
        ],
        [],
        executor=probe,
    )
    agent._hook_engine = hooks
    value = conversation()

    before, after = await agent.run_force_compact(value, ())
    await hooks.aclose()

    assert before > 0
    assert after > 0
    assert "manual summary" in value.messages()[1].content
    assert runtime.anchor_snapshot()[:2] == (0, 0)
    assert [payload["trigger"] for _, payload in probe.calls] == ["manual", "manual"]


@pytest.mark.asyncio
async def test_agent_runs_multiple_iterations_until_natural_completion() -> None:
    first = FakeTool("read_file", Result("contents"))
    second = FakeTool("write_file", Result("written"), read_only=False)
    provider = FakeProvider(
        [
            tool_reply(ToolCall("read-1", "read_file", '{"path":"a"}')),
            tool_reply(ToolCall("write-1", "write_file", '{"path":"b","content":"new"}')),
            text_reply("Task complete."),
        ]
    )
    value = conversation()

    events = await collect(
        build_agent(provider, registry_with(first, second)),
        value,
        mode=Mode.ACCEPT_EDITS,
    )

    assert [event.iteration for event in events if event.iteration] == [1, 2, 3]
    assert [
        event.tool.name for event in events if event.tool and event.tool.phase is Phase.START
    ] == [
        "read_file",
        "write_file",
    ]
    assert len(provider.requests) == 3
    assert value.messages()[-1] == Message(MessageRole.ASSISTANT, "Task complete.")
    assert events[-1] == Event.completed()


@pytest.mark.asyncio
@pytest.mark.usefixtures("synchronous_to_thread")
async def test_agent_hook_blocks_tool_and_post_hook_sees_error(tmp_path: Path) -> None:
    tool = FakeTool("read_file", Result("should not run"))
    provider = FakeProvider(
        [
            tool_reply(ToolCall("read-1", "read_file", '{"path":"README.md"}')),
            text_reply("done"),
        ]
    )
    probe = HookProbeExecutor({"block-read": HookExecutionResult(True, "policy denied")})
    hooks = HookEngine(
        [
            hook_rule("block-read", HookEvent.PRE_TOOL_USE),
            hook_rule("after-read", HookEvent.POST_TOOL_USE),
        ],
        [],
        executor=probe,
    )
    agent = Agent(
        provider,
        registry_with(tool),
        engine_for(tmp_path),
        runtime=runtime_for(tmp_path),
        hook_engine=hooks,
    )
    value = conversation()

    events = await collect(agent, value)
    await hooks.aclose()

    assert tool.calls == []
    result = value.messages()[3].tool_results[0]
    assert result.content == "[hook block-read] policy denied"
    assert result.is_error and result.error_code == "hook_blocked"
    tool_phases = [event.tool.phase for event in events if event.tool is not None]
    assert tool_phases == [Phase.START, Phase.END]
    assert [name for name, _ in probe.calls] == ["block-read", "after-read"]
    post_payload = probe.calls[1][1]
    assert post_payload["tool_name"] == "read_file"
    assert post_payload["tool_input"] == {"path": "README.md"}
    assert post_payload["is_error"] is True
    assert post_payload["tool_result"] == "[hook block-read] policy denied"


@pytest.mark.asyncio
@pytest.mark.usefixtures("synchronous_to_thread")
async def test_pre_user_prompt_reminder_combines_with_plan_and_is_consumed(
    tmp_path: Path,
) -> None:
    provider = FakeProvider([text_reply("done")])
    hooks = HookEngine(
        [hook_rule("remind", HookEvent.PRE_USER_MESSAGE, "HOOK REMINDER")],
        [],
    )
    agent = Agent(
        provider,
        Registry(),
        engine_for(tmp_path),
        runtime=runtime_for(tmp_path),
        hook_engine=hooks,
    )

    await collect(agent, conversation(), mode=Mode.PLAN)
    await hooks.aclose()

    context = provider.request_contexts[0]
    assert context is not None
    assert PLAN_MODE_REMINDER in context.reminder
    assert "HOOK REMINDER" in context.reminder
    assert agent.runtime.take_reminders() == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("synchronous_to_thread")
async def test_stop_hook_runs_before_done_and_not_on_stream_error(tmp_path: Path) -> None:
    timeline: list[str] = []

    class TimelineExecutor(HookProbeExecutor):
        async def run(self, rule, payload, *, blocking):
            timeline.append(rule.name)
            return await super().run(rule, payload, blocking=blocking)

    success_probe = TimelineExecutor()
    success_hooks = HookEngine([hook_rule("stop", HookEvent.STOP)], [], executor=success_probe)
    success_agent = Agent(
        FakeProvider([text_reply("done")]),
        Registry(),
        engine_for(tmp_path),
        runtime=runtime_for(tmp_path),
        hook_engine=success_hooks,
    )
    events: list[Event] = []
    async for event in success_agent.run(conversation()):
        timeline.append("done" if event.done else "agent-event")
        events.append(event)
    await success_hooks.aclose()
    assert events[-1].done
    assert timeline.index("stop") < timeline.index("done")

    error_probe = HookProbeExecutor()
    error_hooks = HookEngine(
        [
            hook_rule("stop", HookEvent.STOP),
            hook_rule("notify", HookEvent.NOTIFICATION),
        ],
        [],
        executor=error_probe,
    )
    error_agent = Agent(
        FakeProvider([(StreamEvent.failed(LLMServiceError()),)]),
        Registry(),
        engine_for(tmp_path),
        runtime=runtime_for(tmp_path),
        hook_engine=error_hooks,
    )
    await collect(error_agent, conversation())
    await error_hooks.aclose()
    assert [name for name, _ in error_probe.calls] == ["notify"]
    assert error_probe.calls[0][1]["kind"] == "stream_error"


@pytest.mark.asyncio
@pytest.mark.usefixtures("synchronous_to_thread")
async def test_approval_notification_runs_before_approval_event(tmp_path: Path) -> None:
    write = FakeTool("write_file", Result("written"), read_only=False)
    provider = FakeProvider(
        [
            tool_reply(
                ToolCall(
                    "write-1",
                    "write_file",
                    '{"path":"generated.txt","content":"value"}',
                )
            ),
            text_reply("done"),
        ]
    )
    timeline: list[str] = []

    class TimelineExecutor(HookProbeExecutor):
        async def run(self, rule, payload, *, blocking):
            timeline.append(f"hook:{payload.get('kind')}")
            return await super().run(rule, payload, blocking=blocking)

    probe = TimelineExecutor()
    hooks = HookEngine([hook_rule("notify", HookEvent.NOTIFICATION)], [], executor=probe)
    agent = Agent(
        provider,
        registry_with(write),
        engine_for(tmp_path),
        runtime=runtime_for(tmp_path),
        hook_engine=hooks,
    )
    async for event in agent.run(conversation()):
        if event.approval is not None:
            timeline.append("approval")
            event.approval.respond.set_result(Outcome.DENY_ONCE)
    await hooks.aclose()

    assert timeline.index("hook:approval") < timeline.index("approval")
    assert probe.calls[0][1]["detail"] == "write_file"


@pytest.mark.asyncio
async def test_agent_emits_usage_and_context_for_streaming_and_non_streaming_requests() -> None:
    usage = TokenUsage(
        input_tokens=7,
        output_tokens=3,
        total_tokens=10,
        cache_write_tokens=5,
        cache_read_tokens=4,
    )
    streaming = FakeProvider(
        [(StreamEvent.delta("done"), StreamEvent.usage_report(usage), StreamEvent.completed())]
    )
    stream_events = await collect(build_agent(streaming, Registry()), conversation())

    result = ChatResult(Message(MessageRole.ASSISTANT, "done"), "fake-model", usage=usage)
    non_streaming = NonStreamingProvider(result)
    chat_events = await collect(
        build_agent(non_streaming, Registry()), conversation(), stream=False
    )

    assert [event.usage for event in stream_events if event.usage] == [usage]
    assert [event.usage for event in chat_events if event.usage] == [usage]
    assert streaming.request_contexts[0] is not None
    assert non_streaming.request_contexts[0] is not None
    assert streaming.request_contexts[0] == non_streaming.request_contexts[0]
    assert streaming.request_contexts[0].environment.startswith("Environment:\n")
    assert streaming.request_contexts[0].reminder == ""


@pytest.mark.asyncio
async def test_agent_degrades_when_environment_collection_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_environment(version: str, model: str) -> None:
        del version, model
        raise OSError("environment unavailable")

    monkeypatch.setattr("codewright.agent.gather_environment", fail_environment)
    provider = FakeProvider([text_reply("done")])

    events = await collect(build_agent(provider, Registry(), version="0.5.0"), conversation())

    assert events[-1] == Event.completed()
    assert provider.request_contexts == [RequestContext()]


@pytest.mark.asyncio
async def test_load_skill_is_visible_in_next_iteration_without_permission_prompt(
    tmp_path: Path,
) -> None:
    skill_path = tmp_path / ".codewright" / "skills" / "review.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\nname: review\ndescription: Review changes\n---\nFULL REVIEW SOP\n",
        encoding="utf-8",
    )
    loader = SkillLoader(tmp_path, tmp_path / "home")
    skills = loader.load_all()
    load_tool = LoadSkillTool(loader)
    registry = Registry()
    registry.register(load_tool)
    provider = FakeProvider(
        [
            tool_reply(ToolCall("load-1", "load_skill", '{"name":"review"}')),
            text_reply("done"),
        ]
    )
    agent = build_runtime_agent(provider, registry, tmp_path)
    load_tool.set_agent(agent)
    agent.set_skill_catalog(render_skill_catalog(skills))
    value = conversation()

    events = await collect(agent, value)

    assert len(provider.request_contexts) == 2
    first = provider.request_contexts[0]
    second = provider.request_contexts[1]
    assert first is not None and "## Active Skills" not in first.environment
    assert second is not None and "## Active Skills" in second.environment
    assert "FULL REVIEW SOP" in second.environment
    assert "FULL REVIEW SOP" not in value.messages()[0].content
    assert "## Available Skills" in value.messages()[0].content
    assert not any(event.approval is not None for event in events)


@pytest.mark.asyncio
async def test_agent_stops_at_iteration_limit_with_valid_history() -> None:
    provider = RepeatingToolProvider()
    tool = FakeTool("read_file", Result("ok"))
    value = conversation()

    events = await collect(build_agent(provider, registry_with(tool)), value)

    assert len(provider.requests) == MAX_ITERATIONS
    assert NOTICE_MAX_ITER in [event.notice for event in events]
    assert value.last_role() is MessageRole.ASSISTANT
    assert value.messages()[-1].content == NOTICE_MAX_ITER


@pytest.mark.asyncio
async def test_agent_uses_configurable_max_turns() -> None:
    provider = RepeatingToolProvider()
    tool = FakeTool("read_file", Result("ok"))
    base = build_agent(provider, registry_with(tool))
    agent = Agent(
        provider,
        base.registry,
        base.permission_engine,
        max_turns=2,
    )
    value = conversation()

    events = await collect(agent, value)

    assert len(provider.requests) == 2
    assert "（已达最大迭代轮数 2，自动停止；可继续发消息推进。）" in [
        event.notice for event in events
    ]


@pytest.mark.asyncio
async def test_allowed_tools_filter_definitions_and_block_forged_execution() -> None:
    read_tool = FakeTool("read_file", Result("read"))
    write_tool = FakeTool("write_file", Result("written"), read_only=False)
    registry = registry_with(read_tool, write_tool)
    provider = FakeProvider(
        [
            tool_reply(ToolCall("forged", "write_file", '{"path":"x"}')),
            text_reply("done"),
        ]
    )
    base = build_agent(provider, registry)
    agent = Agent(
        provider,
        registry,
        base.permission_engine,
        allowed_tools=frozenset({"read_file"}),
    )
    value = conversation()

    await collect(agent, value)

    assert [definition.name for definition in agent.definitions_for_mode(Mode.DEFAULT)] == [
        "read_file"
    ]
    assert agent.tool_count() == 1
    assert write_tool.calls == []
    result = value.messages()[3].tool_results[0]
    assert result.error_code == "tool_not_allowed"


def test_plan_definitions_apply_read_only_before_allowed_tools() -> None:
    registry = registry_with(
        FakeTool("read_file", Result("read")),
        FakeTool("write_file", Result("write"), read_only=False),
    )
    provider = FakeProvider([])
    base = build_agent(provider, registry)
    agent = Agent(
        provider,
        registry,
        base.permission_engine,
        allowed_tools=frozenset({"read_file", "write_file"}),
    )

    assert [definition.name for definition in agent.definitions_for_mode(Mode.PLAN)] == [
        "read_file"
    ]


@pytest.mark.asyncio
async def test_agent_stops_after_consecutive_unknown_tools() -> None:
    replies = [tool_reply(ToolCall(f"unknown-{index}", "missing")) for index in range(4)]
    provider = FakeProvider(replies)
    value = conversation()

    events = await collect(build_agent(provider, Registry()), value)

    assert len(provider.requests) == MAX_UNKNOWN_RUN
    assert NOTICE_UNKNOWN_TOOLS in [event.notice for event in events]
    assert all(
        message.tool_results[0].error_code == "permission_denied"
        for message in value.messages()[3::2]
    )
    assert value.last_role() is MessageRole.ASSISTANT


@pytest.mark.asyncio
async def test_known_tool_resets_unknown_tool_counter() -> None:
    replies = [
        tool_reply(ToolCall("u1", "missing")),
        tool_reply(ToolCall("u2", "missing")),
        tool_reply(ToolCall("known", "read_file", '{"path":"README.md"}')),
        tool_reply(ToolCall("u3", "missing")),
        tool_reply(ToolCall("u4", "missing")),
        text_reply("done"),
    ]
    provider = FakeProvider(replies)

    events = await collect(
        build_agent(provider, registry_with(FakeTool("read_file", Result("ok")))),
        conversation(),
    )

    assert len(provider.requests) == 6
    assert NOTICE_UNKNOWN_TOOLS not in [event.notice for event in events]


@pytest.mark.asyncio
async def test_read_only_tools_run_concurrently_before_side_effect_tool() -> None:
    probe = Probe()
    tools = (
        FakeTool("read_file", Result("one"), delay=0.03, probe=probe),
        FakeTool("glob", Result("two"), delay=0.03, probe=probe),
        FakeTool("write_file", Result("three"), read_only=False, probe=probe),
    )
    calls = (
        ToolCall("call-read", "read_file", '{"path":"README.md"}'),
        ToolCall("call-glob", "glob", '{"pattern":"**/*.py"}'),
        ToolCall(
            "call-write",
            "write_file",
            '{"path":"generated.txt","content":"value"}',
        ),
    )
    provider = FakeProvider([tool_reply(*calls), text_reply("done")])
    value = conversation()

    events = await collect(
        build_agent(provider, registry_with(*tools)),
        value,
        mode=Mode.ACCEPT_EDITS,
    )

    assert probe.peak == 2
    assert probe.timeline.index("start:write_file") > probe.timeline.index("end:read_file")
    assert probe.timeline.index("start:write_file") > probe.timeline.index("end:glob")
    starts = [event.tool.name for event in events if event.tool and event.tool.phase is Phase.START]
    ends = [event.tool.name for event in events if event.tool and event.tool.phase is Phase.END]
    assert starts == ["read_file", "glob", "write_file"]
    assert ends == ["read_file", "glob", "write_file"]
    assert [result.tool_name for result in value.messages()[3].tool_results] == [
        "read_file",
        "glob",
        "write_file",
    ]


@pytest.mark.asyncio
async def test_plan_mode_uses_read_only_definitions_and_request_only_reminder() -> None:
    read = FakeTool("read_file", Result("ok"))
    write = FakeTool("write_file", Result("ok"), read_only=False)
    provider = FakeProvider([text_reply("plan")])
    value = conversation()

    await collect(build_agent(provider, registry_with(read, write)), value, mode=Mode.PLAN)

    assert [definition.name for definition in provider.tool_definitions[0]] == ["read_file"]
    assert provider.requests[0][0].content == "You are Codewright."
    assert provider.request_contexts[0] is not None
    assert provider.request_contexts[0].reminder == plan_reminder(full=True)
    assert PLAN_MODE_REMINDER in provider.request_contexts[0].reminder
    assert PLAN_MODE_REMINDER not in value.messages()[0].content
    assert all(PLAN_MODE_REMINDER not in message.content for message in value.messages())


@pytest.mark.asyncio
async def test_plan_reminder_repeats_full_text_at_fixed_interval() -> None:
    replies = [
        tool_reply(ToolCall(f"call-{index}", "read_file", '{"path":"README.md"}'))
        for index in range(PLAN_REMINDER_INTERVAL)
    ]
    replies.append(text_reply("plan complete"))
    provider = FakeProvider(replies)
    tool = FakeTool("read_file", Result("ok"))
    value = conversation()

    await collect(build_agent(provider, registry_with(tool)), value, mode=Mode.PLAN)

    reminders = [context.reminder for context in provider.request_contexts if context is not None]
    assert reminders == [
        plan_reminder(full=True),
        *[plan_reminder(full=False)] * (PLAN_REMINDER_INTERVAL - 1),
        plan_reminder(full=True),
    ]
    assert all(
        reminder not in message.content for reminder in reminders for message in value.messages()
    )


@pytest.mark.asyncio
async def test_provider_wait_can_be_cancelled_and_conversation_remains_usable() -> None:
    provider = BlockingProvider()
    cancel_event = asyncio.Event()
    value = conversation()
    task = asyncio.create_task(
        collect(build_agent(provider, Registry()), value, cancel_event=cancel_event)
    )

    await provider.started.wait()
    cancel_event.set()
    events = await asyncio.wait_for(task, timeout=1)

    assert NOTICE_CANCELLED in [event.notice for event in events]
    assert events[-1].done is True
    assert value.messages()[-1] == Message(MessageRole.ASSISTANT, NOTICE_CANCELLED)


@pytest.mark.asyncio
async def test_tool_wait_can_be_cancelled_with_paired_results() -> None:
    tool = FakeTool("read_file", Result("late"), delay=10)
    call = ToolCall("call-1", "read_file", '{"path":"README.md"}')
    provider = FakeProvider([tool_reply(call)])
    cancel_event = asyncio.Event()
    value = conversation()
    task = asyncio.create_task(
        collect(
            build_agent(provider, registry_with(tool)),
            value,
            cancel_event=cancel_event,
        )
    )

    await tool.started.wait()
    cancel_event.set()
    events = await asyncio.wait_for(task, timeout=1)

    assert NOTICE_CANCELLED in [event.notice for event in events]
    assert [message.role for message in value.messages()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert value.messages()[3].tool_results[0].error_code == "cancelled"


@pytest.mark.asyncio
async def test_provider_error_is_safe_and_history_can_continue() -> None:
    error = LLMServiceError()
    value = conversation()

    events = await collect(
        build_agent(FakeProvider([(StreamEvent.failed(error),)]), Registry()), value
    )

    assert Event.failed(error) in events
    assert events[-1].done is True
    assert value.messages()[-1] == Message(MessageRole.ASSISTANT, NOTICE_STREAM_ERROR)
    value.add_user("try again")
    assert value.last_role() is MessageRole.USER


@pytest.mark.asyncio
async def test_duplicate_call_ids_stop_before_tool_execution() -> None:
    tool = FakeTool("read_file", Result("ok"))
    duplicate = (
        ToolCall("same", "read_file", '{"path":"README.md"}'),
        ToolCall("same", "read_file", '{"path":"README.md"}'),
    )

    events = await collect(
        build_agent(FakeProvider([tool_reply(*duplicate)]), registry_with(tool)),
        conversation(),
    )

    assert tool.calls == []
    assert any(event.error is not None for event in events)


@pytest.mark.asyncio
async def test_tool_summary_is_bounded_without_changing_history_result() -> None:
    content = "\n".join(f"line {index}" for index in range(20))
    tool = FakeTool("read_file", Result(content))
    provider = FakeProvider(
        [
            tool_reply(ToolCall("call-1", "read_file", '{"path":"README.md"}')),
            text_reply("done"),
        ]
    )
    value = conversation()

    events = await collect(build_agent(provider, registry_with(tool)), value)

    end = next(event.tool for event in events if event.tool and event.tool.phase is Phase.END)
    assert len(end.summary.splitlines()) == MAX_SUMMARY_LINES
    assert end.truncated is True
    assert value.messages()[3].tool_results[0].content == content


def test_approval_event_is_mutually_exclusive() -> None:
    loop = asyncio.new_event_loop()
    try:
        future: asyncio.Future[Outcome] = loop.create_future()
        request = ApprovalRequest("call-1", "bash", '{"command":"pwd"}', "confirm", future)
        assert Event.approval_requested(request).approval is request
        with pytest.raises(ValueError, match="exactly one"):
            Event(text="invalid", approval=request)
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_denied_and_allowed_read_results_keep_original_order(tmp_path: Path) -> None:
    read = FakeTool("read_file", Result("secret"))
    glob = FakeTool("glob", Result("src/main.py"))
    registry = registry_with(read, glob)
    engine = engine_for(tmp_path)
    engine.local.deny.append(Rule("Read", ExactMatcher("secret.txt"), False, "=secret.txt"))
    calls = (
        ToolCall("denied", "read_file", '{"path":"secret.txt"}'),
        ToolCall("allowed", "glob", '{"pattern":"**/*.py"}'),
    )
    provider = FakeProvider([tool_reply(*calls), text_reply("done")])
    value = conversation()

    events = await collect(Agent(provider, registry, engine), value)

    results = value.messages()[3].tool_results
    assert [result.tool_call_id for result in results] == ["denied", "allowed"]
    assert results[0].error_code == "permission_denied"
    assert results[1].is_error is False
    assert read.calls == []
    assert glob.calls == ['{"pattern":"**/*.py"}']
    assert not any(event.approval for event in events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "executions", "is_error"),
    [
        (Outcome.ALLOW_ONCE, 1, False),
        (Outcome.DENY_ONCE, 0, True),
    ],
)
async def test_approval_outcome_controls_side_effect_execution(
    tmp_path: Path,
    outcome: Outcome,
    executions: int,
    is_error: bool,
) -> None:
    write = FakeTool("write_file", Result("written"), read_only=False)
    tool_call = ToolCall(
        "write-1",
        "write_file",
        '{"path":"generated.txt","content":"value"}',
    )
    provider = FakeProvider([tool_reply(tool_call), text_reply("done")])
    value = conversation()

    events = await collect_with_outcome(
        Agent(provider, registry_with(write), engine_for(tmp_path)),
        value,
        outcome,
    )

    approvals = [event.approval for event in events if event.approval]
    assert len(approvals) == 1
    assert len(write.calls) == executions
    result = value.messages()[3].tool_results[0]
    assert result.is_error is is_error
    assert result.error_code == ("permission_denied" if is_error else None)


@pytest.mark.asyncio
async def test_allow_forever_persists_rule_and_updates_live_engine(tmp_path: Path) -> None:
    write = FakeTool("write_file", Result("written"), read_only=False)
    tool_call = ToolCall(
        "write-1",
        "write_file",
        '{"path":"generated.txt","content":"value"}',
    )
    provider = FakeProvider([tool_reply(tool_call), text_reply("done")])
    value = conversation()
    engine = engine_for(tmp_path)

    await collect_with_outcome(
        Agent(provider, registry_with(write), engine),
        value,
        Outcome.ALLOW_FOREVER,
    )

    assert "Write(generated.txt)" in engine.local_path.read_text(encoding="utf-8")
    assert engine.check(Mode.DEFAULT, tool_call, False)[0] is Decision.ALLOW


@pytest.mark.asyncio
async def test_persist_failure_notifies_but_allows_current_call(tmp_path: Path) -> None:
    write = FakeTool("write_file", Result("written"), read_only=False)
    tool_call = ToolCall(
        "write-1",
        "write_file",
        '{"path":"generated.txt","content":"value"}',
    )
    provider = FakeProvider([tool_reply(tool_call), text_reply("done")])
    value = conversation()
    engine = engine_for(tmp_path)
    engine.local_path.parent.mkdir()
    engine.local_path.write_text("permissions: [", encoding="utf-8")

    events = await collect_with_outcome(
        Agent(provider, registry_with(write), engine),
        value,
        Outcome.ALLOW_FOREVER,
    )

    assert write.calls
    assert "永久规则保存失败，本次仅允许。" in [event.notice for event in events]
    assert value.messages()[3].tool_results[0].is_error is False


@pytest.mark.asyncio
async def test_cancel_event_releases_pending_approval_and_repairs_history(
    tmp_path: Path,
) -> None:
    write = FakeTool("write_file", Result("written"), read_only=False)
    tool_call = ToolCall(
        "write-1",
        "write_file",
        '{"path":"generated.txt","content":"value"}',
    )
    provider = FakeProvider([tool_reply(tool_call)])
    value = conversation()
    cancel_event = asyncio.Event()
    approval_seen = asyncio.Event()
    approvals: list[ApprovalRequest] = []

    async def consume() -> list[Event]:
        events: list[Event] = []
        async for event in Agent(
            provider,
            registry_with(write),
            engine_for(tmp_path),
        ).run(value, cancel_event=cancel_event):
            events.append(event)
            if event.approval is not None:
                approvals.append(event.approval)
                approval_seen.set()
        return events

    task = asyncio.create_task(consume())
    await asyncio.wait_for(approval_seen.wait(), timeout=1)
    cancel_event.set()
    events = await asyncio.wait_for(task, timeout=1)

    assert approvals[0].respond.cancelled()
    assert write.calls == []
    assert NOTICE_CANCELLED in [event.notice for event in events]
    assert value.messages()[3].tool_results[0].error_code == "cancelled"
    assert value.last_role() is MessageRole.ASSISTANT


class RecordingMemoryManager(Manager):
    def __init__(self, tmp_path: Path) -> None:
        super().__init__(
            str(tmp_path / "project-memory"),
            str(tmp_path / "user-memory"),
            None,
            "fake-model",
        )
        self.index = "initial memory"
        self.updates: list[list[Message]] = []
        self.updated = asyncio.Event()

    def load_index(self) -> str:
        return self.index

    async def update_async(self, recent_messages: list[Message]) -> None:
        self.updates.append(list(recent_messages))
        self.updated.set()


@pytest.mark.asyncio
async def test_agent_refreshes_dynamic_memory_prompt_on_each_turn(tmp_path: Path) -> None:
    provider = FakeProvider([text_reply("one"), text_reply("two")])
    manager = RecordingMemoryManager(tmp_path)
    value = Conversation("stale")
    value.add_user("first")
    agent = Agent(
        provider,
        Registry(),
        engine_for(tmp_path),
        memory_manager=manager,
        instruction_text="project rules",
        base_prompt="configured base",
    )

    await collect(agent, value)
    manager.index = "fresh memory"
    value.add_user("second")
    await collect(agent, value)

    first_system = provider.requests[0][0].content
    second_system = provider.requests[1][0].content
    assert "configured base" in first_system
    assert "project rules" in first_system
    assert "initial memory" in first_system
    assert "fresh memory" in second_system
    assert "initial memory" not in second_system


@pytest.mark.asyncio
async def test_memory_updates_on_keyword_and_every_fifth_turn(tmp_path: Path) -> None:
    provider = FakeProvider([text_reply(str(index)) for index in range(5)])
    manager = RecordingMemoryManager(tmp_path)
    value = Conversation("system")
    agent = Agent(
        provider,
        Registry(),
        engine_for(tmp_path),
        memory_manager=manager,
    )

    prompts = ["普通问题", "请记住我的偏好", "第三问", "第四问", "第五问"]
    for prompt in prompts:
        value.add_user(prompt)
        await collect(agent, value)
        await asyncio.sleep(0)

    await agent.shutdown_memory_updates()
    assert agent.runtime.turn_count == 5
    assert len(manager.updates) == 2
    assert manager.updates[0][0].content == "请记住我的偏好"
    assert manager.updates[1][0].content == "第五问"
    assert all(messages[-1].role is MessageRole.ASSISTANT for messages in manager.updates)
