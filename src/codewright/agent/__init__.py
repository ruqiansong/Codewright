"""Protocol-neutral ReAct loop orchestration for Codewright."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self, cast

from codewright.agent.completion import (
    CompletionResult,
    MaxTurnsReached,
    consume_events,
)
from codewright.agent.context import (
    ExecutionContext,
    bind_execution_context,
    reset_execution_context,
)
from codewright.agent.runtime import SessionRuntime
from codewright.compact import (
    CompactCircuitBreaker,
    ContentReplacementState,
    ManageInput,
    RecoveryState,
    TriggerKind,
    manage_context,
    new_session_context,
)
from codewright.compact.const import MANUAL_SAFETY_MARGIN
from codewright.compact.token import estimate_tokens, usage_anchor
from codewright.conversation import Conversation
from codewright.hook import DispatchResult, Payload
from codewright.hook import Engine as HookEngine
from codewright.hook import Event as HookEvent
from codewright.llm import (
    LLMError,
    LLMResponseError,
    Message,
    MessageRole,
    PromptTooLongError,
    Provider,
    RequestContext,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from codewright.memory import Manager
from codewright.permission import Decision, Engine, Mode, Outcome
from codewright.prompt import (
    build_system_prompt,
    gather_environment,
    plan_reminder,
    render_active_skills,
)
from codewright.skills import ActiveEntry
from codewright.tool import Registry, Result
from codewright.tool.load_skill import bind_execution_agent, reset_execution_agent
from codewright.utils.logging import redact_sensitive

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 25
MAX_UNKNOWN_RUN = 3
MAX_SUMMARY_LINES = 8
MAX_SUMMARY_LINE_CHARS = 500
PLAN_REMINDER_INTERVAL = 4
EMPTY_RESPONSE_MESSAGE = "模型未返回可显示的最终答复。"
NOTICE_MAX_ITER = "（已达最大迭代轮数 25，自动停止；可继续发消息推进。）"
NOTICE_UNKNOWN_TOOLS = "（连续多轮只请求到未注册的工具，自动停止。）"
NOTICE_STREAM_ERROR = "（请求出错，本轮已中断。）"
NOTICE_CANCELLED = "（已取消。）"
_MEMORY_SIGNALS = ("记住", "记忆", "别忘", "remember", "memo")


class Phase(StrEnum):
    """Lifecycle phase for one tool invocation."""

    START = "start"
    END = "end"


class CompactPhase(StrEnum):
    """Lifecycle phase for automatic and emergency context compaction."""

    BEFORE_AUTO = "before_auto"
    AFTER_AUTO = "after_auto"
    BEFORE_EMERGENCY = "before_emergency"
    AFTER_EMERGENCY = "after_emergency"
    BEFORE_MANUAL = "before_manual"
    AFTER_MANUAL = "after_manual"


@dataclass(frozen=True, slots=True)
class CompactEvent:
    """One context-compaction lifecycle update for presentation layers."""

    phase: CompactPhase
    before: int = 0
    after: int = 0
    error_message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.phase, CompactPhase):
            raise TypeError("phase must be a CompactPhase")
        if self.before < 0 or self.after < 0:
            raise ValueError("compact token counts must not be negative")
        if not isinstance(self.error_message, str):
            raise TypeError("error_message must be a string")


@dataclass(frozen=True, slots=True)
class ToolEvent:
    """A provider-neutral tool lifecycle event for presentation layers."""

    call_id: str
    name: str
    arguments_json: str
    phase: Phase
    summary: str = ""
    is_error: bool = False
    truncated: bool = False
    argument_summary: str = ""

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not isinstance(self.arguments_json, str):
            raise TypeError("arguments_json must be a string")
        if not isinstance(self.phase, Phase):
            raise TypeError("phase must be a Phase")
        if self.phase is Phase.START and (self.summary or self.is_error or self.truncated):
            raise ValueError("a start tool event cannot contain result state")
        if self.phase is Phase.END and not self.summary:
            raise ValueError("an end tool event must contain a summary")


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """One side-effecting tool call awaiting a single user decision."""

    call_id: str
    name: str
    arguments_json: str
    reason: str
    respond: asyncio.Future[Outcome]

    def __post_init__(self) -> None:
        if not self.call_id.strip():
            raise ValueError("call_id must not be empty")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if not isinstance(self.arguments_json, str):
            raise TypeError("arguments_json must be a string")
        if not self.reason.strip():
            raise ValueError("reason must not be empty")
        if not isinstance(self.respond, asyncio.Future):
            raise TypeError("respond must be an asyncio Future")


ApprovalUpgrader = Callable[[ApprovalRequest], Awaitable[Outcome]]
SubagentKind = Literal["main", "defined", "fork", "skill"]


@dataclass(frozen=True, slots=True)
class Event:
    """One mutually exclusive Agent event consumed by the TUI."""

    text: str = ""
    tool: ToolEvent | None = None
    approval: ApprovalRequest | None = None
    compact: CompactEvent | None = None
    usage: TokenUsage | None = None
    iteration: int = 0
    notice: str = ""
    done: bool = False
    error: LLMError | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if self.tool is not None and not isinstance(self.tool, ToolEvent):
            raise TypeError("tool must be a ToolEvent")
        if self.approval is not None and not isinstance(self.approval, ApprovalRequest):
            raise TypeError("approval must be an ApprovalRequest")
        if self.compact is not None and not isinstance(self.compact, CompactEvent):
            raise TypeError("compact must be a CompactEvent")
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise TypeError("usage must be a TokenUsage")
        if not isinstance(self.iteration, int) or isinstance(self.iteration, bool):
            raise TypeError("iteration must be an integer")
        if self.iteration < 0:
            raise ValueError("iteration must not be negative")
        if not isinstance(self.notice, str):
            raise TypeError("notice must be a string")
        if not isinstance(self.done, bool):
            raise TypeError("done must be a boolean")
        if self.error is not None and not isinstance(self.error, LLMError):
            raise TypeError("error must be an LLMError")
        states = sum(
            (
                bool(self.text),
                self.tool is not None,
                self.approval is not None,
                self.compact is not None,
                self.usage is not None,
                self.iteration > 0,
                bool(self.notice),
                self.done,
                self.error is not None,
            )
        )
        if states != 1:
            raise ValueError("an Agent event must have exactly one event state")

    @classmethod
    def delta(cls, text: str) -> Self:
        return cls(text=text)

    @classmethod
    def tool_event(cls, event: ToolEvent) -> Self:
        return cls(tool=event)

    @classmethod
    def approval_requested(cls, request: ApprovalRequest) -> Self:
        return cls(approval=request)

    @classmethod
    def compact_event(cls, event: CompactEvent) -> Self:
        return cls(compact=event)

    @classmethod
    def usage_report(cls, usage: TokenUsage) -> Self:
        return cls(usage=usage)

    @classmethod
    def iteration_started(cls, iteration: int) -> Self:
        return cls(iteration=iteration)

    @classmethod
    def notification(cls, notice: str) -> Self:
        return cls(notice=notice)

    @classmethod
    def completed(cls) -> Self:
        return cls(done=True)

    @classmethod
    def failed(cls, error: LLMError) -> Self:
        return cls(error=error)


class Agent:
    """Run a bounded ReAct loop for one user turn."""

    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        engine: Engine,
        *,
        version: str = "dev",
        runtime: SessionRuntime | None = None,
        memory_manager: Manager | None = None,
        instruction_text: str = "",
        base_prompt: str | None = None,
        hook_engine: HookEngine | None = None,
        max_turns: int = MAX_ITERATIONS,
        allowed_tools: frozenset[str] | None = None,
        permission_mode: Mode | None = None,
        dont_ask: bool = False,
        approval_upgrader: ApprovalUpgrader | None = None,
        subagent_name: str = "",
        subagent_kind: SubagentKind = "main",
    ) -> None:
        if not isinstance(max_turns, int) or isinstance(max_turns, bool) or max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        if allowed_tools is not None and (
            not isinstance(allowed_tools, frozenset)
            or any(
                not isinstance(name, str) or not name or name != name.strip()
                for name in allowed_tools
            )
        ):
            raise TypeError(
                "allowed_tools must be a frozenset of trimmed non-empty strings or None"
            )
        if permission_mode is not None and not isinstance(permission_mode, Mode):
            raise TypeError("permission_mode must be a Mode or None")
        if not isinstance(dont_ask, bool):
            raise TypeError("dont_ask must be a boolean")
        if approval_upgrader is not None and not callable(approval_upgrader):
            raise TypeError("approval_upgrader must be callable or None")
        if not isinstance(subagent_name, str):
            raise TypeError("subagent_name must be a string")
        if subagent_name != subagent_name.strip():
            raise ValueError("subagent_name must be trimmed")
        if subagent_kind not in ("main", "defined", "fork", "skill"):
            raise ValueError("subagent_kind is invalid")
        self._provider = provider
        self._registry = registry
        self._engine = engine
        self._version = version
        self._compact_enabled = runtime is not None
        self.runtime = runtime or SessionRuntime(
            replacement=ContentReplacementState(),
            recovery=RecoveryState(),
            auto_tracking=CompactCircuitBreaker(),
            session=new_session_context("."),
        )
        self._run_lock = asyncio.Lock()
        self._memory_manager = memory_manager
        self._instruction_text = instruction_text
        self._base_prompt = base_prompt
        self._hook_engine = hook_engine
        self.max_turns = max_turns
        self.allowed_tools = allowed_tools
        self.permission_mode = permission_mode
        self.dont_ask = dont_ask
        self.approval_upgrader = approval_upgrader
        self.subagent_name = subagent_name
        self.subagent_kind = subagent_kind
        self._skill_catalog = ""
        self._memory_tasks: set[asyncio.Task[None]] = set()

    def activate_skill(self, name: str, prompt_body: str, source_dir: Path) -> None:
        """Pin one Skill body to the current session."""
        self.runtime.active_skills.activate(name, prompt_body, source_dir)

    def clear_active_skills(self) -> None:
        """Clear every Skill pinned to the current session."""
        self.runtime.active_skills.clear()

    def list_active_skills(self) -> tuple[ActiveEntry, ...]:
        """Return the active Skill snapshot in activation order."""
        return self.runtime.active_skills.snapshot()

    def set_skill_catalog(self, catalog: str) -> None:
        """Replace the bounded Skill catalog used by future system prompts."""
        if not isinstance(catalog, str):
            raise TypeError("catalog must be a string")
        self._skill_catalog = catalog

    def refresh_system_prompt(self, conversation: Conversation) -> None:
        """Refresh dynamic system modules without starting a Provider request."""
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        self._refresh_system_prompt(conversation)

    @property
    def provider(self) -> Provider:
        """Return the Provider shared by isolated child executions."""
        return self._provider

    @property
    def registry(self) -> Registry:
        """Return the read-only Tool Registry dependency reference."""
        return self._registry

    @property
    def permission_engine(self) -> Engine:
        """Return the shared permission decision engine."""
        return self._engine

    @property
    def version(self) -> str:
        """Return the Codewright version used in dynamic environments."""
        return self._version

    @property
    def hook_engine(self) -> HookEngine | None:
        """Return the Hook Engine shared with isolated child Agents."""
        return self._hook_engine

    async def run(
        self,
        conversation: Conversation,
        *,
        stream: bool = True,
        mode: Mode = Mode.DEFAULT,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        """Yield presentation events until the bounded Agent loop stops."""
        effective_mode = self.permission_mode if self.permission_mode is not None else mode
        async with self._run_lock:
            self._refresh_system_prompt(conversation)
            async for event in self._run_locked(
                conversation,
                stream=stream,
                mode=effective_mode,
                cancel_event=cancel_event,
            ):
                yield event

    async def run_to_completion(
        self,
        conversation: Conversation,
        task: str = "",
        *,
        stream: bool = True,
        cancel_event: asyncio.Event | None = None,
        event_sink: asyncio.Queue[Event | None] | None = None,
    ) -> CompletionResult:
        """Append an optional task and consume the existing Agent event stream."""
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        if not isinstance(task, str):
            raise TypeError("task must be a string")
        if not isinstance(stream, bool):
            raise TypeError("stream must be a boolean")
        if cancel_event is not None and not isinstance(cancel_event, asyncio.Event):
            raise TypeError("cancel_event must be an asyncio.Event or None")
        if event_sink is not None and not isinstance(event_sink, asyncio.Queue):
            raise TypeError("event_sink must be an asyncio.Queue or None")
        if task:
            conversation.add_user(task)
        return await consume_events(
            self.run(
                conversation,
                stream=stream,
                cancel_event=cancel_event,
            ),
            event_sink,
        )

    async def _run_locked(
        self,
        conversation: Conversation,
        *,
        stream: bool,
        mode: Mode,
        cancel_event: asyncio.Event | None,
    ) -> AsyncIterator[Event]:
        """Run one turn while the public run lock is held."""
        definitions = self.definitions_for_mode(mode)
        unknown_run = 0
        pending_calls: tuple[ToolCall, ...] = ()
        last_text = ""
        try:
            environment_text = (
                await gather_environment(self._version, self._provider.model_name)
            ).render()
        except Exception:
            environment_text = ""

        try:
            for iteration in range(1, self.max_turns + 1):
                if _is_cancelled(cancel_event):
                    self._ensure_assistant_tail(conversation, NOTICE_CANCELLED)
                    yield Event.notification(NOTICE_CANCELLED)
                    yield Event.completed()
                    return

                if self._compact_enabled:
                    compact_started: list[int] = []
                    compact_signal = asyncio.Event()

                    async def notify_compact_started(
                        tokens: int,
                        started: list[int] = compact_started,
                        signal: asyncio.Event = compact_signal,
                    ) -> None:
                        await self._dispatch_hook(
                            HookEvent.PRE_COMPACT,
                            {"trigger": TriggerKind.AUTO.value},
                            mode,
                        )
                        started.append(tokens)
                        signal.set()

                    compact_task = asyncio.create_task(
                        manage_context(
                            self._manage_input(
                                conversation,
                                definitions,
                                TriggerKind.AUTO,
                                on_layer2_start=notify_compact_started,
                            )
                        )
                    )
                    signal_task = asyncio.create_task(compact_signal.wait())
                    try:
                        await asyncio.wait(
                            {compact_task, signal_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        before_emitted = compact_signal.is_set()
                        if before_emitted:
                            yield Event.compact_event(
                                CompactEvent(
                                    CompactPhase.BEFORE_AUTO,
                                    before=compact_started[0],
                                )
                            )
                        compact_output = await compact_task
                    except LLMError as error:
                        if before_emitted:
                            yield Event.compact_event(
                                CompactEvent(
                                    CompactPhase.AFTER_AUTO,
                                    before=compact_started[0],
                                    error_message=error.safe_message,
                                )
                            )
                        yield Event.failed(error)
                        yield Event.completed()
                        return
                    except Exception:
                        if before_emitted:
                            yield Event.compact_event(
                                CompactEvent(
                                    CompactPhase.AFTER_AUTO,
                                    before=compact_started[0],
                                    error_message="上下文压缩失败。",
                                )
                            )
                        yield Event.failed(LLMResponseError("上下文压缩失败。"))
                        yield Event.completed()
                        return
                    finally:
                        signal_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await signal_task
                        if not compact_task.done():
                            compact_task.cancel()
                            with suppress(asyncio.CancelledError):
                                await compact_task
                    if compact_output.history_rewritten:
                        self.runtime.reset_anchor()
                    if compact_output.layer2_started:
                        if not before_emitted:
                            yield Event.compact_event(
                                CompactEvent(
                                    CompactPhase.BEFORE_AUTO,
                                    before=compact_output.before_tokens,
                                )
                            )
                        yield Event.compact_event(
                            CompactEvent(
                                CompactPhase.AFTER_AUTO,
                                before=compact_output.before_tokens,
                                after=compact_output.after_tokens,
                            )
                        )
                        await self._dispatch_hook(
                            HookEvent.POST_COMPACT,
                            {
                                "trigger": TriggerKind.AUTO.value,
                                "before_tokens": compact_output.before_tokens,
                                "after_tokens": compact_output.after_tokens,
                            },
                            mode,
                        )

                yield Event.iteration_started(iteration)
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                done_seen = False
                provider_error: LLMError | None = None
                stream_error_seen = False
                latest_usage: TokenUsage | None = None
                emergency_retried = False
                while True:
                    await self._dispatch_hook(
                        HookEvent.PRE_USER_MESSAGE,
                        {"prompt": _last_user_prompt(conversation)},
                        mode,
                    )
                    reminder = self._build_reminder(mode, iteration)
                    active_text = render_active_skills(self.runtime.active_skills.snapshot())
                    request_environment = "\n\n".join(
                        part for part in (environment_text, active_text) if part
                    )
                    request_context = RequestContext(
                        environment=request_environment,
                        reminder=reminder,
                    )
                    provider_events = self._provider_events(
                        conversation.messages(),
                        definitions,
                        request_context,
                        stream=stream,
                    )
                    while True:
                        try:
                            event, cancelled, exhausted = await _next_event(
                                provider_events,
                                cancel_event,
                            )
                        except LLMError as error:
                            provider_error = error
                            stream_error_seen = True
                            break
                        except Exception:
                            provider_error = LLMResponseError(
                                "An unexpected error interrupted the response."
                            )
                            stream_error_seen = True
                            break
                        if cancelled:
                            self._ensure_assistant_tail(conversation, NOTICE_CANCELLED)
                            yield Event.notification(NOTICE_CANCELLED)
                            yield Event.completed()
                            return
                        if exhausted:
                            break
                        if event is None:
                            raise RuntimeError("provider event state is inconsistent")
                        if done_seen:
                            provider_error = LLMResponseError()
                            break
                        if event.text:
                            text_parts.append(event.text)
                            yield Event.delta(event.text)
                        elif event.tool_calls:
                            calls.extend(event.tool_calls)
                        elif event.usage is not None:
                            latest_usage = event.usage
                            yield Event.usage_report(event.usage)
                        elif event.error is not None:
                            provider_error = event.error
                            stream_error_seen = True
                            break
                        elif event.done:
                            done_seen = True

                    if not (
                        self._compact_enabled
                        and isinstance(provider_error, PromptTooLongError)
                        and not emergency_retried
                    ):
                        break
                    emergency_retried = True
                    await _close_iterator(provider_events)
                    stream_error_seen = False
                    before = self._estimated_tokens(conversation)
                    await self._dispatch_hook(
                        HookEvent.PRE_COMPACT,
                        {"trigger": TriggerKind.EMERGENCY.value},
                        mode,
                    )
                    yield Event.compact_event(
                        CompactEvent(CompactPhase.BEFORE_EMERGENCY, before=before)
                    )
                    try:
                        compact_output = await manage_context(
                            self._manage_input(
                                conversation,
                                definitions,
                                TriggerKind.EMERGENCY,
                            )
                        )
                    except LLMError as compact_error:
                        yield Event.compact_event(
                            CompactEvent(
                                CompactPhase.AFTER_EMERGENCY,
                                before=before,
                                error_message=compact_error.safe_message,
                            )
                        )
                        provider_error = compact_error
                        break
                    except Exception:
                        provider_error = LLMResponseError("上下文紧急压缩失败。")
                        yield Event.compact_event(
                            CompactEvent(
                                CompactPhase.AFTER_EMERGENCY,
                                before=before,
                                error_message=provider_error.safe_message,
                            )
                        )
                        break
                    self.runtime.reset_anchor()
                    await self._dispatch_hook(
                        HookEvent.POST_COMPACT,
                        {
                            "trigger": TriggerKind.EMERGENCY.value,
                            "before_tokens": compact_output.before_tokens,
                            "after_tokens": compact_output.after_tokens,
                        },
                        mode,
                    )
                    yield Event.compact_event(
                        CompactEvent(
                            CompactPhase.AFTER_EMERGENCY,
                            before=compact_output.before_tokens,
                            after=compact_output.after_tokens,
                        )
                    )
                    if compact_output.after_tokens >= (
                        self.runtime.context_window - MANUAL_SAFETY_MARGIN
                    ):
                        provider_error = PromptTooLongError()
                        break
                    text_parts.clear()
                    calls.clear()
                    done_seen = False
                    provider_error = None
                    stream_error_seen = False
                    latest_usage = None

                if provider_error is not None or not done_seen:
                    final_error = provider_error or LLMResponseError()
                    self._ensure_assistant_tail(conversation, NOTICE_STREAM_ERROR)
                    if stream_error_seen or provider_error is None:
                        await self._dispatch_hook(
                            HookEvent.NOTIFICATION,
                            {"kind": "stream_error", "detail": final_error.safe_message},
                            mode,
                        )
                    yield Event.failed(final_error)
                    yield Event.completed()
                    return
                if len({call.id for call in calls}) != len(calls):
                    self._ensure_assistant_tail(conversation, NOTICE_STREAM_ERROR)
                    duplicate_error = LLMResponseError()
                    await self._dispatch_hook(
                        HookEvent.NOTIFICATION,
                        {"kind": "stream_error", "detail": duplicate_error.safe_message},
                        mode,
                    )
                    yield Event.failed(duplicate_error)
                    yield Event.completed()
                    return

                assistant_text = "".join(text_parts)
                if assistant_text:
                    last_text = assistant_text
                if not calls:
                    if not assistant_text.strip():
                        assistant_text = EMPTY_RESPONSE_MESSAGE
                        yield Event.delta(assistant_text)
                    conversation.add_assistant(assistant_text)
                    self._update_usage_anchor(latest_usage, conversation)
                    self._schedule_memory_update(conversation)
                    await self._dispatch_hook(HookEvent.STOP, {"iter": iteration}, mode)
                    yield Event.completed()
                    return

                pending_calls = tuple(calls)
                conversation.add_assistant_with_tool_calls(assistant_text, pending_calls)
                self._update_usage_anchor(latest_usage, conversation)
                unknown_run = unknown_run + 1 if self._all_unknown(pending_calls) else 0
                tool_results: list[ToolResult] = []
                index = 0

                while index < len(pending_calls):
                    end = index + 1
                    if self._registry.is_read_only(pending_calls[index].name):
                        while end < len(pending_calls) and self._registry.is_read_only(
                            pending_calls[end].name
                        ):
                            end += 1
                    batch = pending_calls[index:end]
                    for call in batch:
                        yield Event.tool_event(_tool_start(call))

                    batch_results: list[Result]
                    cancelled = False
                    hook_results = [await self._pre_tool_result(call, mode) for call in batch]
                    if len(batch) > 1:
                        batch_results, cancelled = await self._execute_read_batch_with_hooks(
                            batch, hook_results, mode, cancel_event, conversation
                        )
                    else:
                        call = batch[0]
                        if hook_results[0] is not None:
                            batch_results = [hook_results[0]]
                        else:
                            decision, reason = self._engine.check(
                                mode,
                                call,
                                self._registry.is_read_only(call.name),
                            )
                            if decision is Decision.DENY:
                                batch_results = [_permission_denied_result(reason)]
                            elif decision is Decision.ALLOW:
                                batch_results, cancelled = await self._execute_batch(
                                    batch,
                                    cancel_event,
                                    conversation,
                                )
                            else:
                                outcome: Outcome | None
                                if self.dont_ask:
                                    outcome = Outcome.ALLOW_ONCE
                                else:
                                    response = asyncio.get_running_loop().create_future()
                                    approval = ApprovalRequest(
                                        call.id,
                                        call.name,
                                        call.arguments_json,
                                        reason,
                                        response,
                                    )
                                    if self.approval_upgrader is not None:
                                        outcome, cancelled = await _await_approval_upgrader(
                                            self.approval_upgrader,
                                            approval,
                                            cancel_event,
                                        )
                                    else:
                                        await self._dispatch_hook(
                                            HookEvent.NOTIFICATION,
                                            {"kind": "approval", "detail": call.name},
                                            mode,
                                        )
                                        yield Event.approval_requested(approval)
                                        outcome, cancelled = await _await_approval(
                                            approval,
                                            cancel_event,
                                        )
                                if cancelled:
                                    batch_results = [_cancelled_result()]
                                elif outcome not in {
                                    Outcome.ALLOW_ONCE,
                                    Outcome.ALLOW_FOREVER,
                                }:
                                    batch_results = [
                                        _permission_denied_result("用户拒绝了本次工具调用。")
                                    ]
                                else:
                                    if outcome is Outcome.ALLOW_FOREVER:
                                        try:
                                            self._engine.persist_local_allow(call)
                                        except Exception as error:
                                            logger.warning(
                                                "Persistent permission save failed error=%s",
                                                type(error).__name__,
                                            )
                                            yield Event.notification(
                                                "永久规则保存失败，本次仅允许。"
                                            )
                                    batch_results, cancelled = await self._execute_batch(
                                        batch,
                                        cancel_event,
                                        conversation,
                                    )
                    converted = [
                        self._to_tool_result(call, result)
                        for call, result in zip(batch, batch_results, strict=True)
                    ]
                    await self._record_read_results(batch, batch_results)
                    tool_results.extend(converted)
                    for call, result in zip(batch, converted, strict=True):
                        await self._dispatch_hook(
                            HookEvent.POST_TOOL_USE,
                            {
                                "tool_name": call.name,
                                "tool_input": _tool_input(call),
                                "tool_result": result.content,
                                "is_error": result.is_error,
                            },
                            mode,
                        )
                        yield Event.tool_event(_tool_end(call, result))

                    index = end
                    if cancelled:
                        tool_results.extend(
                            _cancelled_tool_result(call) for call in pending_calls[index:]
                        )
                        conversation.add_tool_results(tool_results)
                        pending_calls = ()
                        self._ensure_assistant_tail(conversation, NOTICE_CANCELLED)
                        yield Event.notification(NOTICE_CANCELLED)
                        yield Event.completed()
                        return

                conversation.add_tool_results(tool_results)
                pending_calls = ()

                if unknown_run >= MAX_UNKNOWN_RUN:
                    self._ensure_assistant_tail(conversation, NOTICE_UNKNOWN_TOOLS)
                    yield Event.notification(NOTICE_UNKNOWN_TOOLS)
                    yield Event.completed()
                    return

            if self.subagent_kind != "main":
                raise MaxTurnsReached(last_text)
            max_turns_notice = _max_turns_notice(self.max_turns)
            self._ensure_assistant_tail(conversation, max_turns_notice)
            yield Event.notification(max_turns_notice)
            yield Event.completed()
        except asyncio.CancelledError:
            if pending_calls:
                conversation.add_tool_results(
                    tuple(_cancelled_tool_result(call) for call in pending_calls)
                )
            self._ensure_assistant_tail(conversation, NOTICE_CANCELLED)
            raise
        except MaxTurnsReached:
            raise
        except Exception:
            if pending_calls:
                conversation.add_tool_results(
                    tuple(_failed_tool_result(call) for call in pending_calls)
                )
            self._ensure_assistant_tail(conversation, NOTICE_STREAM_ERROR)
            yield Event.failed(LLMResponseError("An unexpected error interrupted the response."))
            yield Event.completed()

    async def _provider_events(
        self,
        messages: tuple[Message, ...],
        definitions: tuple[ToolDefinition, ...],
        request_context: RequestContext,
        *,
        stream: bool,
    ) -> AsyncIterator[StreamEvent]:
        if stream:
            async for event in self._provider.stream_chat(
                messages,
                tools=definitions,
                request_context=request_context,
            ):
                yield event
            return

        result = await self._provider.chat(
            messages,
            tools=definitions,
            request_context=request_context,
        )
        if result.message.content:
            yield StreamEvent.delta(result.message.content)
        if result.message.tool_calls:
            yield StreamEvent.tool_calls_ready(result.message.tool_calls)
        if result.usage is not None:
            yield StreamEvent.usage_report(result.usage)
        yield StreamEvent.completed()

    def _estimated_tokens(self, conversation: Conversation) -> int:
        anchor, anchor_length, _ = self.runtime.anchor_snapshot()
        return estimate_tokens(anchor, conversation.messages(), anchor_length)

    def _manage_input(
        self,
        conversation: Conversation,
        definitions: Sequence[ToolDefinition],
        trigger: TriggerKind,
        *,
        on_layer2_start: Callable[[int], Awaitable[None] | None] | None = None,
    ) -> ManageInput:
        anchor, anchor_length, context_window = self.runtime.anchor_snapshot()
        return ManageInput(
            conv=conversation,
            provider=self._provider,
            context_window=context_window,
            tool_defs=definitions,
            replacement=self.runtime.replacement,
            recovery=self.runtime.recovery,
            auto_tracking=self.runtime.auto_tracking,
            session=self.runtime.session,
            usage_anchor=anchor,
            anchor_msg_len=anchor_length,
            estimated_token=estimate_tokens(anchor, conversation.messages(), anchor_length),
            trigger=trigger,
            on_layer2_start=on_layer2_start,
        )

    async def _dispatch_hook(
        self,
        event: HookEvent,
        payload: Payload,
        mode: Mode,
    ) -> DispatchResult:
        if self._hook_engine is None:
            return DispatchResult()
        complete_payload: Payload = {
            "event": event.value,
            "session_id": self.runtime.session.session_id,
            "cwd": str(self._engine.root),
            "mode": str(mode),
        }
        complete_payload.update(payload)
        result = await self._hook_engine.dispatch(event, complete_payload, self.runtime)
        self.runtime.append_reminders(result.injected_prompts)
        return result

    def _build_reminder(self, mode: Mode, iteration: int) -> str:
        parts: list[str] = []
        if mode is Mode.PLAN:
            full = iteration == 1 or (iteration - 1) % PLAN_REMINDER_INTERVAL == 0
            parts.append(plan_reminder(full=full))
        parts.extend(self.runtime.take_reminders())
        return "\n\n".join(part for part in parts if part)

    async def _pre_tool_result(self, call: ToolCall, mode: Mode) -> Result | None:
        if not self._is_tool_allowed(call.name):
            return _tool_not_allowed_result(call.name)
        outcome = await self._dispatch_hook(
            HookEvent.PRE_TOOL_USE,
            {
                "tool_name": call.name,
                "tool_input": _tool_input(call),
            },
            mode,
        )
        if not outcome.blocked:
            return None
        return _hook_blocked_result(outcome.blocking_hook_name, outcome.reason)

    async def _execute_read_batch_with_hooks(
        self,
        calls: Sequence[ToolCall],
        hook_results: Sequence[Result | None],
        mode: Mode,
        cancel_event: asyncio.Event | None,
        conversation: Conversation,
    ) -> tuple[list[Result], bool]:
        decisions: list[tuple[Decision, str]] = []
        for call, hook_result in zip(calls, hook_results, strict=True):
            if hook_result is None:
                decisions.append(
                    self._engine.check(mode, call, self._registry.is_read_only(call.name))
                )
            else:
                decisions.append((Decision.DENY, hook_result.content))
        results, cancelled = await self._execute_read_batch(
            calls, decisions, cancel_event, conversation
        )
        for index, hook_result in enumerate(hook_results):
            if hook_result is not None:
                results[index] = hook_result
        return results, cancelled

    def _update_usage_anchor(
        self,
        usage: TokenUsage | None,
        conversation: Conversation,
    ) -> None:
        if usage is not None:
            self.runtime.update_anchor(usage_anchor(usage), len(conversation.messages()))

    async def _record_read_results(
        self,
        calls: Sequence[ToolCall],
        results: Sequence[Result],
    ) -> None:
        if not self._compact_enabled:
            return
        for call, result in zip(calls, results, strict=True):
            if call.name != "read_file" or result.is_error:
                continue
            try:
                arguments = json.loads(call.arguments_json)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(arguments, dict):
                continue
            path = arguments.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            try:
                absolute_path = Path(path).resolve()
                data = await asyncio.to_thread(absolute_path.read_bytes)
            except OSError:
                continue
            self.runtime.recovery.record_file(
                str(absolute_path),
                data.decode("utf-8", errors="replace"),
            )

    async def run_force_compact(
        self,
        conversation: Conversation,
        tool_definitions: Sequence[ToolDefinition],
        *,
        mode: Mode = Mode.DEFAULT,
    ) -> tuple[int, int]:
        """Force one summary compaction for the TUI command path."""
        async with self._run_lock:
            await self._dispatch_hook(
                HookEvent.PRE_COMPACT,
                {"trigger": TriggerKind.MANUAL.value},
                mode,
            )
            output = await manage_context(
                self._manage_input(
                    conversation,
                    tuple(tool_definitions),
                    TriggerKind.MANUAL,
                )
            )
            if output.history_rewritten:
                self.runtime.reset_anchor()
            await self._dispatch_hook(
                HookEvent.POST_COMPACT,
                {
                    "trigger": TriggerKind.MANUAL.value,
                    "before_tokens": output.before_tokens,
                    "after_tokens": output.after_tokens,
                },
                mode,
            )
            return output.before_tokens, output.after_tokens

    def definitions_for_mode(self, mode: Mode) -> tuple[ToolDefinition, ...]:
        """Return the same immutable tool set used by the next Agent run."""
        definitions = (
            self._registry.read_only_definitions()
            if mode is Mode.PLAN
            else self._registry.definitions()
        )
        if self.allowed_tools is None:
            return definitions
        return tuple(
            definition for definition in definitions if definition.name in self.allowed_tools
        )

    def tool_count(self) -> int:
        """Return the number of tools exposed by this Agent."""
        return len(self.definitions_for_mode(Mode.DEFAULT))

    def _is_tool_allowed(self, name: str) -> bool:
        return self.allowed_tools is None or name in self.allowed_tools

    def _refresh_system_prompt(self, conversation: Conversation) -> None:
        if (
            self._memory_manager is None
            and not self._instruction_text
            and self._base_prompt is None
            and not self._skill_catalog
        ):
            return
        memory = ""
        if self._memory_manager is not None:
            try:
                memory = self._memory_manager.load_index()
            except Exception as error:
                logger.warning("Memory index refresh failed error=%s", type(error).__name__)
        conversation.replace_system_prompt(
            build_system_prompt(
                self._instruction_text,
                memory,
                self._base_prompt,
                self._skill_catalog,
            )
        )

    def _schedule_memory_update(self, conversation: Conversation) -> None:
        if self.subagent_kind != "main":
            return
        turn_count = self.runtime.increment_turn_count()
        manager = self._memory_manager
        if manager is None:
            return
        recent = _recent_turn(conversation.messages())
        if turn_count % 5 != 0 and not _has_memory_signal(recent):
            return
        task = asyncio.create_task(manager.update_async(recent))
        self._memory_tasks.add(task)
        task.add_done_callback(self._memory_update_finished)

    def _memory_update_finished(self, task: asyncio.Task[None]) -> None:
        self._memory_tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as error:
            logger.warning("Memory background task failed error=%s", type(error).__name__)

    async def shutdown_memory_updates(self, timeout: float = 1.0) -> None:
        """Give outstanding memory writes a bounded window, then cancel them."""
        tasks = tuple(self._memory_tasks)
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _execute_batch(
        self,
        calls: Sequence[ToolCall],
        cancel_event: asyncio.Event | None,
        conversation: Conversation,
    ) -> tuple[list[Result], bool]:
        agent_token = bind_execution_agent(self)
        context_token = bind_execution_context(ExecutionContext(self, conversation))
        try:
            return await self._execute_bound_batch(calls, cancel_event)
        finally:
            reset_execution_context(context_token)
            reset_execution_agent(agent_token)

    async def _execute_bound_batch(
        self,
        calls: Sequence[ToolCall],
        cancel_event: asyncio.Event | None,
    ) -> tuple[list[Result], bool]:
        tasks = [asyncio.create_task(self._execute_allowed_tool(call)) for call in calls]
        if cancel_event is None:
            return list(await asyncio.gather(*tasks)), False

        results: list[Result | None] = [None] * len(tasks)
        task_indexes = {task: index for index, task in enumerate(tasks)}
        pending = set(tasks)
        cancel_task = asyncio.create_task(cancel_event.wait())
        try:
            while pending:
                completed, _ = await asyncio.wait(
                    pending | {cancel_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for completed_task in completed:
                    if completed_task is cancel_task:
                        continue
                    tool_task = cast(asyncio.Task[Result], completed_task)
                    pending.remove(tool_task)
                    results[task_indexes[tool_task]] = tool_task.result()
                if cancel_task in completed:
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
                    for task in pending:
                        results[task_indexes[task]] = _cancelled_result()
                    return [result for result in results if result is not None], True
            return [result for result in results if result is not None], False
        except asyncio.CancelledError:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise
        finally:
            cancel_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancel_task

    async def _execute_allowed_tool(self, call: ToolCall) -> Result:
        if not self._is_tool_allowed(call.name):
            return _tool_not_allowed_result(call.name)
        return await self._registry.execute(call.name, call.arguments_json)

    async def _execute_read_batch(
        self,
        calls: Sequence[ToolCall],
        decisions: Sequence[tuple[Decision, str]],
        cancel_event: asyncio.Event | None,
        conversation: Conversation,
    ) -> tuple[list[Result], bool]:
        results: list[Result | None] = [None] * len(calls)
        allowed_calls: list[ToolCall] = []
        allowed_indexes: list[int] = []
        for index, ((decision, reason), call) in enumerate(zip(decisions, calls, strict=True)):
            if decision is Decision.ALLOW:
                allowed_calls.append(call)
                allowed_indexes.append(index)
            else:
                results[index] = _permission_denied_result(reason)

        cancelled = False
        if allowed_calls:
            executed, cancelled = await self._execute_batch(
                allowed_calls, cancel_event, conversation
            )
            for index, result in zip(allowed_indexes, executed, strict=True):
                results[index] = result
        if any(result is None for result in results):
            raise RuntimeError("permission batch result state is inconsistent")
        return [cast(Result, result) for result in results], cancelled

    def _all_unknown(self, calls: Sequence[ToolCall]) -> bool:
        return bool(calls) and all(self._registry.get(call.name) is None for call in calls)

    @staticmethod
    def _ensure_assistant_tail(conversation: Conversation, fallback: str) -> None:
        if conversation.last_role() is not MessageRole.ASSISTANT:
            conversation.add_assistant(fallback)

    @staticmethod
    def _to_tool_result(call: ToolCall, result: Result) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=redact_sensitive(result.content),
            is_error=result.is_error,
            error_code=result.error_code,
            truncated=result.truncated,
            metadata=_sanitize_metadata(result.metadata),
        )


def _recent_turn(messages: Sequence[Message]) -> list[Message]:
    """Return the latest user-led turn, including tool traffic and final reply."""
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role is MessageRole.USER:
            start = index
            break
    return list(messages[start:])


def _has_memory_signal(messages: Sequence[Message]) -> bool:
    """Detect an explicit request to retain information for later turns."""
    text = "\n".join(
        message.content for message in messages if message.role is MessageRole.USER
    ).casefold()
    return any(signal in text for signal in _MEMORY_SIGNALS)


async def _next_event(
    iterator: AsyncIterator[StreamEvent],
    cancel_event: asyncio.Event | None,
) -> tuple[StreamEvent | None, bool, bool]:
    if cancel_event is None:
        try:
            return await anext(iterator), False, False
        except StopAsyncIteration:
            return None, False, True
    if cancel_event.is_set():
        await _close_iterator(iterator)
        return None, True, False

    next_task: asyncio.Future[StreamEvent] = asyncio.ensure_future(anext(iterator))
    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        wait_set = {
            cast(asyncio.Future[object], next_task),
            cast(asyncio.Future[object], cancel_task),
        }
        completed, _ = await asyncio.wait(
            wait_set,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancel_task in completed:
            next_task.cancel()
            with suppress(asyncio.CancelledError, StopAsyncIteration):
                await next_task
            await _close_iterator(iterator)
            return None, True, False
        cancel_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_task
        try:
            return next_task.result(), False, False
        except StopAsyncIteration:
            return None, False, True
    except asyncio.CancelledError:
        next_task.cancel()
        with suppress(asyncio.CancelledError, StopAsyncIteration):
            await next_task
        await _close_iterator(iterator)
        raise
    finally:
        if not cancel_task.done():
            cancel_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancel_task


async def _await_approval(
    request: ApprovalRequest,
    cancel_event: asyncio.Event | None,
) -> tuple[Outcome | None, bool]:
    """Wait for one approval response while making cancellation leak-free."""
    if cancel_event is None:
        try:
            return await request.respond, False
        except asyncio.CancelledError:
            if not request.respond.done():
                request.respond.cancel()
            raise
    if cancel_event.is_set():
        request.respond.cancel()
        return None, True

    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        wait_set = {
            cast(asyncio.Future[object], request.respond),
            cast(asyncio.Future[object], cancel_task),
        }
        completed, _ = await asyncio.wait(
            wait_set,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if request.respond in completed:
            return request.respond.result(), False
        request.respond.cancel()
        return None, True
    except asyncio.CancelledError:
        if not request.respond.done():
            request.respond.cancel()
        raise
    finally:
        cancel_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_task


async def _await_approval_upgrader(
    upgrader: ApprovalUpgrader,
    request: ApprovalRequest,
    cancel_event: asyncio.Event | None,
) -> tuple[Outcome | None, bool]:
    """Await an external approval route without leaking work on cancellation."""
    upgrade_task: asyncio.Future[Outcome] = asyncio.ensure_future(upgrader(request))
    if cancel_event is None:
        try:
            outcome = await upgrade_task
            return outcome if isinstance(outcome, Outcome) else Outcome.DENY_ONCE, False
        except asyncio.CancelledError:
            upgrade_task.cancel()
            with suppress(asyncio.CancelledError):
                await upgrade_task
            raise
        finally:
            if not request.respond.done():
                request.respond.cancel()
    if cancel_event.is_set():
        upgrade_task.cancel()
        with suppress(asyncio.CancelledError):
            await upgrade_task
        if not request.respond.done():
            request.respond.cancel()
        return None, True

    cancel_task = asyncio.create_task(cancel_event.wait())
    try:
        wait_set = {
            cast(asyncio.Future[object], upgrade_task),
            cast(asyncio.Future[object], cancel_task),
        }
        completed, _ = await asyncio.wait(
            wait_set,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if upgrade_task in completed:
            outcome = upgrade_task.result()
            return outcome if isinstance(outcome, Outcome) else Outcome.DENY_ONCE, False
        upgrade_task.cancel()
        with suppress(asyncio.CancelledError):
            await upgrade_task
        return None, True
    except asyncio.CancelledError:
        upgrade_task.cancel()
        with suppress(asyncio.CancelledError):
            await upgrade_task
        raise
    finally:
        cancel_task.cancel()
        with suppress(asyncio.CancelledError):
            await cancel_task
        if not request.respond.done():
            request.respond.cancel()


async def _close_iterator(iterator: AsyncIterator[StreamEvent]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        await close()


def _tool_start(call: ToolCall) -> ToolEvent:
    return ToolEvent(
        call_id=call.id,
        name=call.name,
        arguments_json=call.arguments_json,
        phase=Phase.START,
        argument_summary=_argument_summary(call),
    )


def _tool_end(call: ToolCall, result: ToolResult) -> ToolEvent:
    summary, summary_truncated = _summarize(result.content)
    return ToolEvent(
        call_id=call.id,
        name=call.name,
        arguments_json=call.arguments_json,
        phase=Phase.END,
        summary=summary,
        is_error=result.is_error,
        truncated=result.truncated or summary_truncated,
        argument_summary=_argument_summary(call),
    )


def _cancelled_result() -> Result:
    return Result(NOTICE_CANCELLED, is_error=True, error_code="cancelled")


def _permission_denied_result(reason: str) -> Result:
    return Result(
        reason or "工具调用未获权限。",
        is_error=True,
        error_code="permission_denied",
    )


def _tool_not_allowed_result(name: str) -> Result:
    return Result(
        f"Tool is not allowed for this Agent: {name}",
        is_error=True,
        error_code="tool_not_allowed",
    )


def _hook_blocked_result(name: str, reason: str) -> Result:
    return Result(
        f"[hook {name}] {reason}",
        is_error=True,
        error_code="hook_blocked",
    )


def _tool_input(call: ToolCall) -> dict[str, object]:
    try:
        decoded = json.loads(call.arguments_json)
    except (json.JSONDecodeError, TypeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _last_user_prompt(conversation: Conversation) -> str:
    for message in reversed(conversation.messages()):
        if message.role is MessageRole.USER:
            return message.content
    return ""


def _cancelled_tool_result(call: ToolCall) -> ToolResult:
    return ToolResult(
        call.id,
        call.name,
        NOTICE_CANCELLED,
        is_error=True,
        error_code="cancelled",
    )


def _failed_tool_result(call: ToolCall) -> ToolResult:
    return ToolResult(
        call.id,
        call.name,
        NOTICE_STREAM_ERROR,
        is_error=True,
        error_code="agent_interrupted",
    )


def _is_cancelled(cancel_event: asyncio.Event | None) -> bool:
    return cancel_event is not None and cancel_event.is_set()


def _max_turns_notice(max_turns: int) -> str:
    if max_turns == MAX_ITERATIONS:
        return NOTICE_MAX_ITER
    return f"（已达最大迭代轮数 {max_turns}，自动停止；可继续发消息推进。）"


def _summarize(content: str) -> tuple[str, bool]:
    lines = redact_sensitive(content).splitlines() or ["(empty result)"]
    line_truncated = any(len(line) > MAX_SUMMARY_LINE_CHARS for line in lines)
    lines = [line[:MAX_SUMMARY_LINE_CHARS] for line in lines]
    if len(lines) > MAX_SUMMARY_LINES:
        return "\n".join(lines[: MAX_SUMMARY_LINES - 1] + ["[summary truncated]"]), True
    if line_truncated:
        lines[-1] += " [summary truncated]"
    return "\n".join(lines), line_truncated


def _sanitize_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    return {key: _sanitize_value(value) for key, value in metadata.items()}


def _sanitize_value(value: object) -> object:
    if isinstance(value, str):
        return redact_sensitive(value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_sanitize_value(item) for item in value)
    return value


def _argument_summary(call: ToolCall) -> str:
    key_by_tool = {
        "read_file": "path",
        "write_file": "path",
        "edit_file": "path",
        "bash": "command",
        "glob": "pattern",
        "grep": "pattern",
    }
    key = key_by_tool.get(call.name)
    if key is None:
        return ""
    try:
        arguments = json.loads(call.arguments_json)
    except json.JSONDecodeError:
        return ""
    if not isinstance(arguments, dict):
        return ""
    value = arguments.get(key)
    if not isinstance(value, str):
        return ""
    safe_value = redact_sensitive(value).replace("\n", " ")
    return safe_value[:120] + ("…" if len(safe_value) > 120 else "")


__all__ = [
    "EMPTY_RESPONSE_MESSAGE",
    "MAX_ITERATIONS",
    "MAX_SUMMARY_LINES",
    "MAX_UNKNOWN_RUN",
    "NOTICE_CANCELLED",
    "NOTICE_MAX_ITER",
    "NOTICE_STREAM_ERROR",
    "NOTICE_UNKNOWN_TOOLS",
    "Agent",
    "ApprovalUpgrader",
    "ApprovalRequest",
    "CompletionResult",
    "CompactEvent",
    "CompactPhase",
    "Event",
    "Mode",
    "MaxTurnsReached",
    "Phase",
    "SubagentKind",
    "ToolEvent",
]
