"""Main chat screen and interaction state machine."""

import asyncio
import inspect
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import cast

from textual import events, on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Input, Static
from textual.worker import Worker

from codewright.agent import (
    NOTICE_CANCELLED,
    Agent,
    ApprovalRequest,
    CompactEvent,
    CompactPhase,
    Event,
    Phase,
)
from codewright.command import Registry as CommandRegistry
from codewright.command import WorktreeAccessor, register_builtins
from codewright.compact import (
    SessionContext,
    new_session_context,
    open_session_context,
)
from codewright.compact.const import AUTO_SAFETY_MARGIN, SUMMARY_RESERVE
from codewright.compact.token import estimate_tokens
from codewright.conversation import Conversation
from codewright.hook import DispatchResult
from codewright.hook import Event as HookEvent
from codewright.hook import Rule as HookRule
from codewright.llm import LLMError, LLMResponseError, MessageRole, ToolDefinition
from codewright.memory import Manager
from codewright.permission import Mode
from codewright.session import SessionInfo, Writer, load_session
from codewright.skills import SkillDef, SkillExecutionError, SkillExecutor, SkillLoader
from codewright.task import BackgroundTask, SubagentApprovalBroker
from codewright.task import Manager as TaskManager
from codewright.tool import with_cwd
from codewright.tui.commands import dispatch_slash, format_compact_notice
from codewright.tui.complete import CompletionMenu
from codewright.tui.resume import ResumePanel, begin_resume
from codewright.tui.widgets.approval import ApprovalWidget
from codewright.tui.widgets.input import MessageInput
from codewright.tui.widgets.message import ConversationMessage
from codewright.tui.widgets.status import StatusWidget
from codewright.tui.widgets.tool import ToolCallWidget
from codewright.tui.worktree_adapter import WorktreeAdapter
from codewright.utils.timing import RequestTimer
from codewright.worktree import Manager as WorktreeManager

logger = logging.getLogger(__name__)


class ChatState(StrEnum):
    """Mutually exclusive states of the chat screen."""

    IDLE = "idle"
    RESUMING = "resuming"
    WAITING = "waiting"
    STREAMING = "streaming"
    APPROVING = "approving"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    EXITING = "exiting"


_ALLOWED_TRANSITIONS: dict[ChatState, frozenset[ChatState]] = {
    ChatState.IDLE: frozenset({ChatState.RESUMING, ChatState.WAITING, ChatState.EXITING}),
    ChatState.RESUMING: frozenset({ChatState.IDLE, ChatState.EXITING}),
    ChatState.WAITING: frozenset(
        {
            ChatState.STREAMING,
            ChatState.APPROVING,
            ChatState.COMPLETED,
            ChatState.ERROR,
            ChatState.CANCELLED,
            ChatState.EXITING,
        }
    ),
    ChatState.STREAMING: frozenset(
        {
            ChatState.APPROVING,
            ChatState.COMPLETED,
            ChatState.ERROR,
            ChatState.CANCELLED,
            ChatState.EXITING,
        }
    ),
    ChatState.APPROVING: frozenset(
        {ChatState.STREAMING, ChatState.ERROR, ChatState.CANCELLED, ChatState.EXITING}
    ),
    ChatState.COMPLETED: frozenset({ChatState.IDLE, ChatState.EXITING}),
    ChatState.ERROR: frozenset({ChatState.IDLE, ChatState.EXITING}),
    ChatState.CANCELLED: frozenset({ChatState.IDLE, ChatState.EXITING}),
    ChatState.EXITING: frozenset(),
}


class ChatScreen(Screen[None]):
    """Display one in-memory conversation and accept user messages."""

    BINDINGS = [
        Binding("ctrl+c", "exit_app", "Cancel/Exit", show=False, priority=True),
        Binding("escape", "cancel_turn", "Cancel", show=False, priority=True),
        Binding("shift+tab", "cycle_mode", "Permission mode", show=False, priority=True),
    ]

    def __init__(
        self,
        agent: Agent,
        conversation: Conversation,
        *,
        model_name: str,
        working_directory: Path,
        version: str,
        initial_mode: Mode = Mode.DEFAULT,
        stream: bool = True,
        writer: Writer | None = None,
        memory_manager: Manager | None = None,
        instruction_text: str = "",
        base_prompt: str | None = None,
        sessions_dir: str | None = None,
        command_registry: CommandRegistry | None = None,
        skill_loader: SkillLoader | None = None,
        skill_executor: SkillExecutor | None = None,
        skill_reloader: Callable[[], Awaitable[tuple[SkillDef, ...]]] | None = None,
        task_manager: TaskManager | None = None,
        approval_broker: SubagentApprovalBroker | None = None,
        worktree_manager: WorktreeManager | None = None,
        team_manager: object | None = None,
        coordinator_mode: bool = False,
    ) -> None:
        super().__init__()
        self._agent = agent
        self._conversation = conversation
        self._model_name = model_name
        self._working_directory = working_directory
        self._version = version
        self._stream = stream
        self._writer = writer
        self._memory_manager = memory_manager
        self._instruction_text = instruction_text
        self._base_prompt = base_prompt
        self._sessions_dir = sessions_dir
        self._skill_loader = skill_loader
        self._skill_executor = skill_executor
        self._skill_reloader = skill_reloader
        self._task_manager = task_manager
        self._approval_broker = approval_broker
        self._team_manager = team_manager
        self._coordinator_mode = coordinator_mode
        self._active_cwd = working_directory
        if worktree_manager is not None:
            session = worktree_manager.current_session()
            if session is not None:
                self._active_cwd = Path(session.worktree_path)
            self._worktree_adapter: WorktreeAdapter | None = WorktreeAdapter(
                worktree_manager, self._set_active_cwd
            )
        else:
            self._worktree_adapter = None
        self._done_consumer: asyncio.Task[None] | None = None
        self._approval_consumer: asyncio.Task[None] | None = None
        self._team_watcher: asyncio.Task[None] | None = None
        self._team_dispatcher: asyncio.Task[None] | None = None
        self._team_messages: asyncio.Queue[object] = asyncio.Queue()
        self._deferred_task_notifications: list[str] = []
        if command_registry is None:
            command_registry = CommandRegistry()
            register_builtins(command_registry)
        self._command_registry = command_registry
        self._completion = CompletionMenu()
        self._resume_panel: ResumePanel | None = None
        self._state = ChatState.IDLE
        self._state_history = [ChatState.IDLE]
        self._request_worker: Worker[None] | None = None
        self._tool_views: dict[str, ToolCallWidget] = {}
        if not isinstance(initial_mode, Mode):
            raise TypeError("initial_mode must be a Mode")
        self._mode = initial_mode
        self._iteration = 0
        self._cancel_event: asyncio.Event | None = None
        self._pending_approval: ApprovalRequest | None = None
        self._approval_widget: ApprovalWidget | None = None
        self._approve_cursor = 0

    @property
    def state(self) -> ChatState:
        """Return the current interaction state."""
        return self._state

    @property
    def state_history(self) -> tuple[ChatState, ...]:
        """Return state transitions for diagnostics and tests."""
        return tuple(self._state_history)

    @property
    def mode(self) -> Mode:
        """Return the persistent Agent mode."""
        return self._mode

    @property
    def pending_approval(self) -> ApprovalRequest | None:
        """Return the request currently awaiting a keyboard decision."""
        return self._pending_approval

    @property
    def approve_cursor(self) -> int:
        """Return the selected approval menu index."""
        return self._approve_cursor

    def compose(self) -> ComposeResult:
        """Compose the application information, history, status, and input areas."""
        info = (
            f"Codewright v{self._version}  |  "
            f"Model: {self._model_name}  |  "
            f"Working directory: {self._working_directory}"
        )
        with Vertical(id="chat-layout"):
            yield Static(info, id="app-info")
            yield VerticalScroll(id="conversation-view")
            yield StatusWidget(id="request-status")
            yield Static(id="command-completion")
            yield MessageInput(id="message-input")

    def on_mount(self) -> None:
        """Focus the input when the chat screen becomes active."""
        self.query_one(StatusWidget).set_mode(str(self._mode))
        self.query_one(StatusWidget).set_coordinator(self._coordinator_mode)
        self.query_one("#command-completion", Static).display = False
        self.query_one(MessageInput).focus()
        if self._task_manager is not None and self._done_consumer is None:
            self._done_consumer = asyncio.create_task(self._consume_task_done())
        if self._approval_broker is not None and self._approval_consumer is None:
            self._approval_consumer = asyncio.create_task(self._consume_subagent_approvals())
        if self._team_manager is not None and self._team_watcher is None:
            self._team_watcher = asyncio.create_task(self._watch_lead_mailbox())
            self._team_dispatcher = asyncio.create_task(self._dispatch_lead_mailbox())

    async def on_unmount(self) -> None:
        """Stop screen-owned queue consumers before application resources close."""
        await self.stop_subagent_consumers()

    async def stop_subagent_consumers(self) -> None:
        """Cancel and await both unique subagent presentation consumers."""
        consumers = tuple(
            task
            for task in (
                self._done_consumer,
                self._approval_consumer,
                self._team_watcher,
                self._team_dispatcher,
            )
            if task is not None
        )
        self._done_consumer = None
        self._approval_consumer = None
        self._team_watcher = None
        self._team_dispatcher = None
        for task in consumers:
            task.cancel()
        if consumers:
            await asyncio.gather(*consumers, return_exceptions=True)

    async def _consume_task_done(self) -> None:
        manager = self._task_manager
        if manager is None:
            return
        queue = manager.subscribe_done()
        while True:
            task = await queue.get()
            if task is None:
                return
            notification = _task_notification(task)
            if self._state is ChatState.IDLE:
                self._agent.runtime.append_reminders([notification])
            else:
                self._deferred_task_notifications.append(notification)

    async def _watch_lead_mailbox(self) -> None:
        manager = self._team_manager
        if manager is None:
            return
        while True:
            messages = await manager.poll_lead_mailboxes()
            for message in messages:
                await self._team_messages.put(message)
            await asyncio.sleep(0.5)

    async def _dispatch_lead_mailbox(self) -> None:
        while True:
            first = await self._team_messages.get()
            batch = [first]
            while not self._team_messages.empty():
                batch.append(self._team_messages.get_nowait())
            reminders = [
                f"Team message from {getattr(item, 'sender', 'unknown')}: "
                f"{getattr(item, 'content', '')}"
                for item in batch
            ]
            self._agent.runtime.append_reminders(reminders)
            while self._state is not ChatState.IDLE:
                await asyncio.sleep(0.05)
            await self._begin_autonomous_turn()

    async def _begin_autonomous_turn(self) -> None:
        """Start one reminder-driven turn without appending an empty user message."""
        if self._state is not ChatState.IDLE:
            return
        input_widget = self.query_one(MessageInput)
        input_widget.disabled = True
        self._transition(ChatState.WAITING)
        status = self.query_one(StatusWidget)
        self._iteration = 0
        status.set_iteration(0)
        status.show_waiting()
        self._cancel_event = asyncio.Event()
        self._request_worker = self.request_response(RequestTimer(), self._cancel_event)

    async def _consume_subagent_approvals(self) -> None:
        broker = self._approval_broker
        if broker is None:
            return
        queue = broker.subscribe()
        while True:
            envelope = await queue.get()
            if envelope is None:
                return
            try:
                await self._handle_background_approval(
                    envelope.request,
                    envelope.source,
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning(
                    "Subagent approval presentation failed error=%s",
                    type(error).__name__,
                )
                if not envelope.request.respond.done():
                    from codewright.permission import Outcome

                    envelope.request.respond.set_result(Outcome.DENY_ONCE)

    async def _handle_background_approval(
        self,
        request: ApprovalRequest,
        source: str,
    ) -> None:
        while self._pending_approval is not None:
            await asyncio.sleep(0.01)
        started_idle = self._state is ChatState.IDLE
        if started_idle:
            self._transition(ChatState.WAITING)
        self._pending_approval = request
        self._approve_cursor = 0
        widget = ApprovalWidget(request, source=source)
        self._approval_widget = widget
        await self._append_widget(widget)
        self._transition(ChatState.APPROVING)
        self.query_one(StatusWidget).show_approving(f"{source}: {request.name}")
        widget.focus()
        try:
            await request.respond
        finally:
            self._pending_approval = None
            await self._remove_approval_widget()
            if self.is_mounted:
                try:
                    if self._state is ChatState.APPROVING:
                        self._transition(ChatState.STREAMING)
                    if started_idle and self._state is ChatState.STREAMING:
                        self._transition(ChatState.COMPLETED)
                        self._transition(ChatState.IDLE)
                        self.query_one(StatusWidget).show_ready("SubAgent approval resolved")
                        self.query_one(MessageInput).focus()
                except NoMatches:
                    pass

    @on(Input.Changed, "#message-input")
    def update_command_completion(self, event: Input.Changed) -> None:
        """Synchronize slash-command candidates with the current input."""
        self._completion.update(event.value, self._command_registry)
        self._render_completion()

    @on(Input.Submitted, "#message-input")
    async def submit_message(self, event: Input.Submitted) -> None:
        """Accept a command or begin one asynchronous streaming request."""
        await self._submit_input(event.value)

    async def _submit_input(self, value: str) -> None:
        """Clear input state and route one submitted value."""
        input_widget = self.query_one(MessageInput)
        content = value.strip()
        if not content:
            input_widget.value = ""
            self._hide_completion()
            return
        if content.startswith("/"):
            input_widget.value = ""
            self._hide_completion()
            await dispatch_slash(content, self._command_registry, self)
            return

        result = await self._dispatch_tui_hook(
            HookEvent.USER_PROMPT_SUBMIT,
            {"prompt": content},
        )
        if result.blocked:
            await self._append_error(f"[hook {result.blocking_hook_name}] {result.reason}")
            input_widget.value = value
            input_widget.focus()
            return

        input_widget.value = ""
        self._hide_completion()
        self._agent.runtime.append_reminders(result.injected_prompts)
        await self._begin_turn(content, display_text=content)

    def _hide_completion(self) -> None:
        self._completion.hide()
        self._render_completion()

    def _render_completion(self) -> None:
        completion = self.query_one("#command-completion", Static)
        if not self._completion.active:
            completion.display = False
            completion.update("")
            return

        was_visible = completion.display
        completion.display = True
        self._render_completion_content()
        if not was_visible:
            self.call_after_refresh(self._render_completion_content)

    def _render_completion_content(self) -> None:
        """Render candidates after the visible menu has a layout width."""
        completion = self.query_one("#command-completion", Static)
        if self._completion.active and completion.display:
            input_widget = self.query_one(MessageInput)
            width = completion.size.width
            if width < 8:
                width = input_widget.content_size.width or self.size.width
            completion.update(self._completion.render(max(1, width)))

    async def _begin_turn(self, content: str, *, display_text: str) -> None:
        """Append one user message and start the Agent request worker."""
        if self._state is not ChatState.IDLE:
            return
        if self._deferred_task_notifications:
            self._agent.runtime.append_reminders(self._deferred_task_notifications)
            self._deferred_task_notifications.clear()
        input_widget = self.query_one(MessageInput)

        input_widget.disabled = True

        user_message = self._conversation.add_user(content)
        await self._append_message(ConversationMessage(user_message.role, display_text))
        self._transition(ChatState.WAITING)
        status = self.query_one(StatusWidget)
        self._iteration = 0
        status.set_iteration(0)
        status.show_waiting()
        self._cancel_event = asyncio.Event()
        self._request_worker = self.request_response(RequestTimer(), self._cancel_event)

    async def println(self, message: str) -> None:
        """Append one informational command result to scrollback."""
        await self._append_notice(message)

    async def error(self, message: str) -> None:
        """Append one visibly distinct safe command error."""
        await self._append_error(message)

    def idle(self) -> bool:
        """Return whether state-changing commands may run."""
        return self._state is ChatState.IDLE

    async def set_mode(self, mode: Mode) -> None:
        """Persist a permission mode and update the status badge."""
        if not isinstance(mode, Mode):
            raise TypeError("mode must be a Mode")
        self._mode = mode
        status = self.query_one(StatusWidget)
        status.set_mode(str(self._mode))
        status.show_ready("Permission mode changed")

    def usage(self) -> tuple[int, int]:
        status = self.query_one(StatusWidget)
        return status.input_tokens, status.output_tokens

    def model_name(self) -> str:
        return self._model_name

    def cwd(self) -> str:
        return str(self._active_cwd)

    def team_accessor(self):
        return self._team_manager

    def _set_active_cwd(self, path: str) -> None:
        self._active_cwd = Path(path)

    def worktree_accessor(self) -> WorktreeAccessor | None:
        return self._worktree_adapter

    def tool_count(self) -> int:
        return self._agent.tool_count()

    def hook_sources(self) -> list[str]:
        getter = getattr(self.app, "hook_sources", None)
        return getter() if callable(getter) else []

    def hook_rules(self) -> list[HookRule]:
        getter = getattr(self.app, "hook_rules", None)
        return getter() if callable(getter) else []

    async def _dispatch_tui_hook(
        self,
        event: HookEvent,
        payload: dict[str, object],
    ) -> DispatchResult:
        dispatcher = getattr(self.app, "dispatch_hook", None)
        if not callable(dispatcher):
            return DispatchResult()
        return cast(DispatchResult, await dispatcher(event, payload, self._mode))

    async def _dispatch_session_start(self) -> None:
        dispatcher = getattr(self.app, "dispatch_session_start", None)
        if callable(dispatcher):
            await dispatcher(self._mode)

    async def _dispatch_session_resume(self) -> None:
        dispatcher = getattr(self.app, "dispatch_session_resume", None)
        if callable(dispatcher):
            await dispatcher(self._mode)

    async def _dispatch_session_end(self) -> None:
        dispatcher = getattr(self.app, "dispatch_session_end", None)
        if callable(dispatcher):
            await dispatcher(self._mode)

    def memory_files(self) -> tuple[list[str], list[str]]:
        if self._memory_manager is None:
            return [], []
        return self._memory_manager.list_files()

    def session_id(self) -> str:
        return self._agent.runtime.session.session_id

    def session_path(self) -> str:
        return str(self._writer.path) if self._writer is not None else ""

    async def inject_and_send(self, display_label: str, preset_prompt: str) -> None:
        await self._begin_turn(preset_prompt, display_text=display_label)

    def list_skills(self) -> tuple[SkillDef, ...]:
        """Return the current immutable Skill catalog snapshot."""
        return self._skill_loader.list() if self._skill_loader is not None else ()

    def get_skill(self, name: str) -> SkillDef | None:
        """Return the latest valid version of one Skill."""
        return self._skill_loader.get(name) if self._skill_loader is not None else None

    async def reload_skills(self) -> tuple[SkillDef, ...]:
        """Reload Skills and synchronize commands plus prompt catalog."""
        if self._skill_reloader is None:
            return ()
        return await self._skill_reloader()

    async def run_inline_skill(self, name: str, args: str) -> None:
        """Hot-reload and submit an inline Skill through the normal user path."""
        executor = self._skill_executor
        skill = self.get_skill(name)
        if executor is None or skill is None:
            await self._append_error(f"未知 Skill: {name}")
            return
        try:
            rendered = executor.execute_inline(skill, args)
        except SkillExecutionError as error:
            await self._append_error(error.safe_message)
            return
        label = f"/{skill.name}" + (f" {args}" if args else "")
        await self.inject_and_send(label, rendered)

    async def run_fork_skill(self, name: str, args: str) -> None:
        """Start one isolated fork Skill as an owned Textual worker."""
        executor = self._skill_executor
        skill = self.get_skill(name)
        if executor is None or skill is None:
            await self._append_error(f"未知 Skill: {name}")
            return
        input_widget = self.query_one(MessageInput)
        input_widget.disabled = True
        self._transition(ChatState.WAITING)
        self.query_one(StatusWidget).show_waiting()
        self._cancel_event = asyncio.Event()
        self._request_worker = self._execute_fork_skill(skill, args, self._cancel_event)

    async def request_exit(self) -> None:
        await self._dispatch_session_end()
        self.action_exit_app()

    async def force_compact(self) -> None:
        input_widget = self.query_one(MessageInput)
        input_widget.disabled = True
        self._transition(ChatState.WAITING)
        self.query_one(StatusWidget).show_waiting()
        notice = format_compact_notice(CompactEvent(CompactPhase.BEFORE_MANUAL))
        await self._append_notice(notice)
        self._request_worker = self._run_force_compact()

    async def open_resume_menu(self) -> None:
        if self._state is not ChatState.IDLE:
            await self._append_notice("请等待当前任务完成。")
            return
        if self._sessions_dir is None or self._writer is None:
            await self._append_notice("会话恢复尚未在当前启动方式中启用。")
            return
        try:
            sessions = await begin_resume(self._sessions_dir)
        except Exception as error:
            logger.warning("Session list failed error=%s", type(error).__name__)
            await self._append_notice("无法读取历史会话列表。")
            return
        if not sessions:
            await self._append_notice("没有可恢复的历史会话。")
            return
        self._transition(ChatState.RESUMING)
        panel = ResumePanel(sessions)
        self._resume_panel = panel
        layout = self.query_one("#chat-layout", Vertical)
        await layout.mount(panel, before=self.query_one(StatusWidget))

    async def clear_and_new_session(self) -> None:
        """Transactionally replace the current conversation and durable session."""
        if self._state is not ChatState.IDLE:
            await self._append_error("当前任务正在运行，请等待完成后再执行该命令。")
            return

        candidate_writer: Writer | None = None
        try:
            context, candidate_writer = self._create_session_writer()
            system_prompt = self._conversation.messages()[0].content
            candidate_conversation = Conversation(
                system_prompt,
                on_append=candidate_writer.on_append,
                on_replace=candidate_writer.on_replace,
            )
        except Exception as error:
            if candidate_writer is not None:
                candidate_writer.close()
            logger.warning("New session creation failed error=%s", type(error).__name__)
            await self._append_error("无法创建新会话，当前会话未改变。")
            return

        old_writer = self._writer
        await self._dispatch_session_end()
        self._conversation = candidate_conversation
        self._writer = candidate_writer
        self._agent.runtime.reset_for_new_session(context)
        setter = getattr(self.app, "set_current_writer", None)
        if callable(setter):
            setter(candidate_writer)
        await self._dispatch_session_start()

        history = self.query_one("#conversation-view", VerticalScroll)
        await history.remove_children()
        self._tool_views.clear()
        status = self.query_one(StatusWidget)
        status.reset_usage()
        status.show_ready("New session")
        self.query_one(MessageInput).focus()

        if old_writer is not None and old_writer is not candidate_writer:
            try:
                old_writer.close()
            except Exception as error:
                logger.warning(
                    "Previous session writer close failed error=%s", type(error).__name__
                )

    def _create_session_writer(self) -> tuple[SessionContext, Writer]:
        """Allocate a collision-free candidate session with bounded retries."""
        for _ in range(3):
            context = new_session_context(str(self._working_directory))
            try:
                return context, Writer(context.session_dir, self._model_name)
            except FileExistsError:
                continue
        raise RuntimeError("could not allocate a unique session directory")

    @on(ResumePanel.Selected)
    async def select_resume_session(self, event: ResumePanel.Selected) -> None:
        if self._state is not ChatState.RESUMING:
            return
        await self._do_resume_session(event.info)

    async def on_key(self, event: events.Key) -> None:
        if self._state is ChatState.RESUMING:
            panel = self._resume_panel
            if event.key == "escape":
                event.prevent_default()
                event.stop()
                await self._close_resume_panel()
                return
            if panel is not None and event.key in {"up", "down", "enter"}:
                event.prevent_default()
                event.stop()
                if event.key == "up":
                    panel.move_up()
                elif event.key == "down":
                    panel.move_down()
                else:
                    panel.select_highlighted()
            return

        if self._state is not ChatState.IDLE or not self._completion.active:
            return
        if event.key in {"up", "down"}:
            event.prevent_default()
            event.stop()
            if event.key == "up":
                self._completion.move_up()
            else:
                self._completion.move_down()
            self._render_completion()
            return
        if event.key == "escape":
            event.prevent_default()
            event.stop()
            self._hide_completion()
            self.query_one(MessageInput).focus()
            return
        if event.key == "tab":
            event.prevent_default()
            event.stop()
            selected = self._completion.selected()
            if selected is None:
                self._hide_completion()
                return
            value = f"/{selected.name}"
            self.query_one(MessageInput).value = value
            await self._submit_input(value)
            return
        if event.key == "enter" and (selected := self._completion.selected()) is not None:
            event.prevent_default()
            event.stop()
            value = f"/{selected.name}"
            self.query_one(MessageInput).value = value
            await self._submit_input(value)

    async def _do_resume_session(self, info: SessionInfo) -> None:
        old_conversation = self._conversation
        old_writer = self._writer
        old_session = self._agent.runtime.session
        old_replacement = self._agent.runtime.replacement
        old_recovery = self._agent.runtime.recovery
        old_tracking = self._agent.runtime.auto_tracking
        old_active = self._agent.runtime.active_skills
        old_anchor = self._agent.runtime.anchor_snapshot()
        old_turn_count = self._agent.runtime.turn_count
        old_reminders = list(self._agent.runtime.pending_reminders)
        old_hook_once = set(self._agent.runtime.hook_once_fired)
        old_session_end = self._agent.runtime.session_end_emitted
        new_writer: Writer | None = None
        committed = False
        try:
            loaded = await asyncio.to_thread(load_session, info.dir)
            new_context = open_session_context(str(self._working_directory), info.id)
            new_writer = Writer.open_existing(info.dir, self._model_name)
            current_system = old_conversation.messages()[0].content
            candidate = Conversation.from_messages(
                current_system,
                loaded.messages,
                on_append=new_writer.on_append,
                on_replace=new_writer.on_replace,
            )
            self._agent.runtime.reset_for_new_session(new_context)
            estimated = estimate_tokens(0, candidate.messages(), 0)
            threshold = self._agent.runtime.context_window - SUMMARY_RESERVE - AUTO_SAFETY_MARGIN
            if estimated >= threshold:
                definitions = self._agent.definitions_for_mode(self._mode)
                await self._force_compact_agent(candidate, definitions)
            if loaded.last_message_ts is not None:
                elapsed = max(0, int(time.time()) - loaded.last_message_ts)
                if elapsed > 6 * 3600:
                    candidate.add_user(
                        "[系统提示] 本会话已暂停 "
                        f"{_format_pause_duration(elapsed)}。部分上下文可能已过时，"
                        "如需最新信息请重新读取相关文件。"
                    )

            self._agent.runtime.session = old_session
            self._agent.runtime.replacement = old_replacement
            self._agent.runtime.recovery = old_recovery
            self._agent.runtime.auto_tracking = old_tracking
            self._agent.runtime.active_skills = old_active
            self._agent.runtime.update_anchor(old_anchor[0], old_anchor[1])
            self._agent.runtime.turn_count = old_turn_count
            self._agent.runtime.pending_reminders = old_reminders
            self._agent.runtime.hook_once_fired = old_hook_once
            self._agent.runtime.session_end_emitted = old_session_end
            await self._dispatch_session_end()

            self._conversation = candidate
            self._writer = new_writer
            self._agent.runtime.reset_for_new_session(new_context)
            setter = getattr(self.app, "set_current_writer", None)
            if callable(setter):
                setter(new_writer)
            committed = True
            await self._dispatch_session_resume()
            if old_writer is not None and old_writer is not new_writer:
                try:
                    old_writer.close()
                except Exception as error:
                    logger.warning(
                        "Previous session writer close failed error=%s",
                        type(error).__name__,
                    )
            new_writer = None
            await self._close_resume_panel()
            await self._append_notice(
                f"已恢复会话 {info.id}，共 {len(candidate.messages()) - 1} 条消息"
            )
        except Exception as error:
            logger.warning("Session resume failed id=%s error=%s", info.id, type(error).__name__)
            if committed:
                if self._state is ChatState.RESUMING:
                    await self._close_resume_panel()
                await self._append_error("会话已恢复，但界面刷新未完整完成。")
                return
            if new_writer is not None:
                new_writer.close()
            self._conversation = old_conversation
            self._writer = old_writer
            self._agent.runtime.session = old_session
            self._agent.runtime.replacement = old_replacement
            self._agent.runtime.recovery = old_recovery
            self._agent.runtime.auto_tracking = old_tracking
            self._agent.runtime.active_skills = old_active
            self._agent.runtime.update_anchor(old_anchor[0], old_anchor[1])
            self._agent.runtime.turn_count = old_turn_count
            self._agent.runtime.pending_reminders = old_reminders
            self._agent.runtime.hook_once_fired = old_hook_once
            self._agent.runtime.session_end_emitted = old_session_end
            await self._close_resume_panel()
            await self._append_error("恢复会话失败，当前会话未改变。")

    async def _close_resume_panel(self) -> None:
        panel = self._resume_panel
        self._resume_panel = None
        if panel is not None and panel.parent is not None:
            await panel.remove()
        if self._state is ChatState.RESUMING:
            self._transition(ChatState.IDLE)
        self.query_one(MessageInput).focus()

    async def _force_compact_agent(
        self,
        conversation: Conversation,
        definitions: Sequence[ToolDefinition],
    ) -> tuple[int, int]:
        """Call the Agent API while tolerating legacy test doubles."""
        compact = self._agent.run_force_compact
        if "mode" in inspect.signature(compact).parameters:
            return await compact(conversation, definitions, mode=self._mode)
        return await compact(conversation, definitions)

    async def _events_with_cwd(self, cancel_event: asyncio.Event) -> AsyncIterator[Event]:
        with with_cwd(self._active_cwd):
            async for event in self._agent.run(
                self._conversation,
                stream=self._stream,
                mode=self._mode,
                cancel_event=cancel_event,
            ):
                yield event

    @work(exclusive=True, group="provider-request")
    async def _run_force_compact(self) -> None:
        """Run manual compaction without adding a conversation command message."""
        try:
            definitions = self._agent.definitions_for_mode(self._mode)
            before, after = await self._force_compact_agent(self._conversation, definitions)
            await self._append_notice(
                format_compact_notice(
                    CompactEvent(CompactPhase.AFTER_MANUAL, before=before, after=after)
                )
            )
            self._transition(ChatState.COMPLETED)
            self.query_one(StatusWidget).show_complete(0.0)
        except LLMError as error:
            await self._append_notice(
                format_compact_notice(
                    CompactEvent(
                        CompactPhase.AFTER_MANUAL,
                        error_message=error.safe_message,
                    )
                )
            )
            self._transition(ChatState.ERROR)
            self.query_one(StatusWidget).show_error(error.safe_message)
        except Exception as error:
            logger.error("Manual compact failed error=%s", type(error).__name__)
            safe_message = "上下文压缩失败。"
            await self._append_notice(
                format_compact_notice(
                    CompactEvent(CompactPhase.AFTER_MANUAL, error_message=safe_message)
                )
            )
            self._transition(ChatState.ERROR)
            self.query_one(StatusWidget).show_error(safe_message)
        finally:
            if self._state in {ChatState.COMPLETED, ChatState.ERROR}:
                self._transition(ChatState.IDLE)
            input_widget = self.query_one(MessageInput)
            input_widget.disabled = False
            input_widget.focus()
            self._request_worker = None

    @work(exclusive=True, group="provider-request")
    async def _execute_fork_skill(
        self,
        skill: SkillDef,
        args: str,
        cancel_event: asyncio.Event,
    ) -> None:
        """Run one fork Skill and commit only its completed assistant result."""
        timer = RequestTimer()
        try:
            executor = self._skill_executor
            if executor is None:
                raise SkillExecutionError("Skill execution is not initialized.")
            result = await executor.execute_fork(
                skill,
                args,
                self._conversation,
                cancel_event,
                approval_handler=lambda request: self._handle_fork_approval(
                    request,
                    cancel_event,
                ),
            )
            if cancel_event.is_set():
                raise SkillExecutionError("Skill execution was cancelled.")
            assistant = self._conversation.add_assistant(result.text)
            await self._append_message(ConversationMessage(assistant.role, assistant.content))
            self.query_one(StatusWidget).add_usage(result.usage)
            self._transition(ChatState.COMPLETED)
            self.query_one(StatusWidget).show_complete(timer.elapsed_seconds)
        except asyncio.CancelledError:
            self._transition(ChatState.CANCELLED)
            self.query_one(StatusWidget).show_cancelled(timer.elapsed_seconds)
        except SkillExecutionError as error:
            if cancel_event.is_set():
                self._transition(ChatState.CANCELLED)
                self.query_one(StatusWidget).show_cancelled(timer.elapsed_seconds)
            else:
                self._transition(ChatState.ERROR)
                await self._append_error(error.safe_message)
                self.query_one(StatusWidget).show_error(error.safe_message)
        except Exception as error:
            logger.error("Fork Skill failed error=%s", type(error).__name__)
            safe_message = "Skill execution failed."
            self._transition(ChatState.ERROR)
            await self._append_error(safe_message)
            self.query_one(StatusWidget).show_error(safe_message)
        finally:
            if self._state in {ChatState.COMPLETED, ChatState.ERROR, ChatState.CANCELLED}:
                self._transition(ChatState.IDLE)
                input_widget = self.query_one(MessageInput)
                input_widget.disabled = False
                input_widget.focus()
            self._request_worker = None
            if self._cancel_event is cancel_event:
                self._cancel_event = None

    async def _handle_fork_approval(
        self,
        request: ApprovalRequest,
        cancel_event: asyncio.Event,
    ) -> None:
        """Bridge a child Agent approval into the existing keyboard UI."""
        self._pending_approval = request
        self._approve_cursor = 0
        widget = ApprovalWidget(request)
        self._approval_widget = widget
        await self._append_widget(widget)
        self._transition(ChatState.APPROVING)
        self.query_one(StatusWidget).show_approving(request.name)
        widget.focus()
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
            if cancel_task in completed and not request.respond.done():
                request.respond.cancel()
                raise SkillExecutionError("Skill execution was cancelled.")
            request.respond.result()
            self._pending_approval = None
            if self._state is ChatState.APPROVING:
                self._transition(ChatState.STREAMING)
            self.query_one(StatusWidget).show_tool(request.name)
        finally:
            cancel_task.cancel()
            await asyncio.gather(cancel_task, return_exceptions=True)
            await self._remove_approval_widget()

    @work(exclusive=True, group="provider-request")
    async def request_response(
        self,
        timer: RequestTimer,
        cancel_event: asyncio.Event,
    ) -> None:
        """Consume Agent events without blocking Textual's event loop."""
        assistant_view: ConversationMessage | None = None
        completed = False
        cancelled = False
        try:
            async for event in self._events_with_cwd(cancel_event):
                if event.error is not None:
                    raise event.error
                if event.compact is not None:
                    await self._append_notice(format_compact_notice(event.compact))
                    continue
                if event.usage is not None:
                    self.query_one(StatusWidget).add_usage(event.usage)
                    continue
                if event.iteration:
                    self._iteration = event.iteration
                    status = self.query_one(StatusWidget)
                    status.set_iteration(event.iteration)
                    status.show_waiting()
                    continue
                if event.notice:
                    await self._append_notice(event.notice)
                    cancelled = cancelled or event.notice == NOTICE_CANCELLED
                    continue
                if event.approval is not None:
                    if self._state not in {ChatState.WAITING, ChatState.STREAMING}:
                        raise LLMResponseError()
                    self._pending_approval = event.approval
                    self._approve_cursor = 0
                    approval_widget = ApprovalWidget(event.approval)
                    self._approval_widget = approval_widget
                    await self._append_widget(approval_widget)
                    self._transition(ChatState.APPROVING)
                    self.query_one(StatusWidget).show_approving(event.approval.name)
                    approval_widget.focus()
                    continue
                if event.done:
                    completed = True
                    break
                if event.text:
                    if self._state is ChatState.WAITING and self._stream:
                        self._transition(ChatState.STREAMING)
                    self.query_one(StatusWidget).show_streaming()
                    if assistant_view is None:
                        assistant_view = ConversationMessage(
                            MessageRole.ASSISTANT, "", streaming=True
                        )
                        await self._append_message(assistant_view)
                    assistant_view.append_delta(event.text)
                    continue
                if event.tool is not None:
                    if self._state is ChatState.WAITING:
                        self._transition(ChatState.STREAMING)
                    if event.tool.phase is Phase.START:
                        if assistant_view is not None:
                            assistant_view.finalize()
                            assistant_view = None
                        tool_view = ToolCallWidget(event.tool)
                        self._tool_views[event.tool.call_id] = tool_view
                        await self._append_widget(tool_view)
                        self.query_one(StatusWidget).show_tool(event.tool.name)
                    else:
                        existing_tool_view = self._tool_views.get(event.tool.call_id)
                        if existing_tool_view is None:
                            raise LLMResponseError()
                        existing_tool_view.complete(event.tool)

            if not completed:
                raise LLMResponseError()

            if assistant_view is not None:
                if cancelled:
                    assistant_view.mark_incomplete("Response cancelled")
                else:
                    assistant_view.finalize()
            self._iteration = 0
            status = self.query_one(StatusWidget)
            status.set_iteration(0)
            if cancelled:
                self._transition(ChatState.CANCELLED)
                status.show_cancelled(timer.elapsed_seconds)
            else:
                self._transition(ChatState.COMPLETED)
                status.show_complete(timer.elapsed_seconds)
        except asyncio.CancelledError:
            if self._state in {
                ChatState.WAITING,
                ChatState.STREAMING,
                ChatState.APPROVING,
            }:
                self._transition(ChatState.CANCELLED)
                if assistant_view is not None:
                    assistant_view.mark_incomplete("Response cancelled")
                self.query_one(StatusWidget).show_cancelled(timer.elapsed_seconds)
        except LLMError as error:
            self._transition(ChatState.ERROR)
            if assistant_view is not None:
                assistant_view.mark_incomplete("Response interrupted")
            await self._append_error(error.safe_message)
            self.query_one(StatusWidget).show_error(error.safe_message)
        except Exception as error:
            safe_message = "An unexpected error interrupted the response."
            logger.error("Unexpected Provider failure error=%s", type(error).__name__)
            self._transition(ChatState.ERROR)
            if assistant_view is not None:
                assistant_view.mark_incomplete("Response interrupted")
            await self._append_error(safe_message)
            self.query_one(StatusWidget).show_error(safe_message)
        finally:
            await self._clear_approval()
            if self._state in {ChatState.COMPLETED, ChatState.ERROR, ChatState.CANCELLED}:
                self._transition(ChatState.IDLE)
                input_widget = self.query_one(MessageInput)
                input_widget.disabled = False
                input_widget.focus()
            self._request_worker = None
            if self._cancel_event is cancel_event:
                self._cancel_event = None

    def action_exit_app(self) -> None:
        """Cancel active generation, or exit when the screen is idle."""
        if self._state in {
            ChatState.WAITING,
            ChatState.STREAMING,
            ChatState.APPROVING,
        }:
            self._request_cancel()
            return
        if self._state is not ChatState.EXITING:
            self._transition(ChatState.EXITING)
        self.app.exit()

    def action_cancel_turn(self) -> None:
        """Cancel an active turn without exiting the application."""
        if self._state is ChatState.RESUMING:
            self.call_after_refresh(self._close_resume_panel)
            return
        if self._state is ChatState.IDLE and self._completion.active:
            self._hide_completion()
            self.query_one(MessageInput).focus()
            return
        if self._state in {
            ChatState.WAITING,
            ChatState.STREAMING,
            ChatState.APPROVING,
        }:
            self._request_cancel()

    async def action_cycle_mode(self) -> None:
        """Cycle permission modes only while no turn is active."""
        if self._state is not ChatState.IDLE:
            return
        modes = (Mode.DEFAULT, Mode.ACCEPT_EDITS, Mode.PLAN, Mode.BYPASS)
        self._mode = modes[(modes.index(self._mode) + 1) % len(modes)]
        status = self.query_one(StatusWidget)
        status.set_mode(str(self._mode))
        status.show_ready("Permission mode changed")
        await self._append_notice(f"权限模式已切换为 {str(self._mode)}。")

    @on(ApprovalWidget.CursorChanged)
    def update_approval_cursor(self, event: ApprovalWidget.CursorChanged) -> None:
        """Keep screen diagnostics synchronized with the focused menu."""
        self._approve_cursor = event.cursor

    @on(ApprovalWidget.Selected)
    async def submit_approval(self, event: ApprovalWidget.Selected) -> None:
        """Resolve a pending approval Future at most once."""
        request = self._pending_approval
        if self._state is not ChatState.APPROVING or request is None or request.respond.done():
            return
        request.respond.set_result(event.outcome)
        self._pending_approval = None
        self._transition(ChatState.STREAMING)
        self.query_one(StatusWidget).show_tool(request.name)
        await self._remove_approval_widget()

    def _request_cancel(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.query_one(StatusWidget).show_cancelling()
        elif self._request_worker is not None:
            self._request_worker.cancel()

    async def _append_message(self, message: ConversationMessage) -> None:
        await self._append_widget(message)

    async def _append_widget(self, widget: Static) -> None:
        history = self.query_one("#conversation-view", VerticalScroll)
        await history.mount(widget)
        history.scroll_end(animate=False)

    async def _append_error(self, message: str) -> None:
        history = self.query_one("#conversation-view", VerticalScroll)
        await history.mount(Static(f"Error: {message}", classes="error-message", markup=False))
        history.scroll_end(animate=False)

    async def _append_notice(self, message: str) -> None:
        history = self.query_one("#conversation-view", VerticalScroll)
        await history.mount(Static(message, classes="notice-message", markup=False))
        history.scroll_end(animate=False)

    async def _remove_approval_widget(self) -> None:
        widget = self._approval_widget
        self._approval_widget = None
        self._approve_cursor = 0
        if widget is not None and widget.parent is not None:
            await widget.remove()

    async def _clear_approval(self) -> None:
        self._pending_approval = None
        await self._remove_approval_widget()

    def _transition(self, target: ChatState) -> None:
        allowed: Iterable[ChatState] = _ALLOWED_TRANSITIONS[self._state]
        if target not in allowed:
            raise RuntimeError(f"Invalid chat state transition: {self._state} -> {target}")
        self._state = target
        self._state_history.append(target)


def _format_pause_duration(seconds: int) -> str:
    hours = max(1, seconds // 3600)
    if hours < 24:
        return f"约 {hours} 小时"
    return f"约 {hours // 24} 天"


def _task_notification(task: BackgroundTask) -> str:
    payload = {
        "task_id": task.id,
        "name": task.name,
        "description": task.description,
        "status": task.status.value,
        "result": task.result,
        "error_type": task.error_type,
        "error_message": task.error_message,
        "generation": task.notification_generation,
    }
    return (
        "<task-notification>\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + "\n</task-notification>"
    )
