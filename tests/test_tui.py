"""Behavior tests for streaming Textual interaction."""

import asyncio
import json
import logging
import re
import shlex
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import httpx
import pytest
from rich.text import Text
from textual.containers import VerticalScroll
from textual.widgets import Static

from codewright.agent import (
    NOTICE_CANCELLED,
    NOTICE_STREAM_ERROR,
    ApprovalRequest,
    CompactEvent,
    CompactPhase,
    Event,
)
from codewright.command import Command, Kind
from codewright.command import Registry as CommandRegistry
from codewright.command.builtin_prompt import REVIEW_DIRECTIVE
from codewright.compact import new_session_context
from codewright.conversation import Conversation
from codewright.hook import Action as HookAction
from codewright.hook import ActionType as HookActionType
from codewright.hook import Engine as HookEngine
from codewright.hook import Event as HookEvent
from codewright.hook import Rule as HookRule
from codewright.hook import ShellAction as HookShellAction
from codewright.llm import (
    ChatResult,
    LLMServiceError,
    Message,
    MessageRole,
    RequestContext,
    RequestParameters,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)
from codewright.permission import Engine, Mode, Outcome
from codewright.permission.rule import RuleSet
from codewright.prompt import EXECUTE_DIRECTIVE, PLAN_MODE_REMINDER, SYSTEM_PROMPT, plan_reminder
from codewright.session import Writer, list_sessions
from codewright.skills import SkillInstaller, SkillLoader
from codewright.subagent import Catalog
from codewright.task import (
    BackgroundTask,
    Status,
    SubagentApprovalBroker,
)
from codewright.task import (
    Manager as TaskManager,
)
from codewright.tool import (
    InstallSkillTool,
    LoadSkillTool,
    Registry,
    Result,
    new_default_registry,
)
from codewright.tui import ChatScreen, ChatState, CodewrightApp
from codewright.tui.commands import dispatch_slash, format_compact_notice
from codewright.tui.widgets.approval import ApprovalWidget
from codewright.tui.widgets.input import MessageInput
from codewright.tui.widgets.message import ConversationMessage
from codewright.tui.widgets.status import StatusWidget
from codewright.utils.logging import register_secrets


class FakeProvider:
    """Controllable offline stream provider used by TUI behavior tests."""

    def __init__(
        self,
        responses: list[tuple[str, ...]] | None = None,
        *,
        error_requests: set[int] | None = None,
        pause_after_first_delta: bool = False,
        usage: TokenUsage | None = None,
    ) -> None:
        self.responses = responses or [("# Answer\n\n", "- item")]
        self.error_requests = error_requests or set()
        self.pause_after_first_delta = pause_after_first_delta
        self.usage = usage
        self.requests: list[tuple[Message, ...]] = []
        self.tool_definitions: list[tuple[ToolDefinition, ...]] = []
        self.request_contexts: list[RequestContext | None] = []
        self.first_delta = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.chat_calls = 0

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del messages, parameters, tools, request_context
        self.chat_calls += 1
        raise AssertionError("T9 must use stream_chat")

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters
        request_index = len(self.requests)
        self.requests.append(tuple(messages))
        self.tool_definitions.append(tuple(tools))
        self.request_contexts.append(request_context)
        response = self.responses[min(request_index, len(self.responses) - 1)]

        for delta_index, text in enumerate(response):
            yield StreamEvent.delta(text)
            self.first_delta.set()
            if self.pause_after_first_delta and delta_index == 0:
                await self.release.wait()

        if request_index in self.error_requests:
            yield StreamEvent.failed(LLMServiceError())
        else:
            if self.usage is not None:
                yield StreamEvent.usage_report(self.usage)
            yield StreamEvent.completed()

    async def close(self) -> None:
        self.closed = True


class UnexpectedFailureProvider(FakeProvider):
    """Provider that violates the contract to exercise the TUI safety boundary."""

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del messages, parameters, tools, request_context
        raise RuntimeError("unsafe provider detail")
        yield StreamEvent.completed()


class ApprovalProvider(FakeProvider):
    """Request one write tool, then return normal text on later requests."""

    def __init__(self, call: ToolCall) -> None:
        super().__init__()
        self.call = call
        self.approval_reply_sent = asyncio.Event()

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters
        request_index = len(self.requests)
        self.requests.append(tuple(messages))
        self.tool_definitions.append(tuple(tools))
        self.request_contexts.append(request_context)
        if request_index == 0:
            yield StreamEvent.tool_calls_ready((self.call,))
            self.approval_reply_sent.set()
        else:
            yield StreamEvent.delta("Approval flow complete")
        yield StreamEvent.completed()


@dataclass(slots=True)
class ApprovalTool:
    """Record whether an approved write call reached the registry."""

    name: str = "write_file"
    description: str = "Write a controlled test file."
    parameters: Mapping[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only: bool = False
    calls: list[str] = field(default_factory=list)

    async def execute(self, arguments_json: str) -> Result:
        self.calls.append(arguments_json)
        return Result("written")


@pytest.mark.asyncio
async def test_subagent_consumers_inject_done_and_show_approval_source(
    tmp_path: Path,
) -> None:
    provider = FakeProvider()
    conversation = Conversation(SYSTEM_PROMPT)
    task_manager = TaskManager()
    broker = SubagentApprovalBroker()
    app = CodewrightApp(
        provider,
        conversation,
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        subagent_catalog=Catalog(),
        task_manager=task_manager,
        approval_broker=broker,
    )

    async with app.run_test() as pilot:
        child = app.main_agent
        completed = BackgroundTask(
            id="task-1",
            name="worker",
            description="inspect files",
            sub_agent=child,
            conversation=Conversation("child"),
            initial_prompt="inspect",
            status=Status.COMPLETED,
            result="three files",
            notification_generation=1,
        )
        task_manager.subscribe_done().put_nowait(completed)
        await pilot.pause()
        reminders = app.runtime.take_reminders()
        assert len(reminders) == 1
        assert "<task-notification>" in reminders[0]
        assert '"task_id":"task-1"' in reminders[0]
        assert "three files" in reminders[0]

        handled: list[tuple[str, str]] = []

        async def record_approval(request: ApprovalRequest, source: str) -> None:
            handled.append((source, request.name))
            request.respond.set_result(Outcome.ALLOW_ONCE)

        screen = cast(ChatScreen, app.screen)
        screen._handle_background_approval = record_approval  # type: ignore[method-assign]
        response = asyncio.get_running_loop().create_future()
        request = ApprovalRequest(
            "call-bg",
            "bash",
            '{"command":"echo ok"}',
            "Command execution requires approval.",
            response,
        )
        routed = asyncio.create_task(broker.request("explore", request))
        assert await asyncio.wait_for(routed, timeout=1) is Outcome.ALLOW_ONCE
        assert handled == [("explore", "bash")]

    response = asyncio.get_running_loop().create_future()
    request = ApprovalRequest(
        "call-bg",
        "bash",
        '{"command":"echo ok"}',
        "Command execution requires approval.",
        response,
    )
    widget = ApprovalWidget(request, source="explore")
    assert "来自 SubAgent explore" in widget.render().plain
    assert await broker.request("explore", request) is Outcome.DENY_ONCE


def permission_engine(root: Path, *, mode: Mode = Mode.DEFAULT) -> Engine:
    return Engine(
        root=root.resolve(),
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=root / ".codewright" / "settings.local.yaml",
        default_mode=mode,
    )


def lifecycle_hook_engine(path: Path) -> HookEngine:
    rules = [
        HookRule(
            event.value,
            event,
            HookAction(
                HookActionType.SHELL,
                shell=HookShellAction(f"printf '{event.value}\\n' >> {shlex.quote(str(path))}"),
            ),
        )
        for event in (
            HookEvent.SESSION_START,
            HookEvent.SESSION_END,
            HookEvent.SESSION_RESUME,
        )
    ]
    return HookEngine(rules, [])


def build_approval_app(
    root: Path,
) -> tuple[CodewrightApp, Conversation, ApprovalProvider, ApprovalTool, Engine]:
    call = ToolCall(
        "write-1",
        "write_file",
        '{"path":"generated.txt","content":"value"}',
    )
    provider = ApprovalProvider(call)
    tool = ApprovalTool()
    registry = Registry()
    registry.register(tool)
    conversation = Conversation(SYSTEM_PROMPT)
    engine = permission_engine(root)
    app = CodewrightApp(
        provider,
        conversation,
        registry,
        engine=engine,
        working_directory=root,
        version="0.5.0",
    )
    return app, conversation, provider, tool, engine


def build_app(
    provider: FakeProvider | None = None,
) -> tuple[CodewrightApp, Conversation, FakeProvider]:
    selected_provider = provider or FakeProvider()
    conversation = Conversation(SYSTEM_PROMPT)
    app = CodewrightApp(
        selected_provider,
        conversation,
        engine=permission_engine(Path.cwd()),
        working_directory=Path("/workspace/codewright"),
        version="0.5.0",
    )
    return app, conversation, selected_provider


def write_skill(
    path: Path,
    name: str,
    *,
    mode: str = "inline",
    context: str = "full",
    body: str = "Run $ARGUMENTS.",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {name} description\n"
        f"mode: {mode}\ncontext: {context}\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


def build_skill_app(
    tmp_path: Path,
    provider: FakeProvider,
    *,
    writer: Writer | None = None,
) -> tuple[CodewrightApp, Conversation, SkillLoader]:
    loader = SkillLoader(tmp_path, tmp_path / "home")
    loader.load_all()
    load_tool = LoadSkillTool(loader)
    tools = Registry()
    tools.register(load_tool)
    conversation = Conversation(
        SYSTEM_PROMPT,
        on_append=writer.on_append if writer is not None else None,
        on_replace=writer.on_replace if writer is not None else None,
    )
    app = CodewrightApp(
        provider,
        conversation,
        tools,
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        writer=writer,
        skill_loader=loader,
        load_skill_tool=load_tool,
    )
    return app, conversation, loader


def active_chat_screen(app: CodewrightApp) -> ChatScreen:
    """Return the active chat screen from Textual's screen stack."""
    return cast(ChatScreen, app.screen)


def test_app_rejects_builtin_command_name_conflicts_at_construction() -> None:
    commands = CommandRegistry()

    async def duplicate_help(_ui, _args: str) -> None:
        return None

    commands.register(Command("help", "duplicate", Kind.LOCAL, duplicate_help))
    with pytest.raises(RuntimeError, match="command conflict: help"):
        CodewrightApp(
            FakeProvider(),
            Conversation(SYSTEM_PROMPT),
            engine=permission_engine(Path.cwd()),
            command_registry=commands,
        )


def assert_idle(screen: ChatScreen) -> None:
    """Assert a fresh state read without retaining mypy narrowing across awaits."""
    assert screen.state is ChatState.IDLE


@pytest.mark.asyncio
async def test_startup_displays_identity_model_and_working_directory() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        screen = active_chat_screen(app)
        rendered_info = str(screen.query_one("#app-info").render())

        assert "Codewright v0.5.0" in rendered_info
        assert "Provider:" not in rendered_info
        assert "Model: fake-model" in rendered_info
        assert "Working directory: /workspace/codewright" in rendered_info
        assert screen.query_one(MessageInput).has_focus


@pytest.mark.asyncio
async def test_command_completion_activates_filters_and_sits_above_input() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(100, 32)) as pilot:
        screen = active_chat_screen(app)
        completion = screen.query_one("#command-completion", Static)
        input_widget = screen.query_one(MessageInput)
        assert completion.display is False
        assert input_widget.placeholder == "Ask Codewright... (/help for commands)"

        await pilot.press("/")
        await pilot.pause()
        assert completion.display is True
        assert len(screen._completion.items) == 16
        first_render = str(completion.render())
        assert "/clear" in first_render
        assert "/help" in first_render
        assert completion.region.y < input_widget.region.y
        assert completion.region.height <= 8

        await pilot.press("s")
        await pilot.pause()
        assert [command.name for command in screen._completion.items] == [
            "session",
            "skill",
            "status",
        ]
        assert "/session" in str(completion.render())
        assert "/status" in str(completion.render())


@pytest.mark.asyncio
async def test_first_completion_render_does_not_wait_for_menu_layout() -> None:
    app, _, _ = build_app()

    async with app.run_test(size=(100, 32)):
        screen = active_chat_screen(app)
        completion = screen.query_one("#command-completion", Static)
        assert completion.display is False

        screen._completion.update("/", screen._command_registry)
        screen._render_completion()

        rendered = str(completion.render())
        assert "/clear" in rendered
        assert "/help" in rendered


@pytest.mark.asyncio
async def test_command_completion_navigation_and_enter_execute_selected() -> None:
    app, conversation, provider = build_app()

    async with app.run_test() as pilot:
        await pilot.press(*"/s", "down", "down", "enter")
        await pilot.pause()
        screen = active_chat_screen(app)

        assert screen.query_one(MessageInput).value == ""
        assert screen._completion.active is False
        assert screen.query_one("#command-completion", Static).display is False
        assert "Mode" in str(screen.query(".notice-message").last().render())

    assert provider.requests == []
    assert len(conversation.messages()) == 1


@pytest.mark.asyncio
async def test_command_completion_tab_executes_and_escape_preserves_input() -> None:
    app, conversation, provider = build_app()

    async with app.run_test() as pilot:
        await pilot.press(*"/perm", "tab")
        await pilot.pause()
        screen = active_chat_screen(app)
        input_widget = screen.query_one(MessageInput)
        assert input_widget.value == ""
        assert screen._completion.active is False
        assert "default" in str(screen.query(".notice-message").last().render())

        await pilot.press(*"/stat", "escape")
        await pilot.pause()
        assert input_widget.value == "/stat"
        assert input_widget.has_focus
        assert screen._completion.active is False

    assert provider.requests == []
    assert len(conversation.messages()) == 1


@pytest.mark.asyncio
async def test_zero_match_tab_hides_but_enter_submits_unknown_command() -> None:
    app, conversation, provider = build_app()

    async with app.run_test() as pilot:
        await pilot.press(*"/zzz", "tab")
        await pilot.pause()
        screen = active_chat_screen(app)
        input_widget = screen.query_one(MessageInput)
        completion = screen.query_one("#command-completion", Static)
        assert input_widget.value == "/zzz"
        assert screen._completion.active is False
        assert completion.display is False

        input_widget.value = ""
        await pilot.pause()
        await pilot.press(*"/zzz", "enter")
        await pilot.pause()
        assert input_widget.value == ""
        assert screen._completion.active is False
        assert "未知命令: /zzz" in str(screen.query(".notice-message").last().render())

    assert provider.requests == []
    assert len(conversation.messages()) == 1


@pytest.mark.asyncio
async def test_streaming_response_is_completed_once_and_shows_elapsed_time() -> None:
    app, conversation, provider = build_app()

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*"hello", "enter")
        await pilot.pause()

        screen = active_chat_screen(app)
        messages = screen.query(ConversationMessage)
        assert [message.role for message in messages] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert messages.last().message_content == "# Answer\n\n- item"
        assert screen.state_history == (
            ChatState.IDLE,
            ChatState.WAITING,
            ChatState.STREAMING,
            ChatState.COMPLETED,
            ChatState.IDLE,
        )
        status = str(screen.query_one(StatusWidget).render())
        assert re.fullmatch(r"Completed in \d+\.\d{2}s \| DEFAULT \| ↑0 ↓0 tok", status)
        assert screen.query_one(MessageInput).disabled is False

    assert provider.chat_calls == 0
    assert conversation.messages()[-1] == Message(MessageRole.ASSISTANT, "# Answer\n\n- item")


@pytest.mark.asyncio
async def test_first_delta_is_visible_before_stream_completion() -> None:
    provider = FakeProvider(pause_after_first_delta=True)
    app, conversation, _ = build_app(provider)

    async with app.run_test() as pilot:
        await pilot.press(*"hello", "enter")
        await provider.first_delta.wait()
        await pilot.pause()

        screen = active_chat_screen(app)
        assert screen.state is ChatState.STREAMING
        assert screen.query(ConversationMessage).last().message_content == "# Answer\n\n"
        assert "Iteration 1" in str(screen.query_one(StatusWidget).render())
        assert [message.role for message in conversation.messages()] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
        ]

        provider.release.set()
        await pilot.pause()
        assert_idle(screen)


@pytest.mark.asyncio
async def test_empty_input_does_not_call_provider() -> None:
    app, conversation, provider = build_app()

    async with app.run_test() as pilot:
        await pilot.press("space", "space", "enter")
        await pilot.pause()
        assert active_chat_screen(app).state is ChatState.IDLE

    assert provider.requests == []
    assert len(conversation.messages()) == 1


@pytest.mark.asyncio
async def test_exit_command_is_not_added_to_conversation() -> None:
    app, conversation, provider = build_app()

    async with app.run_test() as pilot:
        await pilot.press(*"/exit", "enter")

    assert provider.requests == []
    assert len(conversation.messages()) == 1


@pytest.mark.asyncio
async def test_unknown_slash_command_is_friendly_and_does_not_call_provider() -> None:
    app, conversation, provider = build_app()

    async with app.run_test() as pilot:
        await pilot.press(*"/unknown", "enter")
        await pilot.pause()
        screen = active_chat_screen(app)

        assert "未知命令: /unknown" in str(screen.query(".notice-message").last().render())

    assert provider.requests == []
    assert len(conversation.messages()) == 1


@pytest.mark.asyncio
async def test_help_and_status_commands_use_registry_without_calling_provider() -> None:
    app, conversation, provider = build_app()

    async with app.run_test() as pilot:
        await pilot.press(*"/help", "enter")
        await pilot.pause()
        screen = active_chat_screen(app)
        help_text = str(screen.query(".notice-message").last().render())
        assert "/help" in help_text
        assert "/review" in help_text
        assert "/status" in help_text

        await pilot.press(*"/status", "enter")
        await pilot.pause()
        status_text = str(screen.query(".notice-message").last().render())
        assert "Mode" in status_text
        assert "fake-model" in status_text
        assert "/workspace/codewright" in status_text

    assert provider.requests == []
    assert len(conversation.messages()) == 1


@pytest.mark.asyncio
async def test_all_local_commands_leave_history_and_provider_untouched() -> None:
    app, conversation, provider = build_app()

    async with app.run_test() as pilot:
        for command in ("/help", "/status", "/memory", "/permission", "/session"):
            await pilot.press(*command, "enter")
            await pilot.pause()

        screen = active_chat_screen(app)
        assert len(screen.query(".notice-message")) == 5

    assert provider.requests == []
    assert conversation.messages() == (Message(MessageRole.SYSTEM, SYSTEM_PROMPT),)


@pytest.mark.asyncio
async def test_dispatch_allows_local_but_rejects_ui_command_while_busy() -> None:
    app, _, provider = build_app()

    async with app.run_test() as pilot:
        screen = active_chat_screen(app)
        screen._state = ChatState.WAITING
        assert await dispatch_slash("/help", screen._command_registry, screen)
        assert "/help" in str(screen.query(".notice-message").last().render())

        assert await dispatch_slash("/plan", screen._command_registry, screen)
        error_text = str(screen.query(".error-message").last().render())
        assert "当前任务正在运行" in error_text
        assert screen.mode is Mode.DEFAULT
        screen._state = ChatState.IDLE
        await pilot.press("ctrl+c")

    assert provider.requests == []


@pytest.mark.asyncio
async def test_dispatch_passes_raw_args_only_to_accepting_commands() -> None:
    app, conversation, provider = build_app()
    received: list[str] = []

    async def argument_handler(_ui, args: str) -> None:
        received.append(args)

    async with app.run_test() as pilot:
        screen = active_chat_screen(app)
        screen._command_registry.register(
            Command(
                "argument-test",
                "accept arguments",
                Kind.LOCAL,
                argument_handler,
                accepts_args=True,
                source="skill",
            )
        )

        assert await dispatch_slash(
            "/argument-test first  second",
            screen._command_registry,
            screen,
        )
        assert received == ["first  second"]

        assert await dispatch_slash("/help forbidden", screen._command_registry, screen)
        assert "未知命令: /help forbidden" in str(screen.query(".notice-message").last().render())
        await pilot.press("ctrl+c")

    assert provider.requests == []
    assert len(conversation.messages()) == 1


@pytest.mark.asyncio
async def test_skill_management_and_reload_update_help_completion_and_catalog(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    skills_dir = tmp_path / ".codewright" / "skills"
    alpha = write_skill(skills_dir / "alpha.md", "alpha")
    provider = FakeProvider()
    app, conversation, _ = build_skill_app(tmp_path, provider)

    async with app.run_test() as pilot:
        screen = active_chat_screen(app)
        assert screen._command_registry.lookup("alpha") is not None

        await pilot.press(*"/skill list", "enter")
        await pilot.pause()
        assert "/alpha" in str(screen.query(".notice-message").last().render())
        assert provider.requests == []

        alpha.unlink()
        write_skill(skills_dir / "beta.md", "beta", body="BETA PRIVATE BODY")
        bad_body = "BROKEN PRIVATE BODY"
        (skills_dir / "broken.md").write_text(
            f"---\nname: BROKEN\ndescription: invalid\n---\n{bad_body}\n",
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            await pilot.press(*"/skill reload", "enter")
            await pilot.pause()

        assert screen._command_registry.lookup("alpha") is None
        assert screen._command_registry.lookup("beta") is not None
        screen._completion.update("/b", screen._command_registry)
        assert [command.name for command in screen._completion.items] == ["beta"]

        await pilot.press(*"/help", "enter")
        await pilot.pause()
        help_text = str(screen.query(".notice-message").last().render())
        assert "/beta" in help_text and "/alpha" not in help_text
        system = conversation.messages()[0].content
        assert "`beta`" in system
        assert "BETA PRIVATE BODY" not in system
        assert screen._command_registry.lookup("broken") is None
        assert "Skipping invalid skill" in caplog.text
        assert bad_body not in caplog.text


@pytest.mark.asyncio
async def test_install_tool_app_callback_immediately_refreshes_commands_and_catalog(
    tmp_path: Path,
) -> None:
    body = "---\nname: remote\ndescription: Remote Skill\n---\nREMOTE PRIVATE BODY\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, request=request)

    loader = SkillLoader(tmp_path, tmp_path / "home")
    loader.load_all()
    load_tool = LoadSkillTool(loader)
    install_tool = InstallSkillTool(
        SkillInstaller(
            loader.user_dir,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
        loader,
    )
    registry = Registry()
    registry.register(load_tool)
    registry.register(install_tool)
    conversation = Conversation(SYSTEM_PROMPT)
    app = CodewrightApp(
        FakeProvider(),
        conversation,
        registry,
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        skill_loader=loader,
        load_skill_tool=load_tool,
        install_skill_tool=install_tool,
    )

    async with app.run_test():
        result = await install_tool.execute(
            '{"url":"https://raw.githubusercontent.com/o/r/main/SKILL.md"}'
        )
        screen = active_chat_screen(app)

        assert result.content == "Skill installed: remote"
        assert screen._command_registry.lookup("remote") is not None
        screen._completion.update("/r", screen._command_registry)
        assert [command.name for command in screen._completion.items] == [
            "remote",
            "resume",
            "review",
        ]
        assert "`remote`" in conversation.messages()[0].content
        assert "REMOTE PRIVATE BODY" not in conversation.messages()[0].content


@pytest.mark.asyncio
async def test_skill_info_and_builtin_conflict_do_not_expose_body_or_call_provider(
    tmp_path: Path,
) -> None:
    skills_dir = tmp_path / ".codewright" / "skills"
    write_skill(skills_dir / "help.md", "help", body="CONFLICT SECRET BODY")
    provider = FakeProvider()
    app, conversation, _ = build_skill_app(tmp_path, provider)

    async with app.run_test() as pilot:
        screen = active_chat_screen(app)
        help_command = screen._command_registry.lookup("help")
        assert help_command is not None and help_command.source == "builtin"

        await pilot.press(*"/skill info help", "enter")
        await pilot.pause()
        info = str(screen.query(".notice-message").last().render())
        assert "name: help" in info
        assert "resource root:" in info
        assert "CONFLICT SECRET BODY" not in info

    assert provider.requests == []
    assert len(conversation.messages()) == 1


@pytest.mark.asyncio
async def test_inline_skill_uses_normal_user_writer_and_agent_path(tmp_path: Path) -> None:
    write_skill(
        tmp_path / ".codewright" / "skills" / "review.md",
        "review-project",
        body="Review $ARGUMENTS carefully.",
    )
    context = new_session_context(str(tmp_path))
    writer = Writer(context.session_dir, "fake-model")
    provider = FakeProvider(responses=[("reviewed",)])
    app, conversation, _ = build_skill_app(tmp_path, provider, writer=writer)

    async with app.run_test() as pilot:
        await pilot.press(*"/review-project src", "enter")
        await pilot.pause()
        screen = active_chat_screen(app)

        assert screen.state is ChatState.IDLE
        assert conversation.messages()[-2].content == "Review src carefully."
        assert conversation.messages()[-1].content == "reviewed"
        assert provider.requests[0][-1].content == "Review src carefully."
        assert screen._agent.runtime.active_skills.names() == ("review-project",)
        user_view = next(
            item for item in screen.query(ConversationMessage) if item.role is MessageRole.USER
        )
        assert user_view.message_content == "/review-project src"

        records = [json.loads(line) for line in writer.path.read_text().splitlines()]
        assert records[0]["content"] == "Review src carefully."
        assert "/review-project src" not in writer.path.read_text()


@pytest.mark.asyncio
async def test_fork_skill_worker_commits_assistant_result_and_usage(tmp_path: Path) -> None:
    write_skill(
        tmp_path / ".codewright" / "skills" / "research.md",
        "research",
        mode="fork",
        context="none",
        body="Research $ARGUMENTS.",
    )
    usage = TokenUsage(20, 5, 25)
    provider = FakeProvider(responses=[("fork answer",)], usage=usage)
    context = new_session_context(str(tmp_path))
    writer = Writer(context.session_dir, "fake-model")
    app, conversation, _ = build_skill_app(tmp_path, provider, writer=writer)

    async with app.run_test() as pilot:
        await pilot.press(*"/research topic", "enter")
        await pilot.pause()
        screen = active_chat_screen(app)
        status = screen.query_one(StatusWidget)

        assert screen.state is ChatState.IDLE
        assert ChatState.WAITING in screen.state_history
        assert ChatState.COMPLETED in screen.state_history
        assert conversation.messages()[-1] == Message(MessageRole.ASSISTANT, "fork answer")
        assert provider.requests[0][-1] == Message(MessageRole.USER, "Research topic.")
        assert status.input_tokens == 20 and status.output_tokens == 5
        assert screen.query(ConversationMessage).last().message_content == "fork answer"
        assert screen.query_one(MessageInput).disabled is False
        records = [json.loads(line) for line in writer.path.read_text().splitlines()]
        assert [(record["role"], record["content"]) for record in records] == [
            ("assistant", "fork answer")
        ]


@pytest.mark.asyncio
async def test_fork_skill_worker_can_be_cancelled_without_committing_result(
    tmp_path: Path,
) -> None:
    write_skill(
        tmp_path / ".codewright" / "skills" / "research.md",
        "research",
        mode="fork",
        context="none",
    )
    provider = FakeProvider(pause_after_first_delta=True)
    app, conversation, _ = build_skill_app(tmp_path, provider)

    async with app.run_test() as pilot:
        await pilot.press(*"/research topic", "enter")
        await provider.first_delta.wait()
        await pilot.press("escape")
        await pilot.pause()
        screen = active_chat_screen(app)

        assert screen.state is ChatState.IDLE
        assert ChatState.CANCELLED in screen.state_history
        assert conversation.messages() == (conversation.messages()[0],)
        assert screen.query_one(MessageInput).disabled is False


@pytest.mark.asyncio
async def test_fork_skill_write_tool_uses_existing_approval_ui(tmp_path: Path) -> None:
    write_skill(
        tmp_path / ".codewright" / "skills" / "writer.md",
        "writer-skill",
        mode="fork",
        context="none",
    )
    call = ToolCall(
        "write-fork",
        "write_file",
        '{"path":"generated.txt","content":"value"}',
    )
    provider = ApprovalProvider(call)
    write_tool = ApprovalTool()
    loader = SkillLoader(tmp_path, tmp_path / "home")
    loader.load_all()
    load_tool = LoadSkillTool(loader)
    registry = Registry()
    registry.register(write_tool)
    registry.register(load_tool)
    conversation = Conversation(SYSTEM_PROMPT)
    app = CodewrightApp(
        provider,
        conversation,
        registry,
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        skill_loader=loader,
        load_skill_tool=load_tool,
    )

    async with app.run_test() as pilot:
        await pilot.press(*"/writer-skill target", "enter")
        await provider.approval_reply_sent.wait()
        await pilot.pause()
        screen = active_chat_screen(app)

        assert screen.state is ChatState.APPROVING
        assert screen.pending_approval is not None
        assert screen.query_one(ApprovalWidget)

        await pilot.press("1")
        await pilot.pause()

        assert screen.state is ChatState.IDLE
        assert write_tool.calls == ['{"path":"generated.txt","content":"value"}']
        assert conversation.messages()[-1] == Message(
            MessageRole.ASSISTANT,
            "Approval flow complete",
        )


@pytest.mark.asyncio
async def test_review_command_injects_fixed_prompt_but_displays_command_label() -> None:
    app, conversation, provider = build_app(FakeProvider(responses=[("review complete",)]))

    async with app.run_test() as pilot:
        await pilot.press(*"/review", "enter")
        await pilot.pause()
        screen = active_chat_screen(app)
        assert screen.query(ConversationMessage).first().message_content == "/review"

    assert provider.requests[0][-1] == Message(MessageRole.USER, REVIEW_DIRECTIVE)
    assert conversation.messages()[1].content == REVIEW_DIRECTIVE


@pytest.mark.asyncio
async def test_prompt_commands_persist_directives_and_call_provider(tmp_path: Path) -> None:
    context = new_session_context(str(tmp_path))
    writer = Writer(context.session_dir, "fake-model")
    conversation = Conversation(SYSTEM_PROMPT, on_append=writer.on_append)
    provider = FakeProvider(responses=[("reviewed",), ("executed",)])
    app = CodewrightApp(
        provider,
        conversation,
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        writer=writer,
    )

    async with app.run_test() as pilot:
        await pilot.press(*"/review", "enter")
        await pilot.pause()
        await pilot.press(*"/do", "enter")
        await pilot.pause()

        records = [json.loads(line) for line in writer.path.read_text().splitlines()]
        persisted_users = [record["content"] for record in records if record["role"] == "user"]
        assert persisted_users == [REVIEW_DIRECTIVE, EXECUTE_DIRECTIVE]
        assert "/review" not in persisted_users
        assert "/do" not in persisted_users

    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_command_failure_log_excludes_api_key_prompt_and_memory(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api_key = "t8-api-key-not-real"
    prompt = "t8-sensitive-system-prompt"
    memory = "t8-sensitive-memory-body"
    commands = CommandRegistry()

    async def fail_with_sensitive_context(_ui, _args: str) -> None:
        raise RuntimeError(f"{api_key} {prompt} {memory}")

    commands.register(Command("fail", "fail safely", Kind.LOCAL, fail_with_sensitive_context))
    provider = FakeProvider()
    conversation = Conversation(prompt)
    app = CodewrightApp(
        provider,
        conversation,
        engine=permission_engine(Path.cwd()),
        command_registry=commands,
    )

    with caplog.at_level(logging.ERROR, logger="codewright.tui.commands"):
        async with app.run_test() as pilot:
            await pilot.press(*"/fail", "enter")
            await pilot.pause()
            error = str(active_chat_screen(app).query(".error-message").last().render())
            assert "命令执行失败" in error

    assert "command=fail" in caplog.text
    assert api_key not in caplog.text
    assert prompt not in caplog.text
    assert memory not in caplog.text
    assert provider.requests == []
    assert conversation.messages() == (Message(MessageRole.SYSTEM, prompt),)


@pytest.mark.asyncio
async def test_slash_compact_does_not_enter_conversation_or_main_llm_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, conversation, provider = build_app()
    calls = 0

    async def fake_force_compact(value, definitions):
        nonlocal calls
        calls += 1
        assert value is conversation
        assert definitions == ()
        return 100, 20

    async with app.run_test() as pilot:
        screen = active_chat_screen(app)
        monkeypatch.setattr(screen._agent, "run_force_compact", fake_force_compact)
        await pilot.press(*"/compact", "enter")
        await pilot.pause()

        notices = [str(widget.render()) for widget in screen.query(".notice-message")]
        assert any("正在手动压缩上下文" in notice for notice in notices)
        assert any("token 从 100 降至 20" in notice for notice in notices)

    assert calls == 1
    assert provider.requests == []
    assert len(conversation.messages()) == 1


def test_format_compact_notice_covers_auto_emergency_and_safe_failure() -> None:
    auto_notice = format_compact_notice(CompactEvent(CompactPhase.BEFORE_AUTO))
    emergency_notice = format_compact_notice(CompactEvent(CompactPhase.BEFORE_EMERGENCY))
    assert "正在压缩上下文" in auto_notice
    assert "上下文撞墙" in emergency_notice
    assert format_compact_notice(
        CompactEvent(CompactPhase.AFTER_AUTO, error_message="safe failure")
    ).endswith("safe failure")


def test_agent_compact_event_remains_mutually_exclusive() -> None:
    compact = CompactEvent(CompactPhase.BEFORE_AUTO)

    assert Event.compact_event(compact).compact is compact
    with pytest.raises(ValueError, match="exactly one"):
        Event(text="invalid", compact=compact)


@pytest.mark.asyncio
async def test_idle_ctrl_c_exits_cleanly() -> None:
    app, _, provider = build_app()

    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")

    assert provider.requests == []


@pytest.mark.asyncio
async def test_stream_error_keeps_partial_text_out_of_history_and_recovers() -> None:
    provider = FakeProvider(responses=[("partial",), ("recovered",)], error_requests={0})
    app, conversation, _ = build_app(provider)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*"first", "enter")
        await pilot.pause()

        screen = active_chat_screen(app)
        assert ChatState.ERROR in screen.state_history
        assert "Response interrupted" in screen.query(ConversationMessage).last().message_content
        assert len(screen.query(".error-message")) == 1
        assert screen.query_one(MessageInput).disabled is False

        await pilot.press(*"second", "enter")
        await pilot.pause()
        assert conversation.messages()[-1] == Message(MessageRole.ASSISTANT, "recovered")

    assert [message.role for message in conversation.messages()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert conversation.messages()[2].content == NOTICE_STREAM_ERROR


@pytest.mark.asyncio
async def test_unexpected_provider_failure_is_safe_and_recovers_input() -> None:
    provider = UnexpectedFailureProvider()
    app, conversation, _ = build_app(provider)

    async with app.run_test() as pilot:
        await pilot.press(*"hello", "enter")
        await pilot.pause()

        screen = active_chat_screen(app)
        assert ChatState.ERROR in screen.state_history
        assert "unexpected error" in str(screen.query_one(StatusWidget).render())
        assert "unsafe provider detail" not in str(screen.query_one(StatusWidget).render())
        assert screen.query_one(MessageInput).disabled is False

    assert [message.role for message in conversation.messages()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert conversation.messages()[-1].content == NOTICE_STREAM_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_key", ["ctrl+c", "escape"])
async def test_active_cancel_key_closes_generation_and_history(cancel_key: str) -> None:
    provider = FakeProvider(pause_after_first_delta=True)
    app, conversation, _ = build_app(provider)

    async with app.run_test() as pilot:
        await pilot.press(*"hello", "enter")
        await provider.first_delta.wait()
        await pilot.press(cancel_key)
        await pilot.pause()

        screen = active_chat_screen(app)
        assert screen.state is ChatState.IDLE
        assert ChatState.CANCELLED in screen.state_history
        assert "Response cancelled" in screen.query(ConversationMessage).last().message_content
        assert "Cancelled after" in str(screen.query_one(StatusWidget).render())
        assert screen.query_one(MessageInput).disabled is False

    assert [message.role for message in conversation.messages()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert conversation.messages()[-1].content == NOTICE_CANCELLED


@pytest.mark.asyncio
async def test_plan_and_do_switch_mode_without_storing_commands() -> None:
    provider = FakeProvider(responses=[("A read-only plan",), ("Plan executed",)])
    conversation = Conversation(SYSTEM_PROMPT)
    app = CodewrightApp(
        provider,
        conversation,
        new_default_registry(working_directory=Path("/workspace/codewright")),
        engine=permission_engine(Path.cwd()),
        working_directory=Path("/workspace/codewright"),
    )

    async with app.run_test() as pilot:
        await pilot.press(*"/plan", "enter")
        await pilot.pause()
        screen = active_chat_screen(app)
        assert screen.mode is Mode.PLAN
        assert provider.requests == []
        assert "PLAN" in str(screen.query_one(StatusWidget).render())
        assert len(screen.query(".notice-message")) == 1

        await pilot.press(*"inspect the project", "enter")
        await pilot.pause()
        assert [definition.name for definition in provider.tool_definitions[0]] == [
            "read_file",
            "glob",
            "grep",
        ]
        assert provider.requests[0][0].content == SYSTEM_PROMPT
        assert provider.request_contexts[0] is not None
        assert provider.request_contexts[0].reminder == plan_reminder(full=True)
        assert PLAN_MODE_REMINDER in provider.request_contexts[0].reminder
        assert PLAN_MODE_REMINDER not in conversation.messages()[0].content

        await pilot.press(*"/do", "enter")
        await pilot.pause()
        assert screen.mode is Mode.DEFAULT
        assert provider.requests[1][-1].content == EXECUTE_DIRECTIVE
        assert all(message.content not in {"/plan", "/do"} for message in conversation.messages())
        assert conversation.messages()[-1].content == "Plan executed"


@pytest.mark.asyncio
async def test_status_accumulates_usage_across_user_turns() -> None:
    usage = TokenUsage(input_tokens=1_200, output_tokens=250, total_tokens=1_450)
    provider = FakeProvider(responses=[("first",), ("second",)], usage=usage)
    app, _, _ = build_app(provider)

    async with app.run_test() as pilot:
        await pilot.press(*"one", "enter")
        await pilot.pause()
        await pilot.press(*"two", "enter")
        await pilot.pause()

        status = screen_status = active_chat_screen(app).query_one(StatusWidget)
        assert status.input_tokens == 2_400
        assert status.output_tokens == 500
        assert "↑2.4k ↓500 tok" in str(screen_status.render())


@pytest.mark.asyncio
async def test_streaming_state_rejects_duplicate_submission() -> None:
    provider = FakeProvider(pause_after_first_delta=True)
    app, conversation, _ = build_app(provider)

    async with app.run_test() as pilot:
        await pilot.press(*"first", "enter")
        await provider.first_delta.wait()
        await pilot.press(*"second", "enter")
        await pilot.press("shift+tab")
        await pilot.pause()

        assert len(provider.requests) == 1
        assert [message.content for message in conversation.messages()][-1] == "first"
        assert active_chat_screen(app).mode is Mode.DEFAULT

        provider.release.set()
        await pilot.pause()


@pytest.mark.asyncio
async def test_second_request_contains_complete_first_round_history() -> None:
    provider = FakeProvider(responses=[("first answer",), ("second answer",)])
    app, conversation, _ = build_app(provider)

    async with app.run_test() as pilot:
        await pilot.press(*"first question", "enter")
        await pilot.pause()
        await pilot.press(*"second question", "enter")
        await pilot.pause()

    assert provider.requests[1] == (
        Message(MessageRole.SYSTEM, SYSTEM_PROMPT),
        Message(MessageRole.USER, "first question"),
        Message(MessageRole.ASSISTANT, "first answer"),
        Message(MessageRole.USER, "second question"),
    )
    assert conversation.messages()[-1] == Message(MessageRole.ASSISTANT, "second answer")


@pytest.mark.asyncio
async def test_app_closes_provider_during_shutdown() -> None:
    app, _, provider = build_app()

    async with app.run_test() as pilot:
        await pilot.press("ctrl+c")

    assert provider.closed is True


def test_chat_state_contains_all_planned_states() -> None:
    assert {state.value for state in ChatState} == {
        "idle",
        "resuming",
        "waiting",
        "streaming",
        "approving",
        "completed",
        "error",
        "cancelled",
        "exiting",
    }


@pytest.mark.asyncio
async def test_resume_command_restores_selected_history_and_old_timestamp_notice(
    tmp_path: Path,
) -> None:
    current_context = new_session_context(str(tmp_path))
    current_writer = Writer(current_context.session_dir, "fake-model")
    current = Conversation(SYSTEM_PROMPT, on_append=current_writer.on_append)
    current.add_user("current conversation")

    historical_context = new_session_context(str(tmp_path))
    historical_writer = Writer(historical_context.session_dir, "fake-model")
    historical_writer.append(Message(MessageRole.USER, "unique historical topic"))
    historical_writer.append(Message(MessageRole.ASSISTANT, "historical answer"))
    historical_writer.close()
    history_path = Path(historical_context.session_dir) / "conversation.jsonl"
    records = [json.loads(line) for line in history_path.read_text().splitlines()]
    for record in records:
        record["ts"] = int(time.time()) - 7 * 3600
    history_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    provider = FakeProvider()
    lifecycle_log = tmp_path / "resume-hooks.log"
    hooks = lifecycle_hook_engine(lifecycle_log)
    app = CodewrightApp(
        provider,
        current,
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        writer=current_writer,
        sessions_dir=str(tmp_path / ".codewright" / "sessions"),
        hook_engine=hooks,
    )
    app.runtime.active_skills.activate("old-skill", "old body", tmp_path.resolve())

    async with app.run_test() as pilot:
        await pilot.press(*"/resume", "enter")
        await pilot.pause()
        screen = active_chat_screen(app)
        assert screen.state is ChatState.RESUMING

        await pilot.press(*"unique historical", "enter")
        await pilot.pause()

        assert screen.state is ChatState.IDLE
        restored = screen._conversation.messages()
        assert restored[1].content == "unique historical topic"
        assert restored[2].content == "historical answer"
        assert "本会话已暂停" in restored[-1].content
        assert "已恢复会话" in str(screen.query(".notice-message").last().render())
        assert app.runtime.active_skills.names() == ()
        assert lifecycle_log.read_text(encoding="utf-8").splitlines() == [
            "SessionStart",
            "SessionEnd",
            "SessionResume",
        ]

    await hooks.aclose()
    assert lifecycle_log.read_text(encoding="utf-8").splitlines() == [
        "SessionStart",
        "SessionEnd",
        "SessionResume",
        "SessionEnd",
    ]
    assert provider.requests == []


@pytest.mark.asyncio
async def test_clear_starts_new_durable_session_and_resets_session_state(tmp_path: Path) -> None:
    old_context = new_session_context(str(tmp_path))
    old_writer = Writer(old_context.session_dir, "fake-model")
    conversation = Conversation("dynamic system prompt", on_append=old_writer.on_append)
    conversation.add_user("old session message")
    provider = FakeProvider(responses=[("new answer",)])
    lifecycle_log = tmp_path / "clear-hooks.log"
    hooks = lifecycle_hook_engine(lifecycle_log)
    app = CodewrightApp(
        provider,
        conversation,
        engine=permission_engine(tmp_path, mode=Mode.ACCEPT_EDITS),
        working_directory=tmp_path,
        writer=old_writer,
        sessions_dir=str(tmp_path / ".codewright" / "sessions"),
        hook_engine=hooks,
    )
    app.runtime.usage_anchor = 99
    app.runtime.anchor_msg_len = 2
    app.runtime.turn_count = 4
    app.runtime.active_skills.activate("old-skill", "old body", tmp_path.resolve())

    async with app.run_test() as pilot:
        screen = active_chat_screen(app)
        status = screen.query_one(StatusWidget)
        status.add_usage(TokenUsage(30, 20, 50))
        await screen.println("visible old history")
        old_path = old_writer.path

        await pilot.press(*"/clear", "enter")
        await pilot.pause()

        new_writer = screen._writer
        assert new_writer is not None and new_writer is not old_writer
        assert app._current_writer is new_writer
        assert screen._conversation.messages() == (
            Message(MessageRole.SYSTEM, "dynamic system prompt"),
        )
        assert screen.mode is Mode.ACCEPT_EDITS
        assert screen.session_id() != old_context.session_id
        assert screen.session_path() == str(new_writer.path)
        assert list(screen.query_one("#conversation-view", VerticalScroll).children) == []
        assert (status.input_tokens, status.output_tokens) == (0, 0)
        assert app.runtime.usage_anchor == 0
        assert app.runtime.anchor_msg_len == 0
        assert app.runtime.turn_count == 0
        assert app.runtime.active_skills.names() == ()
        assert lifecycle_log.read_text(encoding="utf-8").splitlines() == [
            "SessionStart",
            "SessionEnd",
            "SessionStart",
        ]
        with pytest.raises(RuntimeError, match="closed"):
            old_writer.append(Message(MessageRole.USER, "must fail"))

        await pilot.press(*"new session message", "enter")
        await pilot.pause()
        records = [json.loads(line) for line in new_writer.path.read_text().splitlines()]
        assert records[0]["content"] == "new session message"
        assert old_path.exists()

        sessions = list_sessions(str(tmp_path / ".codewright" / "sessions"))
        assert old_context.session_id in {item.id for item in sessions}

    await hooks.aclose()


@pytest.mark.asyncio
async def test_clear_writer_failure_rolls_back_current_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_context = new_session_context(str(tmp_path))
    old_writer = Writer(old_context.session_dir, "fake-model")
    conversation = Conversation("current system", on_append=old_writer.on_append)
    conversation.add_user("keep this message")
    lifecycle_log = tmp_path / "failed-clear-hooks.log"
    hooks = lifecycle_hook_engine(lifecycle_log)
    app = CodewrightApp(
        FakeProvider(),
        conversation,
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        writer=old_writer,
        hook_engine=hooks,
    )
    app.runtime.active_skills.activate("keep-skill", "keep body", tmp_path.resolve())

    async with app.run_test() as pilot:
        screen = active_chat_screen(app)
        await screen.println("keep this view")
        old_runtime_session = app.runtime.session
        old_runtime_parts = (
            app.runtime.replacement,
            app.runtime.recovery,
            app.runtime.auto_tracking,
        )

        def fail_writer(_session_dir: str, _model: str) -> Writer:
            raise OSError("unsafe detail")

        monkeypatch.setattr("codewright.tui.screens.chat.Writer", fail_writer)
        await pilot.press(*"/clear", "enter")
        await pilot.pause()

        assert screen._conversation is conversation
        assert screen._writer is old_writer
        assert app._current_writer is old_writer
        assert app.runtime.session is old_runtime_session
        assert (
            app.runtime.replacement,
            app.runtime.recovery,
            app.runtime.auto_tracking,
        ) == old_runtime_parts
        assert app.runtime.active_skills.names() == ("keep-skill",)
        notices = [str(item.render()) for item in screen.query(".notice-message")]
        assert "keep this view" in notices
        error_text = str(screen.query(".error-message").last().render())
        assert "当前会话未改变" in error_text
        assert "unsafe detail" not in error_text
        assert lifecycle_log.read_text(encoding="utf-8").splitlines() == ["SessionStart"]

        conversation.add_user("still writable")
        records = [json.loads(line) for line in old_writer.path.read_text().splitlines()]
        assert records[-1]["content"] == "still writable"

    await hooks.aclose()


def test_assistant_message_falls_back_to_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    content = "# Markdown that must not be lost"

    def fail_markdown(_: str) -> None:
        raise ValueError("render failure")

    monkeypatch.setattr("codewright.tui.widgets.message.Markdown", fail_markdown)
    rendered = ConversationMessage(MessageRole.ASSISTANT, content).render()

    assert isinstance(rendered, Text)
    assert rendered.plain == content


@pytest.mark.asyncio
async def test_shift_tab_cycles_all_modes_and_writes_notices() -> None:
    app, _, _ = build_app()

    async with app.run_test() as pilot:
        screen = active_chat_screen(app)
        observed = [screen.mode]
        for _ in range(4):
            await pilot.press("shift+tab")
            await pilot.pause()
            observed.append(screen.mode)
            assert screen.state is ChatState.IDLE

        assert observed == [
            Mode.DEFAULT,
            Mode.ACCEPT_EDITS,
            Mode.PLAN,
            Mode.BYPASS,
            Mode.DEFAULT,
        ]
        assert len(screen.query(".notice-message")) == 4


@pytest.mark.asyncio
async def test_approval_down_enter_selects_allow_forever(tmp_path: Path) -> None:
    app, _, provider, tool, engine = build_approval_app(tmp_path)

    async with app.run_test(size=(100, 32)) as pilot:
        await pilot.press(*"write a file", "enter")
        await provider.approval_reply_sent.wait()
        await pilot.pause()
        screen = active_chat_screen(app)

        assert screen.state is ChatState.APPROVING
        assert screen.pending_approval is not None
        assert screen.approve_cursor == 0
        approval = screen.query_one(ApprovalWidget)
        assert "是否继续?" in str(approval.render())

        await pilot.press("down")
        await pilot.pause()
        assert screen.approve_cursor == 1
        await pilot.press("enter")
        await pilot.pause()

        assert screen.state is ChatState.IDLE
        assert tool.calls == ['{"path":"generated.txt","content":"value"}']
        assert "Write(generated.txt)" in engine.local_path.read_text(encoding="utf-8")
        assert not screen.query(ApprovalWidget)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "executions", "is_error"),
    [("1", 1, False), ("3", 0, True)],
)
async def test_approval_number_keys_choose_once_or_deny(
    tmp_path: Path,
    key: str,
    executions: int,
    is_error: bool,
) -> None:
    app, conversation, provider, tool, _ = build_approval_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.press(*"write a file", "enter")
        await provider.approval_reply_sent.wait()
        await pilot.pause()
        assert active_chat_screen(app).state is ChatState.APPROVING

        await pilot.press(key)
        await pilot.pause()

        assert active_chat_screen(app).state is ChatState.IDLE
        assert len(tool.calls) == executions
        result = conversation.messages()[3].tool_results[0]
        assert result.is_error is is_error


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_key", ["escape", "ctrl+c"])
async def test_approval_cancel_repairs_history_and_app_remains_usable(
    tmp_path: Path,
    cancel_key: str,
) -> None:
    app, conversation, provider, tool, _ = build_approval_app(tmp_path)

    async with app.run_test() as pilot:
        await pilot.press(*"write a file", "enter")
        await provider.approval_reply_sent.wait()
        await pilot.pause()
        screen = active_chat_screen(app)
        request = screen.pending_approval
        assert request is not None

        await pilot.press(cancel_key)
        await pilot.pause()

        assert screen.state is ChatState.IDLE
        assert request.respond.cancelled()
        assert tool.calls == []
        assert conversation.messages()[3].tool_results[0].error_code == "cancelled"
        assert conversation.last_role() is MessageRole.ASSISTANT

        await pilot.press(*"continue", "enter")
        await pilot.pause()
        assert conversation.messages()[-1].content == "Approval flow complete"


@pytest.mark.asyncio
async def test_status_supports_four_modes_and_initial_engine_mode(tmp_path: Path) -> None:
    provider = FakeProvider()
    engine = permission_engine(tmp_path, mode=Mode.ACCEPT_EDITS)
    app = CodewrightApp(
        provider,
        Conversation(SYSTEM_PROMPT),
        engine=engine,
        working_directory=tmp_path,
    )

    async with app.run_test() as pilot:
        screen = active_chat_screen(app)
        status = screen.query_one(StatusWidget)
        assert screen.mode is Mode.ACCEPT_EDITS
        assert "ACCEPT EDITS" in str(status.render())

        expected = ["PLAN", "BYPASS", "DEFAULT"]
        for label in expected:
            await pilot.press("shift+tab")
            await pilot.pause()
            assert label in str(status.render())


def test_status_reset_usage_preserves_mode_and_clears_iteration() -> None:
    status = StatusWidget()
    status.set_mode("plan")
    status.set_iteration(4)
    status.add_usage(TokenUsage(input_tokens=120, output_tokens=30, total_tokens=150))

    status.reset_usage()

    rendered = str(status.render())
    assert status.input_tokens == 0
    assert status.output_tokens == 0
    assert "PLAN" in rendered
    assert "Iteration" not in rendered
    assert "↑0 ↓0 tok" in rendered


@pytest.mark.asyncio
async def test_selected_mode_persists_across_user_turns() -> None:
    provider = FakeProvider(responses=[("first",), ("second",)])
    app, _, _ = build_app(provider)

    async with app.run_test() as pilot:
        await pilot.press("shift+tab")
        await pilot.pause()
        screen = active_chat_screen(app)
        assert screen.mode is Mode.ACCEPT_EDITS

        await pilot.press(*"first", "enter")
        await pilot.pause()
        await pilot.press(*"second", "enter")
        await pilot.pause()

        assert screen.mode is Mode.ACCEPT_EDITS
        assert "ACCEPT EDITS" in str(screen.query_one(StatusWidget).render())


@pytest.mark.asyncio
async def test_approval_preview_is_redacted_and_bounded() -> None:
    secret = "unique-approval-preview-secret"
    register_secrets((secret,))
    future: asyncio.Future[Outcome] = asyncio.get_running_loop().create_future()
    request = ApprovalRequest(
        "call-1",
        "write_file",
        f'{{"path":"safe.txt","content":"{secret}{"x" * 1_000}"}}',
        "default mode confirmation",
        future,
    )

    rendered = str(ApprovalWidget(request).render())

    assert secret not in rendered
    assert "[REDACTED]" in rendered
    assert "[truncated]" in rendered
    future.cancel()
