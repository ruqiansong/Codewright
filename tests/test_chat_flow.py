"""Offline end-to-end acceptance scenarios for Codewright V0.1."""

import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import yaml
from conftest import ScriptedProvider, ScriptedReply
from rich.markdown import Markdown
from textual.pilot import Pilot

from codewright import __version__, cli
from codewright.agent import NOTICE_CANCELLED, NOTICE_STREAM_ERROR
from codewright.agent.runtime import SessionRuntime
from codewright.config import ProviderConfig
from codewright.conversation import Conversation
from codewright.hook import Action as HookAction
from codewright.hook import ActionType as HookActionType
from codewright.hook import Engine as HookEngine
from codewright.hook import Event as HookEvent
from codewright.hook import PromptAction as HookPromptAction
from codewright.hook import Rule as HookRule
from codewright.hook import ShellAction as HookShellAction
from codewright.llm import (
    LLMAuthenticationError,
    LLMError,
    LLMModelNotFoundError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMServiceError,
    LLMTimeoutError,
    Message,
    MessageRole,
)
from codewright.memory import Manager as MemoryManager
from codewright.permission import Engine, Mode
from codewright.permission.rule import RuleSet
from codewright.prompt import SYSTEM_PROMPT
from codewright.session import Writer
from codewright.skills import SkillLoader
from codewright.subagent import Catalog
from codewright.task import Manager as TaskManager
from codewright.task import SubagentApprovalBroker
from codewright.tool import InstallSkillTool, LoadSkillTool, Registry
from codewright.tui import ChatScreen, ChatState, CodewrightApp
from codewright.tui.widgets.input import MessageInput
from codewright.tui.widgets.message import ConversationMessage
from codewright.tui.widgets.status import StatusWidget

SYNTHETIC_SECRET = "e2e-key-not-a-real-secret"


def permission_engine(root: Path | None = None) -> Engine:
    selected_root = (root or Path.cwd()).resolve()
    return Engine(
        root=selected_root,
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=selected_root / ".codewright" / "settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


def active_screen(app: CodewrightApp) -> ChatScreen:
    return cast(ChatScreen, app.screen)


class HeadlessAcceptanceApp:
    """Run the real TUI from CLI assembly with a deterministic autopilot."""

    instances: list["HeadlessAcceptanceApp"] = []

    def __init__(
        self,
        provider: ScriptedProvider,
        conversation: Conversation,
        registry: Registry,
        *,
        engine: Engine,
        working_directory: Path,
        version: str,
        stream: bool,
        runtime: SessionRuntime,
        writer: Writer,
        memory_manager: MemoryManager,
        instruction_text: str,
        base_prompt: str | None,
        sessions_dir: str,
        cleanup_task: object,
        skill_loader: SkillLoader,
        load_skill_tool: LoadSkillTool,
        install_skill_tool: InstallSkillTool,
        provider_configs: tuple[ProviderConfig, ...],
        hook_engine: HookEngine,
        subagent_catalog: Catalog,
        task_manager: TaskManager,
        approval_broker: SubagentApprovalBroker,
        worktree_manager: object | None,
    ) -> None:
        self.conversation = conversation
        self._app = CodewrightApp(
            provider,
            conversation,
            registry,
            engine=engine,
            working_directory=working_directory,
            version=version,
            stream=stream,
            runtime=runtime,
            writer=writer,
            memory_manager=memory_manager,
            instruction_text=instruction_text,
            base_prompt=base_prompt,
            sessions_dir=sessions_dir,
            cleanup_task=cleanup_task,
            skill_loader=skill_loader,
            load_skill_tool=load_skill_tool,
            install_skill_tool=install_skill_tool,
            provider_configs=provider_configs,
            hook_engine=hook_engine,
            subagent_catalog=subagent_catalog,
            task_manager=task_manager,
            approval_broker=approval_broker,
            worktree_manager=worktree_manager,
        )
        self.instances.append(self)

    @property
    def main_agent(self) -> object:
        return self._app.main_agent

    async def stop_subagent_consumers(self) -> None:
        await self._app.stop_subagent_consumers()

    async def run_async(self) -> None:
        async def drive(pilot: Pilot[object]) -> None:
            await pilot.press(*"my name is Zhang San", "enter")
            await pilot.pause()
            await pilot.press(*"what is my name", "enter")
            await pilot.pause()
            await pilot.press(*"/exit", "enter")

        await self._app.run_async(headless=True, size=(100, 30), auto_pilot=drive)


def write_cli_config(path: Path, *, stream: bool = True) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "providers": [
                    {
                        "name": "deepseek",
                        "protocol": "openai-compatible",
                        "api_key": SYNTHETIC_SECRET,
                        "base_url": "https://api.deepseek.com",
                        "model": "deepseek-chat",
                        "stream": stream,
                    }
                ],
                "default_provider": "deepseek",
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_to_tui_two_round_streaming_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = ScriptedProvider(
        [
            ScriptedReply(("I will remember ", "Zhang San.")),
            ScriptedReply(("Your name is ", "**Zhang San**.")),
        ]
    )
    config_path = write_cli_config(tmp_path / "config.yaml")
    HeadlessAcceptanceApp.instances.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setattr(cli, "create_provider", lambda _: provider)
    monkeypatch.setattr(cli, "CodewrightApp", HeadlessAcceptanceApp)
    monkeypatch.setattr(cli.mcp_client, "load_config", lambda _: cli.mcp_client.Config({}))

    assert cli.main(["--config", str(config_path), "--log-level", "ERROR"]) == 0

    assert len(provider.stream_requests) == 2
    assert provider.stream_requests[1] == (
        Message(MessageRole.SYSTEM, SYSTEM_PROMPT),
        Message(MessageRole.USER, "my name is Zhang San"),
        Message(MessageRole.ASSISTANT, "I will remember Zhang San."),
        Message(MessageRole.USER, "what is my name"),
    )
    assert HeadlessAcceptanceApp.instances[0].conversation.messages()[-1] == Message(
        MessageRole.ASSISTANT, "Your name is **Zhang San**."
    )
    assert provider.closed is True


@pytest.mark.asyncio
async def test_stream_delta_precedes_completion_and_final_markdown_is_complete() -> None:
    provider = ScriptedProvider(
        [ScriptedReply(("```python\n", "print('ok')", "\n```"), pause_after_delta=0)]
    )
    conversation = Conversation(SYSTEM_PROMPT)
    app = CodewrightApp(provider, conversation, engine=permission_engine())

    async with app.run_test() as pilot:
        await pilot.press(*"show code", "enter")
        await provider.paused.wait()
        await pilot.pause()

        screen = active_screen(app)
        assistant = screen.query(ConversationMessage).last()
        assert screen.state is ChatState.STREAMING
        assert assistant.message_content == "```python\n"
        assert len(conversation.messages()) == 2

        provider.release.set()
        await pilot.pause()
        assistant = screen.query(ConversationMessage).last()
        assert assistant.message_content == "```python\nprint('ok')\n```"
        assert isinstance(assistant.render(), Markdown)
    assert conversation.messages()[-1].content == assistant.message_content


@pytest.mark.asyncio
async def test_session_start_prompt_reaches_first_provider_request(tmp_path: Path) -> None:
    provider = ScriptedProvider([ScriptedReply(("done",))])
    hooks = HookEngine(
        [
            HookRule(
                "start-reminder",
                HookEvent.SESSION_START,
                HookAction(
                    HookActionType.PROMPT,
                    prompt=HookPromptAction("SESSION START REMINDER"),
                ),
            )
        ],
        [],
    )
    app = CodewrightApp(
        provider,
        Conversation(SYSTEM_PROMPT),
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        hook_engine=hooks,
    )

    async with app.run_test() as pilot:
        await pilot.press(*"hello", "enter")
        await pilot.pause()

    await hooks.aclose()
    context = provider.request_contexts[0]
    assert context is not None
    assert "SESSION START REMINDER" in context.reminder


@pytest.mark.asyncio
async def test_user_prompt_submit_hook_blocks_and_preserves_input(tmp_path: Path) -> None:
    provider = ScriptedProvider([ScriptedReply(("must not run",))])
    hooks = HookEngine(
        [
            HookRule(
                "block-submit",
                HookEvent.USER_PROMPT_SUBMIT,
                HookAction(
                    HookActionType.SHELL,
                    shell=HookShellAction("echo denied by policy >&2; exit 2"),
                ),
            )
        ],
        [],
    )
    conversation = Conversation(SYSTEM_PROMPT)
    app = CodewrightApp(
        provider,
        conversation,
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        hook_engine=hooks,
    )

    async with app.run_test() as pilot:
        await pilot.press(*"keep this input", "enter")
        await pilot.pause()
        screen = active_screen(app)
        assert screen.query_one(MessageInput).value == "keep this input"
        assert screen.query_one(MessageInput).has_focus
        assert "[hook block-submit] denied by policy" in str(
            screen.query(".error-message").last().render()
        )

    await hooks.aclose()
    assert provider.requests == []
    assert conversation.messages() == (Message(MessageRole.SYSTEM, SYSTEM_PROMPT),)


@pytest.mark.asyncio
async def test_hooks_command_reads_rules_from_app_bridge(tmp_path: Path) -> None:
    provider = ScriptedProvider([ScriptedReply(("unused",))])
    hooks = HookEngine(
        [
            HookRule(
                "configured",
                HookEvent.STOP,
                HookAction(
                    HookActionType.PROMPT,
                    prompt=HookPromptAction("remember"),
                ),
                only_once=True,
            )
        ],
        ["project-hooks.yaml"],
    )
    app = CodewrightApp(
        provider,
        Conversation(SYSTEM_PROMPT),
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        hook_engine=hooks,
    )

    async with app.run_test() as pilot:
        await pilot.press(*"/hooks", "enter")
        await pilot.pause()
        output = str(active_screen(app).query(".notice-message").last().render())
        assert "configured  Stop  prompt  [once]" in output
        assert "Loaded from: project-hooks.yaml" in output

    await hooks.aclose()
    assert provider.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        LLMAuthenticationError(),
        LLMNetworkError(),
        LLMTimeoutError(),
        LLMModelNotFoundError(),
        LLMRateLimitError(),
        LLMServiceError(),
    ],
    ids=("authentication", "network", "timeout", "model", "rate-limit", "service"),
)
async def test_provider_errors_are_safe_and_same_session_recovers(error: LLMError) -> None:
    provider = ScriptedProvider(
        [ScriptedReply(error=error), ScriptedReply(("recovered response",))]
    )
    conversation = Conversation(SYSTEM_PROMPT)
    app = CodewrightApp(provider, conversation, engine=permission_engine())

    async with app.run_test() as pilot:
        await pilot.press(*"first request", "enter")
        await pilot.pause()
        screen = active_screen(app)
        assert error.safe_message in str(screen.query_one(StatusWidget).render())
        assert screen.query_one(MessageInput).disabled is False
        assert [message.role for message in conversation.messages()] == [
            MessageRole.SYSTEM,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert conversation.messages()[-1].content == NOTICE_STREAM_ERROR

        await pilot.press(*"retry request", "enter")
        await pilot.pause()
        assert conversation.messages()[-1] == Message(MessageRole.ASSISTANT, "recovered response")


@pytest.mark.asyncio
async def test_non_streaming_configuration_uses_chat_interface() -> None:
    provider = ScriptedProvider([ScriptedReply(("complete non-streaming reply",))])
    conversation = Conversation(SYSTEM_PROMPT)
    app = CodewrightApp(
        provider,
        conversation,
        engine=permission_engine(),
        stream=False,
    )

    async with app.run_test() as pilot:
        await pilot.press(*"hello", "enter")
        await pilot.pause()

        screen = active_screen(app)
        assert screen.state_history == (
            ChatState.IDLE,
            ChatState.WAITING,
            ChatState.COMPLETED,
            ChatState.IDLE,
        )

    assert len(provider.chat_requests) == 1
    assert provider.stream_requests == []
    assert conversation.messages()[-1].content == "complete non-streaming reply"


@pytest.mark.asyncio
async def test_generation_cancellation_preserves_partial_view_and_closes_history() -> None:
    provider = ScriptedProvider(
        [ScriptedReply(("visible partial", "hidden remainder"), pause_after_delta=0)]
    )
    conversation = Conversation(SYSTEM_PROMPT)
    app = CodewrightApp(provider, conversation, engine=permission_engine())

    async with app.run_test() as pilot:
        await pilot.press(*"slow request", "enter")
        await provider.paused.wait()
        await pilot.press("ctrl+c")
        await pilot.pause()

        assistant = active_screen(app).query(ConversationMessage).last()
        assert "visible partial" in assistant.message_content
        assert "Response cancelled" in assistant.message_content
        assert active_screen(app).state is ChatState.IDLE

    assert provider.cancelled is True
    assert [message.role for message in conversation.messages()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert conversation.messages()[-1].content == NOTICE_CANCELLED


@pytest.mark.asyncio
async def test_new_app_starts_with_fresh_conversation() -> None:
    first_conversation = Conversation(SYSTEM_PROMPT)
    first_conversation.add_user("remember this value")
    first_conversation.add_assistant("remembered")

    second_provider = ScriptedProvider([ScriptedReply(("fresh session",))])
    second_conversation = Conversation(SYSTEM_PROMPT)
    second_app = CodewrightApp(
        second_provider,
        second_conversation,
        engine=permission_engine(),
    )

    async with second_app.run_test() as pilot:
        await pilot.press(*"what do you remember", "enter")
        await pilot.pause()

    assert second_provider.requests[0] == (
        Message(MessageRole.SYSTEM, SYSTEM_PROMPT),
        Message(MessageRole.USER, "what do you remember"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_keys", [("/", "e", "x", "i", "t", "enter"), ("ctrl+c",)])
async def test_idle_exit_paths_do_not_call_provider(
    exit_keys: Sequence[str],
    tmp_path: Path,
) -> None:
    provider = ScriptedProvider([ScriptedReply(("unused",))])
    lifecycle_log = tmp_path / "exit-hooks.log"
    hooks = HookEngine(
        [
            HookRule(
                event.value,
                event,
                HookAction(
                    HookActionType.SHELL,
                    shell=HookShellAction(
                        f"printf '{event.value}\\n' >> {shlex.quote(str(lifecycle_log))}"
                    ),
                ),
            )
            for event in (HookEvent.SESSION_START, HookEvent.SESSION_END)
        ],
        [],
    )
    app = CodewrightApp(
        provider,
        Conversation(SYSTEM_PROMPT),
        engine=permission_engine(tmp_path),
        working_directory=tmp_path,
        hook_engine=hooks,
    )

    async with app.run_test() as pilot:
        await pilot.press(*exit_keys)

    await hooks.aclose()
    assert provider.requests == []
    assert provider.closed is True
    assert lifecycle_log.read_text(encoding="utf-8").splitlines() == [
        "SessionStart",
        "SessionEnd",
    ]
    assert __version__ == "1.0.0"
