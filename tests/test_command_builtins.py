"""Behavior tests for Codewright's built-in slash commands."""

from __future__ import annotations

import pytest

from codewright.command import UI, Kind, NopUI, Registry, register_builtins
from codewright.command.builtin_prompt import REVIEW_DIRECTIVE
from codewright.hook import Action, ActionType, Event, PromptAction
from codewright.hook import Rule as HookRule
from codewright.permission import Mode
from codewright.prompt import EXECUTE_DIRECTIVE


class RecordingUI(NopUI):
    def __init__(self) -> None:
        self.printed: list[str] = []
        self.errors: list[str] = []
        self.modes: list[Mode] = []
        self.injected: list[tuple[str, str]] = []
        self.actions: list[str] = []
        self.current_mode = Mode.ACCEPT_EDITS
        self.sources: list[str] = []
        self.rules: list[HookRule] = []

    async def println(self, message: str) -> None:
        self.printed.append(message)

    async def error(self, message: str) -> None:
        self.errors.append(message)

    @property
    def mode(self) -> Mode:
        return self.current_mode

    async def set_mode(self, mode: Mode) -> None:
        self.current_mode = mode
        self.modes.append(mode)

    def usage(self) -> tuple[int, int]:
        return 12, 7

    def model_name(self) -> str:
        return "test-model"

    def cwd(self) -> str:
        return "/workspace/codewright"

    def tool_count(self) -> int:
        return 6

    def hook_sources(self) -> list[str]:
        return list(self.sources)

    def hook_rules(self) -> list[HookRule]:
        return list(self.rules)

    def memory_files(self) -> tuple[list[str], list[str]]:
        return ["MEMORY.md", "project.md"], ["preference.md"]

    def session_id(self) -> str:
        return "20260812-120000-abcd"

    def session_path(self) -> str:
        return "/workspace/.codewright/sessions/id/conversation.jsonl"

    async def inject_and_send(self, display_label: str, preset_prompt: str) -> None:
        self.injected.append((display_label, preset_prompt))

    async def request_exit(self) -> None:
        self.actions.append("exit")

    async def force_compact(self) -> None:
        self.actions.append("compact")

    async def open_resume_menu(self) -> None:
        self.actions.append("resume")

    async def clear_and_new_session(self) -> None:
        self.actions.append("clear")


def builtins() -> Registry:
    registry = Registry()
    register_builtins(registry)
    return registry


def test_register_builtins_has_expected_sorted_commands() -> None:
    registry = builtins()

    assert [command.name for command in registry.visible()] == [
        "clear",
        "compact",
        "do",
        "exit",
        "help",
        "hooks",
        "memory",
        "permission",
        "plan",
        "resume",
        "review",
        "session",
        "skill",
        "status",
        "team",
        "worktree",
    ]
    assert registry.count() == 16
    assert all(not command.aliases and not command.hidden for command in registry.visible())


def test_builtin_kinds_match_contract() -> None:
    registry = builtins()
    by_name = {command.name: command.kind for command in registry.visible()}

    assert {name for name, kind in by_name.items() if kind is Kind.LOCAL} == {
        "help",
        "hooks",
        "memory",
        "permission",
        "session",
        "status",
    }
    assert {name for name, kind in by_name.items() if kind is Kind.PROMPT} == {
        "do",
        "review",
    }


@pytest.mark.asyncio
async def test_all_handlers_satisfy_ui_protocol_and_run_on_noop() -> None:
    registry = builtins()
    ui = NopUI()
    assert isinstance(ui, UI)

    for command in registry.visible():
        await command.handler(ui, "")


@pytest.mark.asyncio
async def test_help_and_status_are_registry_driven_and_ordered() -> None:
    registry = builtins()
    ui = RecordingUI()

    help_command = registry.lookup("help")
    status_command = registry.lookup("status")
    assert help_command is not None and status_command is not None
    await help_command.handler(ui, "")
    await status_command.handler(ui, "")

    help_lines = ui.printed[0].splitlines()
    assert len(help_lines) == 16
    assert [line.split()[0] for line in help_lines] == [
        f"/{command.name}" for command in registry.visible()
    ]
    status_lines = ui.printed[1].splitlines()
    assert [line.split(":", 1)[0].strip() for line in status_lines] == [
        "Mode",
        "Tokens",
        "Tools",
        "Memories",
        "Model",
        "Directory",
    ]
    assert "12 in / 7 out" in ui.printed[1]


@pytest.mark.asyncio
async def test_memory_permission_and_session_output_observable_values() -> None:
    registry = builtins()
    ui = RecordingUI()

    for name in ("memory", "permission", "session"):
        command = registry.lookup(name)
        assert command is not None
        await command.handler(ui, "")

    assert ui.printed[0].splitlines() == [
        "[project] MEMORY.md",
        "[project] project.md",
        "[user] preference.md",
    ]
    assert ui.printed[1] == "acceptEdits"
    assert "Session: 20260812-120000-abcd" in ui.printed[2]
    assert "Path: /workspace/.codewright" in ui.printed[2]


@pytest.mark.asyncio
async def test_ui_and_prompt_handlers_delegate_exact_actions() -> None:
    registry = builtins()
    ui = RecordingUI()

    for name in ("exit", "compact", "resume", "clear", "plan", "do", "review"):
        command = registry.lookup(name)
        assert command is not None
        await command.handler(ui, "")

    assert ui.actions == ["exit", "compact", "resume", "clear"]
    assert ui.modes == [Mode.PLAN, Mode.DEFAULT]
    assert ui.injected == [
        ("/do", EXECUTE_DIRECTIVE),
        ("/review", REVIEW_DIRECTIVE),
    ]
    assert ui.printed[:2] == [
        "已进入计划模式（仅使用只读工具）。",
        "已退出计划模式，开始执行上文计划。",
    ]


@pytest.mark.asyncio
async def test_empty_memory_output_is_friendly() -> None:
    registry = builtins()
    ui = RecordingUI()
    ui.memory_files = lambda: ([], [])  # type: ignore[method-assign]
    command = registry.lookup("memory")
    assert command is not None

    await command.handler(ui, "")

    assert ui.printed == ["无已加载的记忆文件"]


@pytest.mark.asyncio
async def test_hooks_lists_grouped_rules_flags_and_sources() -> None:
    registry = builtins()
    ui = RecordingUI()
    ui.sources = ["project-hooks.yaml", "user-hooks.yaml"]
    ui.rules = [
        HookRule(
            "start",
            Event.SESSION_START,
            Action(ActionType.PROMPT, prompt=PromptAction("one")),
            only_once=True,
        ),
        HookRule(
            "stop",
            Event.STOP,
            Action(ActionType.PROMPT, prompt=PromptAction("two")),
            asyncio_mode=True,
        ),
        HookRule(
            "start-second",
            Event.SESSION_START,
            Action(ActionType.PROMPT, prompt=PromptAction("three")),
        ),
    ]
    command = registry.lookup("hooks")
    assert command is not None

    await command.handler(ui, "")

    assert ui.printed == [
        "  start  SessionStart  prompt  [once]\n"
        "  start-second  SessionStart  prompt\n"
        "  stop  Stop  prompt  [async]\n"
        "Loaded from: project-hooks.yaml, user-hooks.yaml"
    ]


@pytest.mark.asyncio
async def test_hooks_empty_output_is_friendly() -> None:
    registry = builtins()
    ui = RecordingUI()
    command = registry.lookup("hooks")
    assert command is not None

    await command.handler(ui, "")

    assert ui.printed == ["No hooks loaded."]
