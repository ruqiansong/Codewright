"""Tests for deterministic, modular system prompt assembly."""

import asyncio
from collections.abc import Callable

import pytest

from codewright.prompt import (
    EXECUTE_DIRECTIVE,
    PLAN_MODE_REMINDER,
    SYSTEM_PROMPT,
    Environment,
    PromptModule,
    assemble_system,
    build_system_prompt,
    fixed_modules,
    gather_environment,
    optional_modules,
    plan_reminder,
    system_reminder,
)


def test_default_modules_have_expected_order_and_names() -> None:
    modules = fixed_modules()

    assert [module.priority for module in modules] == [10, 20, 30, 40, 50, 60, 70]
    assert [module.name for module in modules] == [
        "identity",
        "system_constraints",
        "task_mode",
        "action_execution",
        "tool_usage",
        "tone",
        "text_output",
    ]


def test_assembly_sorts_modules_and_skips_empty_content() -> None:
    modules = (
        PromptModule("last", 30, "Last"),
        PromptModule("empty", 20, "  "),
        PromptModule("first", 10, "First"),
    )

    assert assemble_system(modules) == "First\n\nLast"


def test_new_module_is_inserted_without_changing_assembly_logic() -> None:
    modules = (*fixed_modules(), PromptModule("extension", 45, "Extension marker"))

    assembled = assemble_system(modules)

    assert assembled.index("actual local action") < assembled.index("Extension marker")
    assert assembled.index("Extension marker") < assembled.index("Prefer read_file")


def test_optional_slots_are_reserved_and_do_not_render() -> None:
    modules = optional_modules()

    assert [module.name for module in modules] == [
        "custom_instructions",
        "skills_catalog",
        "long_term_memory",
    ]
    assert all(not module.content for module in modules)
    assert assemble_system(modules) == ""


def test_optional_slots_render_instruction_and_memory_in_priority_order() -> None:
    modules = optional_modules("Project rules", "Remembered facts")

    assert modules[0].content == "Project rules"
    assert modules[2].content == "Remembered facts"
    assert assemble_system(modules) == "Project rules\n\nRemembered facts"


def test_default_system_prompt_is_deterministic_and_compatible() -> None:
    first = build_system_prompt()
    second = build_system_prompt()

    assert first == second == SYSTEM_PROMPT
    assert "Codewright" in first
    assert "read, write, and uniquely edit files" in first
    assert "execute shell" in first
    assert "across multiple steps" in first
    assert "permission decisions and sandbox boundaries" in first
    assert "no permission confirmation" not in first
    assert "no permission confirmation, sandbox, MCP, Memory" not in first
    assert "Never invent tool" in first


def test_tool_rules_are_reinforced_in_default_prompt() -> None:
    prompt = build_system_prompt()

    assert "Prefer read_file, glob, and grep" in prompt
    assert "Always read a file before editing it" in prompt


def test_build_system_prompt_appends_dynamic_modules_after_configured_base() -> None:
    prompt = build_system_prompt(
        "Project instruction",
        "Long-term fact",
        base_prompt="Configured identity",
    )

    assert prompt == "Configured identity\n\nProject instruction\n\nLong-term fact"
    assert "You are Codewright" not in prompt


def test_build_system_prompt_skips_empty_dynamic_modules() -> None:
    assert build_system_prompt("", "", "Custom base") == "Custom base"
    assert build_system_prompt() == SYSTEM_PROMPT


def test_dynamic_prompt_arguments_are_typed() -> None:
    with pytest.raises(TypeError):
        optional_modules(None, "")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_system_prompt(base_prompt=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "error_type"),
    [
        (lambda: PromptModule("", 10, "content"), ValueError),
        (lambda: PromptModule("name", True, "content"), TypeError),
        (lambda: PromptModule("name", 10, None), TypeError),
    ],
)
def test_prompt_module_rejects_invalid_fields(
    factory: Callable[[], PromptModule], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        factory()


def test_environment_render_uses_stable_order_and_skips_missing_values() -> None:
    environment = Environment(
        working_dir="/workspace/codewright",
        platform="linux",
        date="2026-08-10",
        git_status="",
        version="0.5.0",
        model="example-model",
    )

    assert environment.render().splitlines() == [
        "Environment:",
        "Working directory: /workspace/codewright",
        "Platform: linux",
        "Date: 2026-08-10",
        "Codewright version: 0.5.0",
        "Model: example-model",
    ]


@pytest.mark.asyncio
async def test_gather_environment_degrades_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unavailable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise FileNotFoundError

    monkeypatch.setattr("codewright.prompt.environment.asyncio.create_subprocess_exec", unavailable)

    environment = await gather_environment("0.5.0", "example-model")

    assert environment.working_dir
    assert environment.platform
    assert environment.date
    assert environment.git_status == ""
    assert environment.version == "0.5.0"
    assert environment.model == "example-model"


class _SlowGitProcess:
    returncode: int | None = None

    def __init__(self) -> None:
        self.killed = False
        self.communicate_calls = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicate_calls += 1
        if self.communicate_calls == 1:
            await asyncio.sleep(10)
        self.returncode = -9
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.mark.asyncio
async def test_gather_environment_terminates_timed_out_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _SlowGitProcess()

    async def create_process(*args: object, **kwargs: object) -> _SlowGitProcess:
        del args, kwargs
        return process

    monkeypatch.setattr("codewright.prompt.environment.GIT_TIMEOUT_SECONDS", 0.001)
    monkeypatch.setattr(
        "codewright.prompt.environment.asyncio.create_subprocess_exec", create_process
    )

    environment = await gather_environment("0.5.0", "example-model")

    assert environment.git_status == ""
    assert process.killed is True
    assert process.communicate_calls == 2


def test_reminder_builders_preserve_compatibility_and_wrap_dynamic_messages() -> None:
    full = plan_reminder(full=True)
    concise = plan_reminder(full=False)

    assert full == system_reminder(PLAN_MODE_REMINDER)
    assert full.startswith("<system-reminder>\n")
    assert full.endswith("\n</system-reminder>")
    assert "PLAN MODE" in full
    assert "PLAN MODE" in concise
    assert full != concise
    assert EXECUTE_DIRECTIVE == "请按上面的计划开始执行。"


def test_system_reminder_rejects_empty_body() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        system_reminder("  ")
