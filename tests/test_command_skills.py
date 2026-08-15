"""Tests for Skill management and dynamic slash commands."""

from pathlib import Path

import pytest

from codewright.command import Kind, NopUI, Registry, build_skill_commands, register_builtins
from codewright.command.builtin_skill import USAGE, handle_skill
from codewright.skills import SkillDef, SkillSource


def skill(
    tmp_path: Path,
    name: str,
    *,
    mode: str = "inline",
    body: str = "PRIVATE SOP BODY",
) -> SkillDef:
    path = (tmp_path / f"{name}.md").resolve()
    return SkillDef(
        name=name,
        description=f"Description for {name}",
        prompt_body=body,
        mode=mode,  # type: ignore[arg-type]
        model="secondary" if mode == "fork" else None,
        context="recent" if mode == "fork" else "full",
        source_path=path,
        source_dir=path.parent,
        is_directory=False,
        source=SkillSource.PROJECT,
    )


class SkillUI(NopUI):
    def __init__(self, skills: tuple[SkillDef, ...]) -> None:
        self.skills = skills
        self.printed: list[str] = []
        self.errors: list[str] = []
        self.inline_calls: list[tuple[str, str]] = []
        self.fork_calls: list[tuple[str, str]] = []
        self.reload_count = 0

    async def println(self, message: str) -> None:
        self.printed.append(message)

    async def error(self, message: str) -> None:
        self.errors.append(message)

    def list_skills(self) -> tuple[SkillDef, ...]:
        return self.skills

    def get_skill(self, name: str) -> SkillDef | None:
        return next((item for item in self.skills if item.name == name), None)

    async def reload_skills(self) -> tuple[SkillDef, ...]:
        self.reload_count += 1
        return self.skills

    async def run_inline_skill(self, name: str, args: str) -> None:
        self.inline_calls.append((name, args))

    async def run_fork_skill(self, name: str, args: str) -> None:
        self.fork_calls.append((name, args))


def test_register_builtins_includes_skill_command() -> None:
    registry = Registry()
    register_builtins(registry)

    command = registry.lookup("skill")
    assert registry.count() == 16
    assert command is not None
    assert command.kind is Kind.UI
    assert command.accepts_args is True


@pytest.mark.asyncio
async def test_skill_management_lists_and_inspects_metadata_without_body(
    tmp_path: Path,
) -> None:
    inline = skill(tmp_path, "review")
    fork = skill(tmp_path, "research", mode="fork")
    ui = SkillUI((fork, inline))

    await handle_skill(ui, "list")
    await handle_skill(ui, "info research")

    listing, info = ui.printed
    assert "/review" in listing and "mode=inline" in listing
    assert "/research" in listing and "source=project" in listing
    assert "name: research" in info
    assert "model: secondary" in info
    assert f"path: {fork.source_path}" in info
    assert f"resource root: {fork.source_dir}" in info
    assert "PRIVATE SOP BODY" not in listing + info


@pytest.mark.asyncio
async def test_skill_management_reload_unknown_and_usage(tmp_path: Path) -> None:
    ui = SkillUI((skill(tmp_path, "review"),))

    await handle_skill(ui, "reload")
    await handle_skill(ui, "info missing")
    await handle_skill(ui, "bad arguments")

    assert ui.reload_count == 1
    assert ui.printed == ["Skills 已重新加载：1 个。"]
    assert ui.errors == ["未知 Skill: missing", USAGE]


@pytest.mark.asyncio
async def test_dynamic_commands_preserve_kind_description_args_and_source(
    tmp_path: Path,
) -> None:
    ui = SkillUI(())
    commands = build_skill_commands(
        (skill(tmp_path, "inline"), skill(tmp_path, "forked", mode="fork")),
        {"help"},
    )
    by_name = {command.name: command for command in commands}

    assert by_name["inline"].kind is Kind.PROMPT
    assert by_name["forked"].kind is Kind.UI
    assert all(command.accepts_args and command.source == "skill" for command in commands)
    assert by_name["inline"].description.endswith(" [skill]")
    await by_name["inline"].handler(ui, "one  two")
    await by_name["forked"].handler(ui, "three")
    assert ui.inline_calls == [("inline", "one  two")]
    assert ui.fork_calls == [("forked", "three")]


def test_builtin_conflict_is_skipped_but_skill_remains_available(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    conflicting = skill(tmp_path, "help")
    normal = skill(tmp_path, "review-project")

    commands = build_skill_commands((conflicting, normal), {"help", "h"})

    assert [command.name for command in commands] == ["review-project"]
    assert "Skill command conflicts with built-in name=help" in caplog.text
    assert conflicting.prompt_body == "PRIVATE SOP BODY"
