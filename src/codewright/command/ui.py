"""Minimal UI boundary consumed by slash-command handlers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from codewright.hook import Rule
from codewright.permission import Mode
from codewright.skills.models import SkillDef


@dataclass(frozen=True, slots=True)
class WorktreeSummary:
    name: str
    path: str
    branch: str
    manual: bool
    active: bool = False


class WorktreeAccessor(Protocol):
    async def create(self, name: str) -> WorktreeSummary: ...

    def list(self) -> tuple[WorktreeSummary, ...]: ...

    async def enter(self, name: str) -> str: ...

    async def exit(self, *, remove: bool, discard: bool) -> str: ...

    async def remove(self, name: str, *, discard: bool) -> None: ...


class TeamAccessor(Protocol):
    @property
    def active_team(self) -> object | None: ...

    def get(self, name: str) -> object | None: ...

    def list(self) -> tuple[object, ...]: ...

    def use(self, name: str) -> object: ...

    async def delete(
        self, name: str, *, force: bool = False, purge_sessions: bool = False
    ) -> object: ...

    async def kill_member(self, team: str, member: str) -> object: ...


@runtime_checkable
class UI(Protocol):
    """Framework-neutral capabilities needed by built-in commands."""

    async def println(self, message: str) -> None: ...

    async def error(self, message: str) -> None: ...

    def idle(self) -> bool: ...

    @property
    def mode(self) -> Mode: ...

    async def set_mode(self, mode: Mode) -> None: ...

    def usage(self) -> tuple[int, int]: ...

    def model_name(self) -> str: ...

    def cwd(self) -> str: ...

    def worktree_accessor(self) -> WorktreeAccessor | None: ...

    def team_accessor(self) -> TeamAccessor | None: ...

    def tool_count(self) -> int: ...

    def hook_sources(self) -> list[str]: ...

    def hook_rules(self) -> list[Rule]: ...

    def memory_files(self) -> tuple[list[str], list[str]]: ...

    def session_id(self) -> str: ...

    def session_path(self) -> str: ...

    async def inject_and_send(self, display_label: str, preset_prompt: str) -> None: ...

    async def request_exit(self) -> None: ...

    async def force_compact(self) -> None: ...

    async def open_resume_menu(self) -> None: ...

    async def clear_and_new_session(self) -> None: ...

    def list_skills(self) -> tuple[SkillDef, ...]: ...

    def get_skill(self, name: str) -> SkillDef | None: ...

    async def reload_skills(self) -> tuple[SkillDef, ...]: ...

    async def run_inline_skill(self, name: str, args: str) -> None: ...

    async def run_fork_skill(self, name: str, args: str) -> None: ...


class NopUI:
    """Safe no-op implementation used by command-domain tests."""

    async def println(self, message: str) -> None:
        del message

    async def error(self, message: str) -> None:
        del message

    def idle(self) -> bool:
        return True

    @property
    def mode(self) -> Mode:
        return Mode.DEFAULT

    async def set_mode(self, mode: Mode) -> None:
        del mode

    def usage(self) -> tuple[int, int]:
        return 0, 0

    def model_name(self) -> str:
        return ""

    def cwd(self) -> str:
        return ""

    def worktree_accessor(self) -> WorktreeAccessor | None:
        return None

    def team_accessor(self) -> TeamAccessor | None:
        return None

    def tool_count(self) -> int:
        return 0

    def hook_sources(self) -> list[str]:
        return []

    def hook_rules(self) -> list[Rule]:
        return []

    def memory_files(self) -> tuple[list[str], list[str]]:
        return [], []

    def session_id(self) -> str:
        return ""

    def session_path(self) -> str:
        return ""

    async def inject_and_send(self, display_label: str, preset_prompt: str) -> None:
        del display_label, preset_prompt

    async def request_exit(self) -> None:
        return None

    async def force_compact(self) -> None:
        return None

    async def open_resume_menu(self) -> None:
        return None

    async def clear_and_new_session(self) -> None:
        return None

    def list_skills(self) -> tuple[SkillDef, ...]:
        return ()

    def get_skill(self, name: str) -> SkillDef | None:
        del name
        return None

    async def reload_skills(self) -> tuple[SkillDef, ...]:
        return ()

    async def run_inline_skill(self, name: str, args: str) -> None:
        del name, args

    async def run_fork_skill(self, name: str, args: str) -> None:
        del name, args


__all__ = ["NopUI", "TeamAccessor", "UI", "WorktreeAccessor", "WorktreeSummary"]
