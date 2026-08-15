"""Public slash-command domain primitives."""

from codewright.command.builtins import register_builtins
from codewright.command.dispatch import Invocation, parse, parse_invocation
from codewright.command.models import Command, CommandSource, Handler, Kind
from codewright.command.registry import Registry
from codewright.command.skills import build_skill_commands
from codewright.command.ui import UI, NopUI, TeamAccessor, WorktreeAccessor, WorktreeSummary

__all__ = [
    "Command",
    "CommandSource",
    "Handler",
    "Invocation",
    "Kind",
    "NopUI",
    "TeamAccessor",
    "WorktreeAccessor",
    "WorktreeSummary",
    "Registry",
    "UI",
    "build_skill_commands",
    "parse",
    "parse_invocation",
    "register_builtins",
]
