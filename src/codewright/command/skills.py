"""Construction of dynamic slash commands backed by loaded Skills."""

import logging
from collections.abc import Collection, Sequence

from codewright.command.models import Command, Handler, Kind
from codewright.command.ui import UI
from codewright.skills.models import SkillDef, SkillMode

logger = logging.getLogger(__name__)


def build_skill_commands(
    skills: Sequence[SkillDef],
    reserved_names: Collection[str],
) -> tuple[Command, ...]:
    """Build dynamic commands while preserving every reserved built-in name."""
    reserved = {name.casefold() for name in reserved_names}
    commands: list[Command] = []
    for skill in skills:
        if skill.name.casefold() in reserved:
            logger.warning("Skill command conflicts with built-in name=%s", skill.name)
            continue
        commands.append(
            Command(
                skill.name,
                f"{skill.description} [skill]",
                Kind.PROMPT if skill.mode == "inline" else Kind.UI,
                _handler(skill.name, skill.mode),
                accepts_args=True,
                source="skill",
            )
        )
    return tuple(commands)


def _handler(name: str, mode: SkillMode) -> Handler:
    async def handle(ui: UI, args: str) -> None:
        if mode == "inline":
            await ui.run_inline_skill(name, args)
        else:
            await ui.run_fork_skill(name, args)

    return handle


__all__ = ["build_skill_commands"]
