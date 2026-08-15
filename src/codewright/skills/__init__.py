"""Public Skill parsing, loading, and activation primitives."""

from typing import TYPE_CHECKING, Any

from codewright.skills.install import InstallResult, SkillInstaller, SkillInstallError
from codewright.skills.loader import PROJECT_SKILLS_DIR, USER_SKILLS_DIR, SkillLoader
from codewright.skills.models import (
    ActiveEntry,
    ActiveSkills,
    SkillContext,
    SkillDef,
    SkillMode,
    SkillSource,
)
from codewright.skills.parser import (
    SkillParseError,
    parse_frontmatter,
    parse_skill_file,
    substitute_arguments,
)

if TYPE_CHECKING:
    from codewright.skills.executor import ForkResult, SkillExecutionError, SkillExecutor


def __getattr__(name: str) -> Any:
    """Load executor symbols lazily to keep Agent/Skill imports acyclic."""
    if name in {"ForkResult", "SkillExecutionError", "SkillExecutor"}:
        from codewright.skills import executor

        return getattr(executor, name)
    raise AttributeError(name)


__all__ = [
    "PROJECT_SKILLS_DIR",
    "USER_SKILLS_DIR",
    "ActiveEntry",
    "ActiveSkills",
    "ForkResult",
    "InstallResult",
    "SkillContext",
    "SkillDef",
    "SkillExecutionError",
    "SkillInstallError",
    "SkillInstaller",
    "SkillExecutor",
    "SkillLoader",
    "SkillMode",
    "SkillParseError",
    "SkillSource",
    "parse_frontmatter",
    "parse_skill_file",
    "substitute_arguments",
]
