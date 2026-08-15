"""Public prompt API for Codewright."""

from codewright.prompt.environment import Environment, gather_environment
from codewright.prompt.modules import (
    PromptModule,
    assemble_system,
    build_system_prompt,
    fixed_modules,
    optional_modules,
)
from codewright.prompt.reminder import (
    EXECUTE_DIRECTIVE,
    PLAN_MODE_REMINDER,
    PLAN_MODE_REMINDER_CONCISE,
    plan_reminder,
    system_reminder,
)
from codewright.prompt.skills import render_active_skills, render_skill_catalog

SYSTEM_PROMPT = build_system_prompt()

__all__ = [
    "EXECUTE_DIRECTIVE",
    "Environment",
    "PLAN_MODE_REMINDER",
    "PLAN_MODE_REMINDER_CONCISE",
    "PromptModule",
    "SYSTEM_PROMPT",
    "assemble_system",
    "build_system_prompt",
    "fixed_modules",
    "gather_environment",
    "optional_modules",
    "plan_reminder",
    "render_active_skills",
    "render_skill_catalog",
    "system_reminder",
]
