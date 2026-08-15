"""Tool for activating a locally discovered Skill in the current session."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

from codewright.skills.loader import SkillLoader
from codewright.tool.models import Result

if TYPE_CHECKING:
    from codewright.agent import Agent

_EXECUTION_AGENT: ContextVar[Agent | None] = ContextVar(
    "codewright_load_skill_agent",
    default=None,
)


class LoadSkillTool:
    """Load the latest valid body of one known Skill into Agent context."""

    name = "load_skill"
    read_only = True
    description = "Activate a known Skill SOP in the current session by name."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Exact name of the Skill to activate.",
            }
        },
        "required": ["name"],
        "additionalProperties": False,
    }

    def __init__(self, loader: SkillLoader) -> None:
        if not isinstance(loader, SkillLoader):
            raise TypeError("loader must be a SkillLoader")
        self._loader = loader
        self._agent: Agent | None = None

    def set_agent(self, agent: Agent) -> None:
        """Inject the Agent after both tool registry and Agent construction."""
        self._agent = agent

    async def execute(self, arguments_json: str) -> Result:
        """Validate a name, hot-reload it, and activate its latest valid body."""
        name_or_error = _parse_name(arguments_json)
        if isinstance(name_or_error, Result):
            return name_or_error
        skill = self._loader.get(name_or_error)
        if skill is None:
            return _error("unknown_skill", f"Unknown Skill: {name_or_error}")
        agent = _EXECUTION_AGENT.get() or self._agent
        if agent is None:
            return _error("not_initialized", "Skill activation is not initialized.")
        try:
            agent.activate_skill(skill.name, skill.prompt_body, skill.source_dir)
        except (TypeError, ValueError):
            return _error("activation_failed", "The Skill could not be activated.")
        return Result(content=f"Skill activated: {skill.name}")


def _parse_name(arguments_json: str) -> str | Result:
    try:
        arguments = json.loads(arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return _error("invalid_arguments", "Arguments must be a valid JSON object.")
    if not isinstance(arguments, dict):
        return _error("invalid_arguments", "Arguments must be a JSON object.")
    if set(arguments) != {"name"}:
        return _error("invalid_arguments", "Exactly one name argument is required.")
    name = arguments["name"]
    if not isinstance(name, str) or not name.strip():
        return _error("invalid_arguments", "name must be a non-empty string.")
    if name != name.strip():
        return _error("invalid_arguments", "name must not have surrounding whitespace.")
    return name


def _error(code: str, message: str) -> Result:
    return Result(content=message, is_error=True, error_code=code)


def bind_execution_agent(agent: Agent) -> Token[Agent | None]:
    """Bind load_skill to the Agent executing the current async tool batch."""
    return _EXECUTION_AGENT.set(agent)


def reset_execution_agent(token: Token[Agent | None]) -> None:
    """Restore the previous task-local load_skill Agent binding."""
    _EXECUTION_AGENT.reset(token)


__all__ = ["LoadSkillTool", "bind_execution_agent", "reset_execution_agent"]
