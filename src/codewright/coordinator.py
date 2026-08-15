"""Coordinator-mode policy, prompt, and double-lock feature helper."""

from __future__ import annotations

import os
from collections.abc import Mapping

from codewright.config import Config

COORDINATOR_ALLOWED_TOOLS = frozenset(
    {
        "Agent",
        "TeamCreate",
        "TeamDelete",
        "TeamTaskCreate",
        "TeamTaskGet",
        "TeamTaskList",
        "TeamTaskUpdate",
        "TeamSendMessage",
        "read_file",
        "glob",
        "grep",
        "bash",
        "load_skill",
    }
)
COORDINATOR_PROMPT_SUFFIX = """

Coordinator mode is active. Coordinate work through Agent Teams and shared tasks.
Inspect the repository as needed, but delegate implementation instead of editing files directly.
""".strip()
_TRUTHY = {"1", "true", "yes", "on"}


def coordinator_enabled(
    config: Config,
    environ: Mapping[str, str] | None = None,
) -> bool:
    values = os.environ if environ is None else environ
    enabled = values.get("CODEWRIGHT_COORDINATOR_MODE", "").casefold() in _TRUTHY
    return config.enable_coordinator_mode and enabled


def coordinator_allowed_tools(registry_names: tuple[str, ...]) -> frozenset[str]:
    return frozenset(name for name in registry_names if name in COORDINATOR_ALLOWED_TOOLS)


__all__ = [
    "COORDINATOR_ALLOWED_TOOLS",
    "COORDINATOR_PROMPT_SUFFIX",
    "coordinator_allowed_tools",
    "coordinator_enabled",
]
