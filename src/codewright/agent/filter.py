"""Stable tool filtering rules for child Agent instances."""

from codewright.subagent import Source

TEAM_MANAGEMENT_TOOLS = frozenset({"TeamCreate", "TeamDelete"})
TEAM_COLLABORATION_TOOLS = frozenset(
    {"TeamTaskCreate", "TeamTaskGet", "TeamTaskList", "TeamTaskUpdate", "TeamSendMessage"}
)
TEAM_TOOLS = TEAM_MANAGEMENT_TOOLS | TEAM_COLLABORATION_TOOLS
SUBAGENT_META_TOOLS = frozenset(
    {"Agent", "TaskList", "TaskGet", "TaskStop", "SendMessage", *TEAM_TOOLS}
)
CUSTOM_AGENT_DISALLOWED_TOOLS: frozenset[str] = frozenset()
BACKGROUND_BASE_TOOLS = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "bash",
        "load_skill",
        "install_skill",
    }
)
_CUSTOM_SOURCES = frozenset({Source.USER, Source.PROJECT, Source.PLUGIN})


def filter_tool_names(
    registry_names: tuple[str, ...],
    *,
    source: Source,
    tools: tuple[str, ...] = (),
    disallowed_tools: tuple[str, ...] = (),
    background: bool = False,
    team_member: bool = False,
) -> tuple[str, ...]:
    """Filter registered names in policy order while preserving registration order."""
    if not isinstance(registry_names, tuple) or any(
        not isinstance(name, str) or not name for name in registry_names
    ):
        raise ValueError("registry_names must be a tuple of non-empty strings")
    if not isinstance(source, Source):
        raise TypeError("source must be a Source")
    if not isinstance(tools, tuple) or any(not isinstance(name, str) for name in tools):
        raise TypeError("tools must be a tuple of strings")
    if not isinstance(disallowed_tools, tuple) or any(
        not isinstance(name, str) for name in disallowed_tools
    ):
        raise TypeError("disallowed_tools must be a tuple of strings")
    if not isinstance(background, bool):
        raise TypeError("background must be a boolean")
    if not isinstance(team_member, bool):
        raise TypeError("team_member must be a boolean")

    allowed = [name for name in registry_names if name not in SUBAGENT_META_TOOLS]
    if source in _CUSTOM_SOURCES:
        allowed = [name for name in allowed if name not in CUSTOM_AGENT_DISALLOWED_TOOLS]
    if background:
        allowed = [
            name for name in allowed if name in BACKGROUND_BASE_TOOLS or name.startswith("mcp__")
        ]
    if tools:
        whitelist = frozenset(tools)
        allowed = [name for name in allowed if name in whitelist]
    blacklist = frozenset(disallowed_tools)
    result = [name for name in allowed if name not in blacklist]
    if team_member:
        result.extend(
            name
            for name in registry_names
            if name in TEAM_COLLABORATION_TOOLS and name not in blacklist
        )
    return tuple(result)


__all__ = [
    "BACKGROUND_BASE_TOOLS",
    "CUSTOM_AGENT_DISALLOWED_TOOLS",
    "SUBAGENT_META_TOOLS",
    "TEAM_COLLABORATION_TOOLS",
    "TEAM_MANAGEMENT_TOOLS",
    "TEAM_TOOLS",
    "filter_tool_names",
]
