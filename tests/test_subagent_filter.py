"""Tests for stable child-Agent tool filtering."""

from codewright.agent.filter import SUBAGENT_META_TOOLS, filter_tool_names
from codewright.subagent import Source

REGISTERED = (
    "write_file",
    "Agent",
    "read_file",
    "TaskList",
    "custom_tool",
    "mcp__docs__search",
    "load_skill",
    "SendMessage",
    "TeamCreate",
    "TeamTaskCreate",
    "TeamTaskGet",
    "TeamTaskList",
    "TeamTaskUpdate",
    "TeamSendMessage",
)


def test_all_child_agents_remove_meta_tools_and_preserve_order() -> None:
    result = filter_tool_names(REGISTERED, source=Source.BUILTIN)

    assert result == (
        "write_file",
        "read_file",
        "custom_tool",
        "mcp__docs__search",
        "load_skill",
    )
    assert SUBAGENT_META_TOOLS.isdisjoint(result)


def test_background_keeps_base_tools_and_mcp_prefix_only() -> None:
    result = filter_tool_names(REGISTERED, source=Source.PROJECT, background=True)

    assert result == ("write_file", "read_file", "mcp__docs__search", "load_skill")


def test_role_whitelist_then_blacklist_gives_blacklist_priority() -> None:
    result = filter_tool_names(
        REGISTERED,
        source=Source.USER,
        tools=("read_file", "write_file", "Agent"),
        disallowed_tools=("write_file",),
    )

    assert result == ("read_file",)


def test_fork_uses_the_same_meta_tool_boundary() -> None:
    result = filter_tool_names(
        REGISTERED,
        source=Source.BUILTIN,
        disallowed_tools=("custom_tool",),
    )

    assert result == ("write_file", "read_file", "mcp__docs__search", "load_skill")


def test_team_member_gets_only_team_collaboration_tools_not_meta_tools() -> None:
    result = filter_tool_names(REGISTERED, source=Source.BUILTIN, team_member=True)

    assert "Agent" not in result
    assert "TaskList" not in result
    assert "SendMessage" not in result
    assert "TeamCreate" not in result
    assert result[-5:] == (
        "TeamTaskCreate",
        "TeamTaskGet",
        "TeamTaskList",
        "TeamTaskUpdate",
        "TeamSendMessage",
    )
