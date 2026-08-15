"""Parsing and deterministic matching for Codewright permission rules."""

import re
from dataclasses import dataclass, field

from codewright.permission.matcher import Matcher, compile_matcher, match_pattern
from codewright.permission.models import Decision

FRIENDLY_TOOLS = frozenset(
    {
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Agent",
        "TaskList",
        "TaskGet",
        "TaskStop",
        "SendMessage",
    }
)
_CANONICAL_TOOLS = {name.casefold(): name for name in FRIENDLY_TOOLS}
_MCP_COMPONENT = r"[A-Za-z0-9_-]+"
_MCP_TOOL_NAME = re.compile(rf"^mcp__{_MCP_COMPONENT}__{_MCP_COMPONENT}$")
_MCP_TOOL_SELECTOR = re.compile(rf"^mcp__{_MCP_COMPONENT}__[A-Za-z0-9_*-]+$")
_MAX_TOOL_NAME_LENGTH = 64


@dataclass(frozen=True, slots=True)
class Rule:
    """One parsed allow or deny rule."""

    tool: str
    matcher: Matcher | None
    allow: bool
    raw: str = ""


@dataclass(slots=True)
class RuleSet:
    """Rules from one configuration layer, with deny-before-allow matching."""

    allow: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)

    def match(self, friendly: str, target: str) -> tuple[Decision, bool]:
        """Return this layer's first effect and whether any rule matched."""
        for rule in self.deny:
            if _rule_matches(rule, friendly, target):
                return Decision.DENY, True
        for rule in self.allow:
            if _rule_matches(rule, friendly, target):
                return Decision.ALLOW, True
        return Decision.ALLOW, False


def parse_rule(value: str, *, allow: bool = False) -> tuple[Rule, bool]:
    """Parse a rule while preserving the established ``(Rule, valid)`` API."""
    parsed, error = parse_rule_detailed(value, allow=allow)
    if parsed is None:
        return Rule("", None, allow), False
    return parsed, error is None


def parse_rule_detailed(
    value: str,
    *,
    allow: bool = False,
) -> tuple[Rule | None, str | None]:
    """Parse Tool(pattern) and return a safe error suitable for stderr."""
    if not isinstance(value, str):
        return None, "rule must be a string"
    text = value.strip()
    if not text:
        return None, "rule must not be empty"

    opening = text.find("(")
    if opening == -1:
        if ")" in text:
            return None, "unexpected closing parenthesis"
        tool_text = text
        pattern = ""
    else:
        if not text.endswith(")"):
            return None, "missing closing parenthesis"
        if not _parentheses_balanced(text):
            return None, "unbalanced parentheses"
        tool_text = text[:opening].strip()
        pattern = text[opening + 1 : -1]

    if is_mcp_tool_selector(tool_text):
        if opening != -1:
            return None, "MCP selectors do not accept a target pattern"
        return Rule(tool_text, None, allow, tool_text), None

    tool = _CANONICAL_TOOLS.get(tool_text.casefold())
    if tool is None:
        return None, f"unknown tool {tool_text!r}"
    if not pattern:
        return Rule(tool, None, allow, pattern), None
    try:
        matcher = compile_matcher(pattern, is_command=tool == "Bash")
    except ValueError as error:
        return None, str(error)
    return Rule(tool, matcher, allow, pattern), None


def is_mcp_tool_name(value: object) -> bool:
    """Return whether value is one exact, provider-safe MCP tool name."""
    return (
        isinstance(value, str)
        and len(value) <= _MAX_TOOL_NAME_LENGTH
        and _MCP_TOOL_NAME.fullmatch(value) is not None
    )


def is_mcp_tool_selector(value: object) -> bool:
    """Return whether value is an exact or tool-segment-wildcard MCP selector."""
    return (
        isinstance(value, str)
        and len(value) <= _MAX_TOOL_NAME_LENGTH
        and _MCP_TOOL_SELECTOR.fullmatch(value) is not None
    )


def _rule_matches(rule: Rule, friendly: str, target: str) -> bool:
    if is_mcp_tool_selector(rule.tool):
        return (
            is_mcp_tool_name(friendly)
            and rule.matcher is None
            and match_pattern(rule.tool, friendly, path_mode=False)
        )
    if rule.tool != friendly:
        return False
    return rule.matcher is None or rule.matcher.match(target)


def _parentheses_balanced(value: str) -> bool:
    depth = 0
    escaped = False
    for character in value:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


__all__ = [
    "FRIENDLY_TOOLS",
    "Rule",
    "RuleSet",
    "is_mcp_tool_name",
    "is_mcp_tool_selector",
    "match_pattern",
    "parse_rule",
    "parse_rule_detailed",
]
