"""Heuristic, non-exhaustive dangerous-command denylist that cannot be configured away."""

import re
from re import Pattern

_COMMAND_START = r"(?:^|[;&|\n]\s*)"
_OPTIONAL_SUDO = r"(?:sudo\s+)?"
_BLACKLIST: tuple[Pattern[str], ...] = (
    re.compile(
        _COMMAND_START
        + _OPTIONAL_SUDO
        + r"rm\s+(?:(?:-[A-Za-z]*[rRfF][A-Za-z]*|--recursive|--force)\s+)+"
        + r"(?:--\s+)?(?:/|/\*|~|\$HOME)(?:\s|$|[;&|])"
    ),
    re.compile(_COMMAND_START + _OPTIONAL_SUDO + r"dd\s+[^\n;&|]*\bof\s*=\s*/dev/"),
    re.compile(_COMMAND_START + _OPTIONAL_SUDO + r"mkfs(?:\.[A-Za-z0-9_-]+)?\b"),
    re.compile(r":\s*\(\s*\)\s*\{[^}]*:\s*\|\s*:\s*&[^}]*\}\s*;?\s*:"),
    re.compile(r"(?:^|[^>])>\s*/dev/(?:sd[a-z]|hd[a-z]|nvme\d+n\d+|disk\d*)\b"),
    re.compile(_COMMAND_START + _OPTIONAL_SUDO + r"chmod\s+-R\s+0?777\s+(?:/|/\*)(?:\s|$|[;&|])"),
)


def hits_blacklist(command: str) -> bool:
    """Return whether a raw shell command matches a built-in dangerous pattern."""
    if not isinstance(command, str):
        raise TypeError("command must be a string")
    return any(pattern.search(command) is not None for pattern in _BLACKLIST)


__all__ = ["hits_blacklist"]
