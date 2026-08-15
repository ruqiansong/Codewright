"""Apply compiled permission Matchers to nested Hook payload fields."""

from __future__ import annotations

import json
from typing import Any

from codewright.hook.rule import CombineMode, Condition, Payload


def get_by_path(payload: Payload, path: str) -> str:
    """Read one dot-separated mapping path and normalize it to a string."""
    current: Any = payload
    for component in path.split("."):
        if not component or not isinstance(current, dict) or component not in current:
            return ""
        current = current[component]
        if current is None:
            return ""
    if isinstance(current, str):
        return current
    if isinstance(current, (bool, int, float)):
        return str(current)
    return json.dumps(current, sort_keys=True, separators=(",", ":"))


def eval_condition(condition: Condition | None, payload: Payload) -> bool:
    """Evaluate one precompiled all-of or any-of condition."""
    if condition is None:
        return True
    outcomes = (atom.matcher.match(get_by_path(payload, atom.field)) for atom in condition.atoms)
    if condition.mode is CombineMode.ALL_OF:
        return all(outcomes)
    return any(outcomes)


__all__ = ["eval_condition", "get_by_path"]
