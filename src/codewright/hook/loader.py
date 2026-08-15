"""Load and validate layered lifecycle Hook YAML configuration."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path

import yaml

from codewright.hook.engine import Engine
from codewright.hook.event import is_blocking, parse_event
from codewright.hook.rule import (
    Action,
    ActionType,
    AtomCondition,
    CombineMode,
    Condition,
    HttpAction,
    PromptAction,
    Rule,
    ShellAction,
    SubagentAction,
)
from codewright.permission.matcher import (
    ExactMatcher,
    GlobMatcher,
    Matcher,
    NotMatcher,
    RegexMatcher,
)

_DURATION = re.compile(r"\d+(?:\.\d+)?([smh]?)")
_DURATION_FACTORS = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}
_MISSING = object()


class _RuleError(ValueError):
    """One invalid rule that may be skipped without rejecting its file."""


def load(project_root: str | Path) -> Engine:
    """Load project and user Hook files, logging and skipping invalid input."""
    project_path = Path(project_root) / ".codewright" / "hooks.yaml"
    candidates = [project_path]
    try:
        candidates.append(Path.home() / ".codewright" / "hooks.yaml")
    except (OSError, RuntimeError) as error:
        _warn(f"load user hooks failed: {error}")

    rules: list[Rule] = []
    sources: list[str] = []
    names: set[str] = set()
    for path in candidates:
        raw_rules = _load_file(path)
        if raw_rules is None:
            continue
        sources.append(str(path))
        for index, raw in enumerate(raw_rules, start=1):
            if not isinstance(raw, Mapping):
                _warn(f"hook at {path}:{index}: definition must be a mapping, skipped")
                continue
            try:
                rule = _compile_rule(path, index, raw)
            except _RuleError as error:
                _warn(f"hook at {path}:{index}: {error}, skipped")
                continue
            if rule.name in names:
                _warn(f'hook "{rule.name}": duplicate name, skipped')
                continue
            names.add(rule.name)
            rules.append(rule)
    return Engine(rules, sources)


def _load_file(path: Path) -> list[object] | None:
    try:
        if not path.exists():
            return None
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        _warn(f"load hooks {path} failed: invalid YAML: {error}")
        return None
    except (OSError, UnicodeError) as error:
        _warn(f"load hooks {path} failed: {error}")
        return None
    if not isinstance(raw, Mapping):
        _warn(f"load hooks {path} failed: root must be a mapping")
        return None
    hooks = raw.get("hooks")
    if not isinstance(hooks, list):
        _warn(f"load hooks {path} failed: hooks must be a list")
        return None
    return hooks


def _compile_rule(source: Path, index: int, raw: Mapping[object, object]) -> Rule:
    name = _required_non_empty_string(raw, "name")
    raw_event = raw.get("event")
    if not isinstance(raw_event, str) or (event := parse_event(raw_event)) is None:
        raise _RuleError(f'hook "{name}": unknown event "{raw_event}"')
    condition = _compile_condition(raw.get("if", _MISSING))
    action = _compile_action(raw.get("action", _MISSING))
    only_once = _optional_bool(raw, "only_once", False)
    asyncio_mode = _optional_bool(raw, "async", False)
    timeout_s = _compile_timeout(raw.get("timeout", "30s"))
    if asyncio_mode and is_blocking(event):
        raise _RuleError(f'hook "{name}": async not allowed for blocking events')
    if asyncio_mode and action.type in {ActionType.PROMPT, ActionType.SUBAGENT}:
        raise _RuleError(f'hook "{name}": async only supports shell/http actions')
    return Rule(
        name=name,
        event=event,
        action=action,
        condition=condition,
        only_once=only_once,
        asyncio_mode=asyncio_mode,
        timeout_s=timeout_s,
        source=str(source),
    )


def _compile_condition(raw: object) -> Condition | None:
    if raw is _MISSING:
        return None
    if not isinstance(raw, Mapping):
        raise _RuleError("if must be a mapping")
    modes = [key for key in ("all_of", "any_of") if key in raw]
    if len(modes) != 1 or len(raw) != 1:
        raise _RuleError("if must contain exactly one of all_of or any_of")
    mode_name = modes[0]
    raw_atoms = raw[mode_name]
    if not isinstance(raw_atoms, list) or not raw_atoms:
        raise _RuleError(f"if.{mode_name} must be a non-empty list")
    atoms: list[AtomCondition] = []
    for index, raw_atom in enumerate(raw_atoms, start=1):
        if not isinstance(raw_atom, Mapping) or set(raw_atom) != {"field", "match"}:
            raise _RuleError(f"if.{mode_name}[{index}] must contain only field and match")
        field = raw_atom["field"]
        if not isinstance(field, str) or not field.strip():
            raise _RuleError(f"if.{mode_name}[{index}].field must be a non-empty string")
        try:
            matcher = _compile_match_object(raw_atom["match"])
        except (TypeError, ValueError) as error:
            raise _RuleError(f"if.{mode_name}[{index}].match invalid: {error}") from error
        atoms.append(AtomCondition(field, matcher))
    mode = CombineMode.ALL_OF if mode_name == "all_of" else CombineMode.ANY_OF
    return Condition(mode, tuple(atoms))


def _compile_match_object(raw: object) -> Matcher:
    if not isinstance(raw, Mapping):
        raise TypeError("must be a mapping")
    kind = raw.get("type")
    if not isinstance(kind, str) or kind not in {"exact", "glob", "regex", "not"}:
        raise ValueError(f"unknown matcher type {kind!r}")
    expected = {"type", "inner"} if kind == "not" else {"type", "value"}
    if set(raw) != expected:
        raise ValueError(f"{kind} matcher must contain only {', '.join(sorted(expected))}")
    if kind == "not":
        return NotMatcher(_compile_match_object(raw["inner"]))
    value = raw["value"]
    if not isinstance(value, str):
        raise TypeError(f"{kind} matcher value must be a string")
    if kind == "exact":
        return ExactMatcher(value)
    if kind == "glob":
        return GlobMatcher(value, is_command=False)
    try:
        compiled = re.compile(value)
    except re.error as error:
        raise ValueError(f"invalid regex: {error}") from error
    return RegexMatcher(value, compiled)


def _compile_action(raw: object) -> Action:
    if not isinstance(raw, Mapping):
        raise _RuleError("action must be a mapping")
    raw_type = raw.get("type")
    if not isinstance(raw_type, str):
        raise _RuleError("action.type must be a string")
    try:
        action_type = ActionType(raw_type)
    except ValueError as error:
        raise _RuleError(f"unknown action type {raw_type!r}") from error
    if action_type is ActionType.SHELL:
        command = _required_string(raw, "command")
        return Action(action_type, shell=ShellAction(command))
    if action_type is ActionType.PROMPT:
        text = _required_string(raw, "text")
        return Action(action_type, prompt=PromptAction(text))
    if action_type is ActionType.HTTP:
        url = _required_string(raw, "url")
        method = _optional_string(raw, "method", "POST")
        body = _optional_nullable_string(raw, "body")
        headers = _string_mapping(raw.get("headers", {}), "action.headers")
        return Action(action_type, http=HttpAction(url, method, headers, body))
    agent_name = _required_string(raw, "agent_name")
    prompt = _required_string(raw, "prompt")
    return Action(action_type, subagent=SubagentAction(agent_name, prompt))


def _compile_timeout(raw: object) -> float:
    if not isinstance(raw, str) or (matched := _DURATION.fullmatch(raw)) is None:
        raise _RuleError("timeout must be a positive duration string")
    unit = matched.group(1)
    number = raw[: -len(unit)] if unit else raw
    seconds = float(number) * _DURATION_FACTORS[unit]
    if seconds <= 0:
        raise _RuleError("timeout must be greater than zero")
    return seconds


def _required_non_empty_string(raw: Mapping[object, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise _RuleError(f"{key} must be a non-empty string")
    return value


def _required_string(raw: Mapping[object, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise _RuleError(f"action.{key} must be a string")
    return value


def _optional_string(raw: Mapping[object, object], key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise _RuleError(f"action.{key} must be a string")
    return value


def _optional_nullable_string(raw: Mapping[object, object], key: str) -> str | None:
    value = raw.get(key)
    if value is not None and not isinstance(value, str):
        raise _RuleError(f"action.{key} must be a string")
    return value


def _optional_bool(raw: Mapping[object, object], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise _RuleError(f"{key} must be a boolean")
    return value


def _string_mapping(raw: object, field: str) -> dict[str, str]:
    if not isinstance(raw, Mapping) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
    ):
        raise _RuleError(f"{field} must be a string mapping")
    return dict(raw)


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


__all__ = ["load"]
