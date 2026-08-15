"""Strict YAML-frontmatter parsing for Codewright Skills."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import yaml

from codewright.skills.models import SkillContext, SkillDef, SkillMode, SkillSource

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9-]*$")
_ALLOWED_FIELDS = frozenset({"name", "description", "mode", "model", "context", "license"})


class SkillParseError(ValueError):
    """Raised when a Skill file cannot be parsed safely."""


def parse_frontmatter(raw: str) -> tuple[dict[str, object], str]:
    """Split and validate one frontmatter mapping and non-empty Markdown body."""
    if not isinstance(raw, str):
        raise TypeError("raw must be a string")
    lines = raw.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise SkillParseError("Skill must start with a YAML frontmatter delimiter")

    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.rstrip("\r\n") == "---"),
        None,
    )
    if closing_index is None:
        raise SkillParseError("Skill frontmatter is not closed")

    frontmatter = "".join(lines[1:closing_index])
    try:
        loaded = yaml.safe_load(frontmatter)
    except yaml.YAMLError as error:
        raise SkillParseError("Skill frontmatter contains invalid YAML") from error
    if not isinstance(loaded, Mapping):
        raise SkillParseError("Skill frontmatter must be a mapping")
    if not all(isinstance(key, str) for key in loaded):
        raise SkillParseError("Skill frontmatter keys must be strings")

    meta = dict(loaded)
    unknown = set(meta) - _ALLOWED_FIELDS
    if unknown:
        raise SkillParseError("Skill frontmatter contains unknown fields")
    body = "".join(lines[closing_index + 1 :]).strip()
    if not body:
        raise SkillParseError("Skill body must not be empty")
    return meta, body


def parse_skill_file(
    path: str | Path,
    source: SkillSource | str,
    *,
    is_directory: bool,
) -> SkillDef:
    """Read and parse one selected Skill source file."""
    source_path = Path(path)
    if not isinstance(is_directory, bool):
        raise TypeError("is_directory must be a boolean")
    try:
        selected_source = SkillSource(source)
    except ValueError as error:
        raise ValueError("source must be project or user") from error
    try:
        if source_path.is_symlink() or not source_path.is_file():
            raise SkillParseError("Skill source must be a regular non-symbolic-link file")
        resolved_path = source_path.resolve(strict=True)
        raw = source_path.read_text(encoding="utf-8")
    except SkillParseError:
        raise
    except (OSError, UnicodeError) as error:
        raise SkillParseError("Skill source could not be read as UTF-8") from error

    meta, body = parse_frontmatter(raw)
    name, description, mode, model, context = _validate_meta(meta)
    return SkillDef(
        name=name,
        description=description,
        prompt_body=body,
        mode=mode,
        model=model,
        context=context,
        source_path=resolved_path,
        source_dir=resolved_path.parent,
        is_directory=is_directory,
        source=selected_source,
    )


def substitute_arguments(prompt_body: str, args: str) -> str:
    """Replace every Skill argument placeholder without changing other text."""
    if not isinstance(prompt_body, str) or not isinstance(args, str):
        raise TypeError("prompt_body and args must be strings")
    return prompt_body.replace("$ARGUMENTS", args)


def _validate_meta(
    meta: Mapping[str, object],
) -> tuple[str, str, SkillMode, str | None, SkillContext]:
    name = meta.get("name")
    if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
        raise SkillParseError("Skill name must match ^[a-z][a-z0-9-]*$")

    description = meta.get("description")
    if not isinstance(description, str) or not description.strip():
        raise SkillParseError("Skill description must be a non-empty string")
    if description != description.strip() or "\n" in description or "\r" in description:
        raise SkillParseError("Skill description must be one trimmed line")

    mode_value = meta.get("mode", "inline")
    if mode_value not in {"inline", "fork"}:
        raise SkillParseError("Skill mode must be inline or fork")
    mode: SkillMode = mode_value

    context_value = meta.get("context", "full")
    if context_value not in {"full", "recent", "none"}:
        raise SkillParseError("Skill context must be full, recent, or none")
    context: SkillContext = context_value

    model_value = meta.get("model")
    if model_value is not None:
        if not isinstance(model_value, str) or not model_value.strip():
            raise SkillParseError("Skill model must be a non-empty string or null")
        if model_value != model_value.strip() or any(char.isspace() for char in model_value):
            raise SkillParseError("Skill model must be a trimmed provider name")

    return name, description, mode, model_value, context


__all__ = [
    "SkillParseError",
    "parse_frontmatter",
    "parse_skill_file",
    "substitute_arguments",
]
