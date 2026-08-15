"""Exact, unique-match UTF-8 file editing tool."""

import json
from collections.abc import Mapping
from pathlib import Path

from codewright.tool.ctx import resolve_path
from codewright.tool.models import Result


class EditFileTool:
    """Replace text only when the old text occurs exactly once."""

    name = "edit_file"
    read_only = False
    description = (
        "Replace one exact text fragment in a UTF-8 file; the old text must be unique. "
        "Always use read_file first to inspect the target and confirm old_string is unique."
    )
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to edit."},
            "old_string": {"type": "string", "description": "Exact unique text to replace."},
            "new_string": {"type": "string", "description": "Replacement text."},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    async def execute(self, arguments_json: str) -> Result:
        """Validate arguments and perform an atomic decision before writing."""
        parsed = _parse_arguments(arguments_json)
        if isinstance(parsed, Result):
            return parsed
        path, old_string, new_string = parsed
        return _edit_file(resolve_path(path), old_string, new_string)


def _parse_arguments(arguments_json: str) -> tuple[Path, str, str] | Result:
    try:
        arguments = json.loads(arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return _invalid_arguments("Arguments must be a valid JSON object.")
    if not isinstance(arguments, dict):
        return _invalid_arguments("Arguments must be a JSON object.")

    path = arguments.get("path")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")
    if not isinstance(path, str) or not path.strip():
        return _invalid_arguments("path must be a non-empty string.")
    if not isinstance(old_string, str) or not old_string:
        return _invalid_arguments("old_string must be a non-empty string.")
    if not isinstance(new_string, str):
        return _invalid_arguments("new_string must be a string.")
    return Path(path), old_string, new_string


def _edit_file(path: Path, old_string: str, new_string: str) -> Result:
    try:
        if not path.exists():
            return _edit_error("file_not_found", f"File does not exist: {path}")
        if not path.is_file():
            return _edit_error("not_a_file", f"Path is not a file: {path}")
        content = path.read_text(encoding="utf-8")
        match_count = content.count(old_string)
        if match_count == 0:
            return _edit_error(
                "match_not_found",
                "old_string matched 0 times; the file was not changed.",
                match_count=0,
            )
        if match_count != 1:
            return _edit_error(
                "match_not_unique",
                f"old_string matched {match_count} times; exactly 1 match is required.",
                match_count=match_count,
            )
        path.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    except UnicodeDecodeError:
        return _edit_error("encoding_error", f"File is not valid UTF-8: {path}")
    except PermissionError:
        return _edit_error("permission_denied", f"File is not readable or writable: {path}")
    except OSError:
        return _edit_error("edit_failed", f"Could not edit file: {path}")

    return Result(
        content=f"Replaced 1 exact match in {path}.",
        metadata={"path": str(path), "match_count": 1},
    )


def _invalid_arguments(message: str) -> Result:
    return Result(content=message, is_error=True, error_code="invalid_arguments")


def _edit_error(code: str, message: str, *, match_count: int | None = None) -> Result:
    metadata: dict[str, object] = {}
    if match_count is not None:
        metadata["match_count"] = match_count
    return Result(content=message, is_error=True, error_code=code, metadata=metadata)
