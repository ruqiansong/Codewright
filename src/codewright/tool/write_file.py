"""UTF-8 file creation and replacement tool."""

import json
from collections.abc import Mapping
from pathlib import Path

from codewright.tool.ctx import resolve_path
from codewright.tool.models import Result


class WriteFileTool:
    """Create or overwrite a UTF-8 text file, including missing parents."""

    name = "write_file"
    read_only = False
    description = "Create or overwrite a UTF-8 text file, creating parent directories as needed."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Destination file path."},
            "content": {"type": "string", "description": "Complete replacement content."},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    async def execute(self, arguments_json: str) -> Result:
        """Validate arguments and write off the event loop."""
        parsed = _parse_arguments(arguments_json)
        if isinstance(parsed, Result):
            return parsed
        path, content = parsed
        return _write_file(resolve_path(path), content)


def _parse_arguments(arguments_json: str) -> tuple[Path, str] | Result:
    try:
        arguments = json.loads(arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return _invalid_arguments("Arguments must be a valid JSON object.")
    if not isinstance(arguments, dict):
        return _invalid_arguments("Arguments must be a JSON object.")
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        return _invalid_arguments("path must be a non-empty string.")
    if "content" not in arguments or not isinstance(arguments["content"], str):
        return _invalid_arguments("content must be a string; an empty string is allowed.")
    return Path(path), arguments["content"]


def _write_file(path: Path, content: str) -> Result:
    try:
        if path.exists() and path.is_dir():
            return _write_error("not_a_file", f"Destination is a directory: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except PermissionError:
        return _write_error("permission_denied", f"Destination is not writable: {path}")
    except OSError:
        return _write_error("write_failed", f"Could not write file: {path}")

    byte_count = len(content.encode("utf-8"))
    return Result(
        content=f"Wrote {byte_count} UTF-8 bytes to {path}.",
        metadata={"path": str(path), "bytes": byte_count},
    )


def _invalid_arguments(message: str) -> Result:
    return Result(content=message, is_error=True, error_code="invalid_arguments")


def _write_error(code: str, message: str) -> Result:
    return Result(content=message, is_error=True, error_code=code)
