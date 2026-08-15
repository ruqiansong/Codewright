"""Bounded UTF-8 file reading with stable line numbers."""

import json
from collections.abc import Mapping
from pathlib import Path

from codewright.tool.ctx import resolve_path
from codewright.tool.models import Result

MAX_LINES = 2_000
MAX_CONTENT_BYTES = 256 * 1024
_TRUNCATION_MARKER = "\n[truncated]"


class ReadFileTool:
    """Read a UTF-8 text file without allowing unbounded model context."""

    name = "read_file"
    read_only = True
    description = "Read a UTF-8 text file and return its content with numbered lines."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path to the text file to read."}},
        "required": ["path"],
        "additionalProperties": False,
    }

    async def execute(self, arguments_json: str) -> Result:
        """Validate arguments and read the selected file off the event loop."""
        path_or_error = _parse_path(arguments_json)
        if isinstance(path_or_error, Result):
            return path_or_error
        return _read_file(resolve_path(path_or_error))


def _parse_path(arguments_json: str) -> Path | Result:
    try:
        arguments = json.loads(arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return _invalid_arguments("Arguments must be a valid JSON object.")
    if not isinstance(arguments, dict):
        return _invalid_arguments("Arguments must be a JSON object.")
    path = arguments.get("path")
    if not isinstance(path, str) or not path.strip():
        return _invalid_arguments("path must be a non-empty string.")
    return Path(path)


def _read_file(path: Path) -> Result:
    try:
        if not path.exists():
            return _file_error("file_not_found", f"File does not exist: {path}")
        if not path.is_file():
            return _file_error("not_a_file", f"Path is not a file: {path}")

        rendered: list[str] = []
        byte_count = 0
        line_count = 0
        truncated = False
        marker_bytes = len(_TRUNCATION_MARKER.encode("utf-8"))
        with path.open("r", encoding="utf-8") as handle:
            while line_count < MAX_LINES:
                line = handle.readline(MAX_CONTENT_BYTES + 1)
                if line == "":
                    break
                line_count += 1
                formatted = f"{line_count:6d}\t{line.rstrip(chr(10) + chr(13))}"
                if line.endswith(("\n", "\r")):
                    formatted += "\n"
                encoded = formatted.encode("utf-8")
                if byte_count + len(encoded) + marker_bytes > MAX_CONTENT_BYTES:
                    available = max(0, MAX_CONTENT_BYTES - marker_bytes - byte_count)
                    rendered.append(encoded[:available].decode("utf-8", errors="ignore"))
                    truncated = True
                    break
                rendered.append(formatted)
                byte_count += len(encoded)
                if len(line) > MAX_CONTENT_BYTES and not line.endswith(("\n", "\r")):
                    truncated = True
                    break
            else:
                truncated = handle.read(1) != ""
    except UnicodeDecodeError:
        return _file_error("encoding_error", f"File is not valid UTF-8: {path}")
    except PermissionError:
        return _file_error("permission_denied", f"File is not readable: {path}")
    except OSError:
        return _file_error("read_failed", f"Could not read file: {path}")

    content = "".join(rendered).rstrip("\r\n")
    if truncated:
        content += _TRUNCATION_MARKER
    if not content:
        content = "File is empty."
    return Result(
        content=content,
        truncated=truncated,
        metadata={"path": str(path), "lines": line_count},
    )


def _invalid_arguments(message: str) -> Result:
    return Result(content=message, is_error=True, error_code="invalid_arguments")


def _file_error(code: str, message: str) -> Result:
    return Result(content=message, is_error=True, error_code=code)
