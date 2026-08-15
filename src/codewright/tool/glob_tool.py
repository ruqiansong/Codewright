"""Bounded, ordered file discovery using pathlib glob patterns."""

import asyncio
import json
from collections.abc import Mapping
from pathlib import Path

from codewright.permission.sandbox import sandbox_ok
from codewright.permission.settings import path_pattern_safe
from codewright.tool.ctx import resolve_path
from codewright.tool.models import Result

MAX_RESULTS = 100


class GlobTool:
    """Find files matching a glob pattern beneath an optional directory."""

    name = "glob"
    read_only = True
    description = "Find files by glob pattern, including recursive patterns such as **/*.py."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern to match."},
            "path": {
                "type": "string",
                "description": "Optional directory to search; defaults to the current directory.",
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def __init__(self, sandbox_root: Path) -> None:
        if not isinstance(sandbox_root, Path):
            raise TypeError("sandbox_root must be a Path")
        self._sandbox_root = sandbox_root.resolve()

    async def execute(self, arguments_json: str) -> Result:
        """Find a bounded set of files while periodically yielding control."""
        parsed = _parse_arguments(arguments_json)
        if isinstance(parsed, Result):
            return parsed
        pattern, root = parsed
        if not path_pattern_safe(pattern):
            return _sandbox_error("Glob pattern is outside the project sandbox.")
        root = resolve_path(root, fallback=self._sandbox_root)
        if not sandbox_ok(self._sandbox_root, str(root)):
            return _sandbox_error("Search path is outside the project sandbox.")
        try:
            if not root.exists():
                return _path_error("path_not_found", f"Search path does not exist: {root}")
            if not root.is_dir():
                return _path_error("not_a_directory", f"Search path is not a directory: {root}")

            matches: list[str] = []
            sandbox_skipped = 0
            for index, candidate in enumerate(root.glob(pattern), start=1):
                if candidate.is_file():
                    if not sandbox_ok(self._sandbox_root, str(candidate)):
                        sandbox_skipped += 1
                        continue
                    matches.append(candidate.as_posix())
                    if len(matches) > MAX_RESULTS:
                        break
                if index % 32 == 0:
                    await asyncio.sleep(0)
        except (OSError, ValueError, NotImplementedError):
            return _path_error("invalid_glob", "The glob pattern or search path is invalid.")

        matches.sort()
        truncated = len(matches) > MAX_RESULTS
        matches = matches[:MAX_RESULTS]
        if not matches:
            if sandbox_skipped:
                return _sandbox_error("All matched files were outside the project sandbox.")
            return Result("No files matched the glob pattern.", metadata={"matches": 0})
        content = "\n".join(matches)
        if truncated:
            content += "\n[truncated]"
        return Result(
            content,
            truncated=truncated,
            metadata={
                "matches": len(matches),
                "path": str(root),
                "pattern": pattern,
                "sandbox_skipped": sandbox_skipped,
            },
        )


def _parse_arguments(arguments_json: str) -> tuple[str, Path] | Result:
    try:
        arguments = json.loads(arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return _invalid_arguments("Arguments must be a valid JSON object.")
    if not isinstance(arguments, dict):
        return _invalid_arguments("Arguments must be a JSON object.")
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")
    if not isinstance(pattern, str) or not pattern.strip():
        return _invalid_arguments("pattern must be a non-empty string.")
    if not isinstance(path, str):
        return _invalid_arguments("path must be a string when provided.")
    if not path.strip():
        path = "."
    return pattern, Path(path)


def _invalid_arguments(message: str) -> Result:
    return Result(content=message, is_error=True, error_code="invalid_arguments")


def _path_error(code: str, message: str) -> Result:
    return Result(content=message, is_error=True, error_code=code)


def _sandbox_error(message: str) -> Result:
    return _path_error("path_outside_sandbox", message)
