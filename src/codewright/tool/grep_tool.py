"""Bounded regular-expression search across UTF-8 text files."""

import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import Path
from re import Pattern

from codewright.permission.sandbox import sandbox_ok
from codewright.permission.settings import path_pattern_safe
from codewright.tool.ctx import resolve_path
from codewright.tool.models import Result

MAX_RESULTS = 100
MAX_LINE_CHARS = 2_000


class GrepTool:
    """Search file contents and return file, line, and matching text."""

    name = "grep"
    read_only = True
    description = "Search UTF-8 file contents with a Python regular expression."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression."},
            "path": {
                "type": "string",
                "description": "Optional file or directory; defaults to the current directory.",
            },
            "glob": {
                "type": "string",
                "description": "Optional file glob filter; defaults to **/*.",
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
        """Search candidates off-loop and yield between files."""
        parsed = _parse_arguments(arguments_json)
        if isinstance(parsed, Result):
            return parsed
        pattern_text, root, file_glob = parsed
        if not path_pattern_safe(file_glob):
            return _sandbox_error("File glob is outside the project sandbox.")
        root = resolve_path(root, fallback=self._sandbox_root)
        if not sandbox_ok(self._sandbox_root, str(root)):
            return _sandbox_error("Search path is outside the project sandbox.")
        try:
            pattern = re.compile(pattern_text)
        except re.error:
            return Result(
                "pattern must be a valid Python regular expression.",
                is_error=True,
                error_code="invalid_regex",
            )
        if not root.exists():
            return _path_error("path_not_found", f"Search path does not exist: {root}")

        try:
            candidates = [root] if root.is_file() else sorted(root.glob(file_glob))
        except (OSError, ValueError, NotImplementedError):
            return _path_error("invalid_glob", "The file glob or search path is invalid.")

        matches: list[str] = []
        skipped_files = 0
        long_lines = 0
        sandbox_skipped = 0
        truncated = False
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if not sandbox_ok(self._sandbox_root, str(candidate)):
                sandbox_skipped += 1
                continue
            remaining = MAX_RESULTS + 1 - len(matches)
            file_matches, skipped, file_long_lines = _search_file(candidate, pattern, remaining)
            skipped_files += int(skipped)
            long_lines += file_long_lines
            matches.extend(file_matches)
            if len(matches) > MAX_RESULTS:
                truncated = True
                break
            await asyncio.sleep(0)

        matches = matches[:MAX_RESULTS]
        if not matches:
            if sandbox_skipped:
                return _sandbox_error("All searched files were outside the project sandbox.")
            return Result(
                "No content matches were found.",
                metadata={
                    "matches": 0,
                    "skipped_files": skipped_files,
                    "long_lines": long_lines,
                    "sandbox_skipped": sandbox_skipped,
                },
            )
        content = "\n".join(matches)
        if truncated:
            content += "\n[truncated]"
        return Result(
            content,
            truncated=truncated,
            metadata={
                "matches": len(matches),
                "skipped_files": skipped_files,
                "long_lines": long_lines,
                "sandbox_skipped": sandbox_skipped,
            },
        )


def _search_file(
    path: Path,
    pattern: Pattern[str],
    limit: int,
) -> tuple[list[str], bool, int]:
    matches: list[str] = []
    long_lines = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            line_number = 0
            while len(matches) < limit:
                line = handle.readline(MAX_LINE_CHARS + 1)
                if line == "":
                    break
                line_number += 1
                incomplete = len(line) > MAX_LINE_CHARS and not line.endswith(("\n", "\r"))
                display_line = line[:MAX_LINE_CHARS].rstrip("\r\n")
                if incomplete:
                    long_lines += 1
                    while line and not line.endswith(("\n", "\r")):
                        line = handle.readline(MAX_LINE_CHARS + 1)
                if pattern.search(display_line):
                    suffix = " [line truncated]" if incomplete else ""
                    matches.append(f"{path.as_posix()}:{line_number}:{display_line}{suffix}")
    except (OSError, UnicodeDecodeError):
        return [], True, long_lines
    return matches, False, long_lines


def _parse_arguments(arguments_json: str) -> tuple[str, Path, str] | Result:
    try:
        arguments = json.loads(arguments_json or "{}")
    except (json.JSONDecodeError, TypeError):
        return _invalid_arguments("Arguments must be a valid JSON object.")
    if not isinstance(arguments, dict):
        return _invalid_arguments("Arguments must be a JSON object.")
    pattern = arguments.get("pattern")
    path = arguments.get("path", ".")
    file_glob = arguments.get("glob", "**/*")
    if not isinstance(pattern, str) or not pattern:
        return _invalid_arguments("pattern must be a non-empty string.")
    if not isinstance(path, str):
        return _invalid_arguments("path must be a string when provided.")
    if not path.strip():
        path = "."
    if not isinstance(file_glob, str) or not file_glob.strip():
        return _invalid_arguments("glob must be a non-empty string when provided.")
    return pattern, Path(path), file_glob


def _invalid_arguments(message: str) -> Result:
    return Result(content=message, is_error=True, error_code="invalid_arguments")


def _path_error(code: str, message: str) -> Result:
    return Result(content=message, is_error=True, error_code=code)


def _sandbox_error(message: str) -> Result:
    return _path_error("path_outside_sandbox", message)
