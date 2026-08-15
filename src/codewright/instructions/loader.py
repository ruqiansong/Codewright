"""Load layered Codewright Markdown instructions with bounded includes."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_INCLUDE_PATTERN = re.compile(r"^@include\s+(.+)$")


@dataclass(frozen=True, slots=True)
class Loader:
    """Load project, project-config, and user instruction files in priority order."""

    project_root: str
    user_home: str | None = None
    max_depth: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.project_root, str) or not self.project_root.strip():
            raise ValueError("project_root must be a non-empty string")
        if self.user_home is not None and (
            not isinstance(self.user_home, str) or not self.user_home.strip()
        ):
            raise ValueError("user_home must be a non-empty string or None")
        if not isinstance(self.max_depth, int) or isinstance(self.max_depth, bool):
            raise TypeError("max_depth must be an integer")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least one")

    def load(self) -> str:
        """Return all readable instruction layers from highest to lowest priority."""
        project_root = Path(self.project_root).resolve()
        user_home = Path(self.user_home or os.path.expanduser("~")).resolve()
        user_root = (user_home / ".codewright").resolve()
        sources = (
            (project_root / "codewright.md", project_root),
            (project_root / ".codewright" / "codewright.md", project_root),
            (user_root / "codewright.md", user_root),
        )
        layers = [
            content
            for path, boundary in sources
            if (content := self._load_file(path, boundary, 1, set()))
        ]
        return "\n\n".join(layers)

    def _load_file(
        self,
        path: str | Path,
        boundary: str | Path,
        depth: int,
        visited: set[str],
    ) -> str:
        """Load one file and recursively expand standalone include directives."""
        requested = Path(path)
        if depth > self.max_depth:
            return self._warning("超过最大嵌套深度", requested)

        boundary_path = Path(boundary).resolve()
        resolved = requested.resolve()
        try:
            resolved.relative_to(boundary_path)
        except ValueError:
            return self._warning("路径超出允许范围", requested)

        key = str(resolved)
        if key in visited:
            return self._warning("检测到环路", requested)
        if not resolved.is_file():
            return ""

        try:
            data = resolved.read_bytes()
        except OSError as error:
            logger.warning(
                "Instruction file could not be read path=%s error=%s",
                resolved,
                type(error).__name__,
            )
            return ""
        if b"\x00" in data[:512]:
            return self._warning("二进制文件不可读", requested)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return self._warning("文件不是有效 UTF-8 文本", requested)

        chain = visited | {key}
        output: list[str] = []
        for line in text.splitlines(keepends=True):
            match = _INCLUDE_PATTERN.fullmatch(line.rstrip("\r\n"))
            if match is None:
                output.append(line)
                continue
            include_value = match.group(1).strip()
            include_path = resolved.parent / include_value
            replacement = self._load_file(include_path, boundary_path, depth + 1, chain)
            output.append(replacement)
            if line.endswith(("\n", "\r")) and replacement and not replacement.endswith("\n"):
                output.append("\n")
        return "".join(output)

    @staticmethod
    def _warning(reason: str, path: Path) -> str:
        return f"<!-- @include {reason}，已跳过: {path} -->"
