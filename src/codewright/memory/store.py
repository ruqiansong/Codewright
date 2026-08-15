"""Atomic, path-contained Markdown note storage."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from codewright.memory.types import Note, NoteType, UpdateAction

logger = logging.getLogger(__name__)

MAX_INDEX_BYTES = 25 * 1024
MAX_INDEX_LINES = 200
_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_NOTE_FILENAME_PATTERN = re.compile(
    rf"^(?:{'|'.join(re.escape(value.value) for value in NoteType)})_"
    r"[a-z0-9]+(?:_[a-z0-9]+)*\.md$"
)


class Store:
    """Manage one project-level or user-level memory directory."""

    def __init__(self, directory: str) -> None:
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("directory must be a non-empty string")
        self._dir = Path(directory).resolve()
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return self._dir

    def ensure_dir(self) -> None:
        """Create the store root without accepting a symlink root."""
        self._dir.mkdir(parents=True, exist_ok=True)
        if self._dir.is_symlink() or not self._dir.is_dir():
            raise OSError("memory directory must be a real directory")

    def load_index(self) -> str:
        """Read the bounded index; a missing index is an empty memory."""
        with self._lock:
            path = self._dir / "MEMORY.md"
            if not path.exists():
                return ""
            if path.is_symlink() or not path.is_file():
                logger.warning("Unsafe memory index ignored")
                return ""
            try:
                return _bounded_index(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                logger.warning("Memory index could not be read")
                return ""

    def apply(self, actions: list[UpdateAction]) -> None:
        """Validate and atomically apply actions, then rebuild the bounded index."""
        if not isinstance(actions, list) or not all(
            isinstance(action, UpdateAction) for action in actions
        ):
            raise TypeError("actions must be a list of UpdateAction values")
        with self._lock:
            self.ensure_dir()
            for action in actions:
                try:
                    self._apply_one_locked(action)
                except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
                    logger.warning(
                        "Unsafe or invalid memory action skipped action=%s error=%s",
                        _safe_action(action.action),
                        type(error).__name__,
                    )
            self._rebuild_index_locked()

    def _apply_one_locked(self, action: UpdateAction) -> None:
        if action.action not in {"create", "update", "delete"}:
            raise ValueError("unsupported action")
        if action.level not in {"project", "user"}:
            raise ValueError("unsupported level")
        if action.action == "create":
            note_type = NoteType(action.type)
            title = _required_text(action.title, "title")
            content = _required_text(action.content, "content")
            slug = _safe_slug(action.slug)
            filename = f"{note_type.value}_{slug}.md"
            target = self._safe_target(filename, must_exist=False)
            if target.exists() or target.is_symlink():
                raise ValueError("note already exists")
            now = datetime.now(UTC)
            self._write_note_locked(target, note_type, title, content, now, now)
            return

        target = self._safe_target(action.filename, must_exist=True)
        if action.action == "delete":
            target.unlink()
            return

        existing = self._read_note_locked(target)
        title = _required_text(action.title, "title")
        content = _required_text(action.content, "content")
        self._write_note_locked(
            target,
            existing.type,
            title,
            content,
            existing.created,
            datetime.now(UTC),
        )

    def _safe_target(self, filename: str, *, must_exist: bool) -> Path:
        if not isinstance(filename, str) or _NOTE_FILENAME_PATTERN.fullmatch(filename) is None:
            raise ValueError("unsafe note filename")
        if Path(filename).name != filename or Path(filename).is_absolute():
            raise ValueError("unsafe note filename")
        target = self._dir / filename
        if target.parent.resolve() != self._dir:
            raise ValueError("note target escapes memory root")
        if target.is_symlink():
            raise ValueError("note target must not be a symlink")
        if must_exist and not target.is_file():
            raise FileNotFoundError("note does not exist")
        return target

    def _write_note_locked(
        self,
        target: Path,
        note_type: NoteType,
        title: str,
        content: str,
        created: datetime,
        updated: datetime,
    ) -> None:
        frontmatter = yaml.safe_dump(
            {
                "type": note_type.value,
                "title": title,
                "created": created.isoformat(),
                "updated": updated.isoformat(),
            },
            allow_unicode=True,
            sort_keys=False,
        )
        _atomic_write(target, f"---\n{frontmatter}---\n{content.strip()}\n")

    def _read_note_locked(self, path: Path) -> Note:
        text = path.read_text(encoding="utf-8")
        metadata, content = _split_frontmatter(text)
        note_type = NoteType(metadata["type"])
        title = _required_text(metadata["title"], "title")
        created = _parse_datetime(metadata["created"])
        updated = _parse_datetime(metadata["updated"])
        prefix = f"{note_type.value}_"
        slug = path.stem[len(prefix) :]
        _safe_slug(slug)
        return Note(note_type, title, slug, content.strip(), path.name, created, updated)

    def _rebuild_index_locked(self) -> None:
        notes: list[Note] = []
        for path in sorted(self._dir.glob("*.md")):
            if path.name == "MEMORY.md" or path.is_symlink():
                continue
            if _NOTE_FILENAME_PATTERN.fullmatch(path.name) is None:
                continue
            try:
                notes.append(self._read_note_locked(path))
            except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError):
                logger.warning("Invalid memory note omitted from index filename=%s", path.name)
        lines = [
            f"- [{note.type.value}] {note.title} — {_description(note.content)}" for note in notes
        ]
        index = _bounded_index("\n".join(lines) + ("\n" if lines else ""))
        _atomic_write(self._dir / "MEMORY.md", index)


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise ValueError("missing frontmatter")
    boundary = text.find("\n---\n", 4)
    if boundary < 0:
        raise ValueError("unterminated frontmatter")
    value = yaml.safe_load(text[4:boundary])
    if not isinstance(value, dict):
        raise ValueError("frontmatter must be an object")
    return value, text[boundary + 5 :]


def _atomic_write(path: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _bounded_index(text: str) -> str:
    lines = text.splitlines()[:MAX_INDEX_LINES]
    candidate = "\n".join(lines) + ("\n" if lines else "")
    encoded = candidate.encode("utf-8")
    if len(encoded) <= MAX_INDEX_BYTES:
        return candidate
    return encoded[:MAX_INDEX_BYTES].decode("utf-8", errors="ignore")


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _safe_slug(value: str) -> str:
    if not isinstance(value, str) or _SLUG_PATTERN.fullmatch(value) is None:
        raise ValueError("unsafe note slug")
    return value


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise TypeError("note timestamp must be a string")
    return datetime.fromisoformat(value)


def _description(content: str) -> str:
    compact = " ".join(content.split())
    return compact if len(compact) <= 120 else compact[:119] + "…"


def _safe_action(value: object) -> str:
    return (
        value if isinstance(value, str) and value in {"create", "update", "delete"} else "invalid"
    )
