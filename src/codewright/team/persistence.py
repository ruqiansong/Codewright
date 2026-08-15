"""Safe JSON persistence and cross-process locking for Agent Teams."""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any

_INVALID_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_WINDOWS_RESERVED = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_INTERNAL_RESERVED = {"config", "mailbox", "tasks"}


class LockTimeoutError(TimeoutError):
    """Raised when a shared Team lock cannot be acquired safely."""


def sanitize_team_name(name: str) -> str:
    """Convert a display name to a safe, direct-child slug."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("team name must be a non-empty string")
    slug = _INVALID_NAME.sub("-", name.strip()).strip("-_")[:48].rstrip("-_")
    if not slug or slug in {".", ".."}:
        raise ValueError("team name does not contain a usable safe name")
    if slug.casefold() in _WINDOWS_RESERVED | _INTERNAL_RESERVED:
        raise ValueError("team name resolves to a reserved name")
    if _SAFE_NAME.fullmatch(slug) is None:
        raise ValueError("team name could not be made safe")
    return slug


def contained_team_dir(root: Path, slug: str) -> Path:
    """Return one direct Team child while rejecting symlink escape."""
    if not isinstance(slug, str) or _SAFE_NAME.fullmatch(slug) is None:
        raise ValueError("invalid team slug")
    if slug in {".", ".."} or slug.casefold() in _WINDOWS_RESERVED | _INTERNAL_RESERVED:
        raise ValueError("invalid or reserved team slug")
    root = root.resolve()
    candidate = root / slug
    if candidate.parent != root:
        raise ValueError("team path escapes the teams root")
    if candidate.is_symlink():
        resolved = candidate.resolve()
        if resolved.parent != root or resolved != candidate:
            raise ValueError("team symlink escapes the teams root")
    return candidate


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_write_json(path: Path, value: object) -> None:
    """Durably replace JSON using a unique same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{token}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class FileLock:
    """An owner-token lock based on atomic O_EXCL file creation."""

    def __init__(
        self,
        path: Path,
        *,
        timeout: float = 10.0,
        stale_after: float = 30.0,
    ) -> None:
        if timeout < 0 or stale_after <= 0:
            raise ValueError("invalid lock timeout")
        self.path = path
        self.timeout = timeout
        self.stale_after = stale_after
        self.owner_token = uuid.uuid4().hex
        self._owned = False

    async def __aenter__(self) -> FileLock:
        deadline = time.monotonic() + self.timeout
        while True:
            if await asyncio.to_thread(self._try_acquire):
                self._owned = True
                return self
            await asyncio.to_thread(self._clear_stale)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LockTimeoutError(f"timed out acquiring lock: {self.path.name}")
            await asyncio.sleep(min(remaining, random.uniform(0.005, 0.1)))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await asyncio.to_thread(self.release)

    def _try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token": self.owner_token,
            "pid": os.getpid(),
            "created_at": time.time(),
        }
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return False
        try:
            data = json.dumps(payload, separators=(",", ":")).encode()
            os.write(fd, data)
        except BaseException:
            os.close(fd)
            self.path.unlink(missing_ok=True)
            raise
        os.close(fd)
        return True

    def _read_owner(self) -> dict[str, Any] | None:
        try:
            value = read_json(self.path)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _clear_stale(self) -> None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return
        if time.time() - stat.st_mtime <= self.stale_after:
            return
        owner = self._read_owner()
        if owner is None:
            return
        pid = owner.get("pid")
        token = owner.get("token")
        if not isinstance(pid, int) or not isinstance(token, str) or _pid_alive(pid):
            return
        current = self._read_owner()
        if current is not None and current.get("token") == token:
            self.path.unlink(missing_ok=True)

    def release(self) -> None:
        if not self._owned:
            return
        owner = self._read_owner()
        if owner is not None and owner.get("token") == self.owner_token:
            self.path.unlink(missing_ok=True)
        self._owned = False


__all__ = [
    "FileLock",
    "LockTimeoutError",
    "atomic_write_json",
    "contained_team_dir",
    "read_json",
    "sanitize_team_name",
]
