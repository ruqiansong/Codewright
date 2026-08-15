"""Task-local logical working directory for model-facing tools."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_cwd: ContextVar[Path | None] = ContextVar("codewright_tool_cwd", default=None)


@contextmanager
def with_cwd(path: str | Path) -> Iterator[None]:
    token = _cwd.set(Path(path).absolute())
    try:
        yield
    finally:
        _cwd.reset(token)


def cwd_from_ctx() -> Path | None:
    return _cwd.get()


def resolve_path(path: str | Path, fallback: str | Path | None = None) -> Path:
    selected = Path(path)
    if selected.is_absolute():
        return selected
    base = cwd_from_ctx()
    if base is None and fallback is not None:
        base = Path(fallback)
    return (base or Path.cwd()) / selected


__all__ = ["cwd_from_ctx", "resolve_path", "with_cwd"]
