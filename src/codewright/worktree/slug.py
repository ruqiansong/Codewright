"""Slug validation and contained path construction."""

from __future__ import annotations

import re
from pathlib import Path

_SEGMENT = re.compile(r"^[a-zA-Z0-9._-]+$")


def validate_slug(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("Worktree 名称必须是非空字符串")
    if len(name) > 64:
        raise ValueError("Worktree 名称不能超过 64 个字符")
    if name.startswith("/") or name.endswith("/") or "//" in name:
        raise ValueError("Worktree 名称不能以 / 开头或结尾，也不能包含 //")
    if "+" in name:
        raise ValueError("Worktree 名称不能包含 +")
    if ".." in name:
        raise ValueError("Worktree 名称不能包含 ..")
    if name.endswith(".") or name.casefold().endswith(".lock"):
        raise ValueError("Worktree 名称不能以 . 或 .lock 结尾")
    segments = name.split("/")
    if any(part in {".", ".."} or _SEGMENT.fullmatch(part) is None for part in segments):
        raise ValueError("Worktree 名称只能包含字母、数字、点、下划线、短横线和单个 /")
    if any(part.casefold() == ".metadata" for part in segments):
        raise ValueError("Worktree 名称使用了保留名称 .metadata")
    return name


def flat_slug(name: str) -> str:
    return validate_slug(name).replace("/", "+")


def contained_child(root: Path, name: str) -> Path:
    """Return a direct child without following a pre-existing child symlink."""
    root = root.resolve()
    child = root / flat_slug(name)
    if child.parent != root:
        raise ValueError("Worktree 路径超出管理目录")
    if child.is_symlink() and child.resolve().parent != root:
        raise ValueError("Worktree 路径指向管理目录之外")
    return child


__all__ = ["contained_child", "flat_slug", "validate_slug"]
