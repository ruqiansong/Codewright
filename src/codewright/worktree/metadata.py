"""Trusted worktree metadata serialization."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from codewright.worktree.models import Worktree
from codewright.worktree.session import _atomic_json
from codewright.worktree.slug import contained_child, flat_slug, validate_slug

_FIELDS = {"name", "path", "branch", "based_on", "head_commit", "created", "manual"}


def metadata_path(metadata_dir: Path, name: str) -> Path:
    return metadata_dir / f"{flat_slug(name)}.json"


def save_metadata(metadata_dir: Path, worktree: Worktree) -> None:
    if worktree.created.tzinfo is None or worktree.created.utcoffset() is None:
        raise ValueError("Worktree created 必须包含时区")
    _atomic_json(
        metadata_path(metadata_dir, worktree.name),
        {
            "based_on": worktree.based_on,
            "branch": worktree.branch,
            "created": worktree.created.isoformat(),
            "head_commit": worktree.head_commit,
            "manual": worktree.manual,
            "name": worktree.name,
            "path": worktree.path,
        },
    )


def load_metadata(path: Path, worktree_dir: Path) -> Worktree:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != _FIELDS:
        raise ValueError("Worktree metadata 字段无效")
    name = value.get("name")
    if not isinstance(name, str):
        raise ValueError("Worktree metadata name 无效")
    validate_slug(name)
    if path.name != f"{flat_slug(name)}.json":
        raise ValueError("Worktree metadata 文件名不匹配")
    expected = contained_child(worktree_dir, name)
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or Path(raw_path).absolute() != expected.absolute():
        raise ValueError("Worktree metadata 路径无效")
    branch = value.get("branch")
    if not isinstance(branch, str) or branch != f"worktree-{flat_slug(name)}":
        raise ValueError("Worktree metadata 分支无效")
    for field in ("based_on", "head_commit"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"Worktree metadata {field} 无效")
    based_on = value["based_on"]
    head_commit = value["head_commit"]
    manual = value.get("manual")
    if not isinstance(manual, bool):
        raise ValueError("Worktree metadata manual 无效")
    try:
        created = datetime.fromisoformat(value["created"])
    except (TypeError, ValueError) as error:
        raise ValueError("Worktree metadata created 无效") from error
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("Worktree metadata created 缺少时区")
    return Worktree(
        name=name,
        path=str(expected),
        branch=branch,
        based_on=based_on,
        head_commit=head_commit,
        created=created,
        manual=manual,
    )


__all__ = ["load_metadata", "metadata_path", "save_metadata"]
