"""Non-interactive Git helpers used by the worktree manager."""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from codewright.worktree.models import WorktreeError

_SHA = re.compile(r"^[0-9a-fA-F]{40,64}$")


async def _run_git(work_dir: str | Path, *args: str) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_ASKPASS"] = ""
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(work_dir),
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
    except (OSError, ValueError) as error:
        raise WorktreeError("Git 命令无法启动") from error
    if process.returncode != 0:
        raise WorktreeError("Git 命令执行失败")
    decoded = stdout.decode("utf-8", errors="replace")
    return decoded.rstrip("\0") if "-z" in args else decoded.strip()


def _resolve_head_sha_from_fs(path: str | Path) -> str | None:
    try:
        root = Path(path)
        dotgit = root / ".git"
        if dotgit.is_file():
            marker = dotgit.read_text(encoding="utf-8").strip()
            if not marker.startswith("gitdir:"):
                return None
            gitdir = Path(marker[7:].strip())
            if not gitdir.is_absolute():
                gitdir = (root / gitdir).resolve()
        elif dotgit.is_dir():
            gitdir = dotgit.resolve()
        else:
            return None
        common = gitdir
        commondir = gitdir / "commondir"
        if commondir.is_file():
            common_ref = Path(commondir.read_text(encoding="utf-8").strip())
            common = common_ref if common_ref.is_absolute() else (gitdir / common_ref).resolve()
        head = (gitdir / "HEAD").read_text(encoding="ascii").strip()
        if _SHA.fullmatch(head):
            return head.lower()
        if not head.startswith("ref: "):
            return None
        ref = head[5:].strip()
        for base in (gitdir, common):
            loose = base / ref
            if loose.is_file():
                sha = loose.read_text(encoding="ascii").strip()
                return sha.lower() if _SHA.fullmatch(sha) else None
        packed = common / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="ascii").splitlines():
                if not line or line[0] in "#^":
                    continue
                sha, separator, packed_ref = line.partition(" ")
                if separator and packed_ref == ref and _SHA.fullmatch(sha):
                    return sha.lower()
    except (OSError, UnicodeError):
        return None
    return None


async def _has_worktree_changes(path: str | Path, base_commit: str) -> bool:
    try:
        status = await _run_git(path, "status", "--porcelain")
        if status:
            return True
        head = await _run_git(path, "rev-parse", "HEAD")
        return head.casefold() != base_commit.casefold()
    except Exception:
        return True


__all__ = ["_has_worktree_changes", "_resolve_head_sha_from_fs", "_run_git"]
