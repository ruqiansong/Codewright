"""Trusted, fail-closed Git worktree lifecycle management."""

from __future__ import annotations

import asyncio
import builtins
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from codewright.worktree.git import (
    _has_worktree_changes,
    _resolve_head_sha_from_fs,
    _run_git,
)
from codewright.worktree.metadata import load_metadata, metadata_path, save_metadata
from codewright.worktree.models import (
    AutoCleanupReport,
    ExitAction,
    ExitOptions,
    ExitReport,
    Worktree,
    WorktreeError,
    WorktreeSession,
)
from codewright.worktree.session import load_session, save_session
from codewright.worktree.slug import contained_child, flat_slug, validate_slug

logger = logging.getLogger(__name__)
_AGENT_NAME = re.compile(r"^agent-a[0-9a-f]{7}$")


def random_agent_name() -> str:
    return f"agent-a{uuid.uuid4().hex[:7]}"


class Manager:
    """Own worktrees rooted below one repository's .codewright directory."""

    def __init__(self, repo_root: str | Path) -> None:
        requested = Path(repo_root).absolute()
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=requested,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
                text=True,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""},
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise WorktreeError("当前目录不是可用的 Git 仓库") from error
        top = Path(completed.stdout.strip()).resolve()
        if requested.resolve() != top:
            raise WorktreeError("Worktree Manager 必须在 Git 仓库顶层创建")
        self.repo_root = str(top)
        self.worktree_dir = top / ".codewright" / "worktrees"
        self.metadata_dir = self.worktree_dir / ".metadata"
        self.session_file = top / ".codewright" / "worktree_session.json"
        self.worktree_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.lock = asyncio.Lock()
        self.active: dict[str, Worktree] = {}
        self._current_session: WorktreeSession | None = None
        self._recover()

    def list(self) -> tuple[Worktree, ...]:
        return tuple(sorted(self.active.values(), key=lambda item: item.name.casefold()))

    def get(self, name: str) -> Worktree | None:
        validate_slug(name)
        return self.active.get(name)

    def current_session(self) -> WorktreeSession | None:
        return self._current_session

    def _recover(self) -> None:
        for path in sorted(self.metadata_dir.glob("*.json")):
            try:
                item = load_metadata(path, self.worktree_dir)
                item_path = Path(item.path)
                if not item_path.exists():
                    path.unlink(missing_ok=True)
                    continue
                if item_path.is_symlink() or _resolve_head_sha_from_fs(item_path) is None:
                    raise ValueError("Worktree 目录无有效 Git 指针")
                flat = flat_slug(item.name).casefold()
                if any(flat_slug(name).casefold() == flat for name in self.active):
                    raise ValueError("Worktree 名称发生大小写碰撞")
                if any(
                    child.name != ".metadata"
                    and child.name.casefold() == flat
                    and child != item_path
                    for child in self.worktree_dir.iterdir()
                ):
                    raise ValueError("Worktree 目录发生大小写碰撞")
                self.active[item.name] = item
            except Exception as error:
                logger.warning(
                    "跳过不可信 Worktree metadata file=%s error=%s",
                    path.name,
                    type(error).__name__,
                )
        try:
            session = load_session(self.session_file)
        except Exception as error:
            logger.warning("Worktree session 已损坏，已清空 error=%s", type(error).__name__)
            self._discard_invalid_session()
            return
        if session is None:
            return
        active = self.active.get(session.worktree_name)
        if (
            active is None
            or active.path != session.worktree_path
            or session.original_cwd != self.repo_root
            or not Path(active.path).is_dir()
        ):
            logger.warning("Worktree session 已失效，已清空")
            self._discard_invalid_session()
            return
        self._current_session = session

    def _discard_invalid_session(self) -> None:
        try:
            save_session(self.session_file, None)
        except OSError as error:
            logger.warning("Worktree session 无法重写 error=%s", type(error).__name__)

    def _assert_no_collision(self, name: str, target: Path) -> None:
        folded = flat_slug(name).casefold()
        for existing in self.active:
            if flat_slug(existing).casefold() == folded and existing != name:
                raise WorktreeError("Worktree 名称与已有名称发生大小写碰撞")
        for path in self.metadata_dir.glob("*.json"):
            expected = metadata_path(self.metadata_dir, name)
            if path == expected:
                raise WorktreeError("存在未被恢复的 Worktree metadata，拒绝覆盖")
            if path.stem.casefold() == folded:
                raise WorktreeError("Worktree 名称与已有 metadata 发生大小写碰撞")
        for path in self.worktree_dir.iterdir():
            if path.name == ".metadata":
                continue
            if path.name.casefold() == target.name.casefold() and path != target:
                raise WorktreeError("Worktree 名称与已有目录发生大小写碰撞")

    async def create(self, name: str, base_ref: str = "HEAD", manual: bool = False) -> Worktree:
        validate_slug(name)
        if not isinstance(base_ref, str) or not base_ref:
            raise ValueError("base_ref 必须是非空字符串")
        if not isinstance(manual, bool):
            raise TypeError("manual 必须是布尔值")
        async with self.lock:
            existing = self.active.get(name)
            if existing is not None:
                return existing
            target = contained_child(self.worktree_dir, name)
            self._assert_no_collision(name, target)
            if target.exists() or target.is_symlink():
                raise WorktreeError("目标 Worktree 目录已存在但不受可信 metadata 管理")
            branch = f"worktree-{flat_slug(name)}"
            base_dir = (
                self._current_session.worktree_path
                if self._current_session is not None
                else self.repo_root
            )
            base_sha = await _run_git(base_dir, "rev-parse", f"{base_ref}^{{commit}}")
            if not base_sha:
                raise WorktreeError("无法解析 Worktree 创建基线")
            created_path = False
            try:
                await _run_git(base_dir, "worktree", "add", "-b", branch, str(target), base_sha)
                created_path = target.exists()
                head = _resolve_head_sha_from_fs(target)
                if head is None:
                    raise WorktreeError("无法读取新 Worktree 的 HEAD")
                item = Worktree(
                    name=name,
                    path=str(target),
                    branch=branch,
                    based_on=base_ref,
                    head_commit=head,
                    created=datetime.now(UTC),
                    manual=manual,
                )
                save_metadata(self.metadata_dir, item)
                self.active[name] = item
            except Exception:
                created_path = created_path or target.exists()
                await self._rollback_create(target, branch, created_path)
                raise
            await self._post_create_setup(target)
            return item

    async def _rollback_create(self, target: Path, branch: str, created_path: bool) -> None:
        if not created_path:
            return
        try:
            await _run_git(self.repo_root, "worktree", "remove", "--force", str(target))
            await _run_git(self.repo_root, "branch", "-D", branch)
        except Exception as error:
            logger.warning("Worktree 创建回滚未完全成功 error=%s", type(error).__name__)

    async def _post_create_setup(self, target: Path) -> None:
        for relative in (".codewright/config.yaml", ".codewright/settings.local.yaml"):
            source = Path(self.repo_root) / relative
            destination = target / relative
            try:
                if source.is_file() and not source.is_symlink() and not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
            except OSError as error:
                logger.warning(
                    "复制 Worktree 配置失败 file=%s error=%s",
                    relative,
                    type(error).__name__,
                )
        try:
            await _run_git(target, "config", "--get", "core.hooksPath")
        except WorktreeError:
            pass
        root = Path(self.repo_root).resolve()
        for relative in ("node_modules", ".venv", "vendor"):
            source = root / relative
            destination = target / relative
            try:
                if destination.exists() or destination.is_symlink() or not source.is_dir():
                    continue
                resolved = source.resolve()
                if not resolved.is_relative_to(root) or source.is_symlink():
                    logger.warning("跳过越界或软链共享目录 path=%s", relative)
                    continue
                destination.symlink_to(resolved, target_is_directory=True)
            except OSError as error:
                logger.warning(
                    "创建 Worktree 共享目录软链失败 path=%s error=%s",
                    relative,
                    type(error).__name__,
                )
        await self._copy_included_files(target)

    async def _copy_included_files(self, target: Path) -> None:
        include_file = Path(self.repo_root) / ".worktreeinclude"
        if not include_file.is_file() or include_file.is_symlink():
            return
        try:
            patterns = [
                line.strip()
                for line in include_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
            if not patterns:
                return
            output = await _run_git(
                self.repo_root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"
            )
        except (OSError, UnicodeError, WorktreeError) as error:
            logger.warning("读取 .worktreeinclude 失败 error=%s", type(error).__name__)
            return
        root = Path(self.repo_root).resolve()
        for raw in output.split("\0"):
            if not raw:
                continue
            relative = Path(raw)
            if not any(relative.match(pattern) for pattern in patterns):
                continue
            source = root / relative
            destination = target / relative
            try:
                if (
                    source.is_symlink()
                    or not source.is_file()
                    or not source.resolve().is_relative_to(root)
                ):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source, destination)
            except OSError as error:
                logger.warning(
                    "复制 Worktree include 文件失败 file=%s error=%s",
                    raw,
                    type(error).__name__,
                )

    async def enter(self, name: str) -> WorktreeSession:
        validate_slug(name)
        async with self.lock:
            item = self.active.get(name)
            if item is None:
                raise WorktreeError("Worktree 不存在")
            if self._current_session is not None:
                if self._current_session.worktree_name == name:
                    return self._current_session
                raise WorktreeError("已有另一个 Worktree session，请先退出")
            session = WorktreeSession(
                original_cwd=self.repo_root,
                worktree_path=item.path,
                worktree_name=name,
                session_id=uuid.uuid4().hex,
            )
            save_session(self.session_file, session)
            self._current_session = session
            return session

    async def exit(
        self,
        name: str,
        action: ExitAction = ExitAction.KEEP,
        opts: ExitOptions | None = None,
    ) -> ExitReport:
        async with self.lock:
            opts = opts or ExitOptions()
            session = self._current_session
            if session is None or session.worktree_name != name:
                raise WorktreeError("只能退出当前 Worktree session")
            item = self.active.get(name)
            if item is None:
                raise WorktreeError("当前 Worktree 已失效")
            if action is ExitAction.REMOVE:
                await self._remove_unlocked(item, opts)
                removed = True
                try:
                    save_session(self.session_file, None)
                finally:
                    self._current_session = None
            else:
                removed = False
                save_session(self.session_file, None)
                self._current_session = None
            return ExitReport(removed, item.path, item.branch, session.original_cwd)

    async def remove(self, name: str, opts: ExitOptions | None = None) -> ExitReport:
        validate_slug(name)
        async with self.lock:
            opts = opts or ExitOptions()
            if self._current_session is not None and self._current_session.worktree_name == name:
                raise WorktreeError("当前 Worktree 正在使用，请先执行 exit")
            item = self.active.get(name)
            if item is None:
                raise WorktreeError("Worktree 不存在")
            await self._remove_unlocked(item, opts)
            return ExitReport(True, item.path, item.branch, self.repo_root)

    async def _remove_unlocked(self, item: Worktree, opts: ExitOptions) -> None:
        if not opts.discard_changes and await _has_worktree_changes(item.path, item.head_commit):
            raise WorktreeError("Worktree 包含未提交修改或新提交；使用 --discard 才能删除")
        await _run_git(self.repo_root, "worktree", "remove", "--force", item.path)
        try:
            await _run_git(self.repo_root, "branch", "-D", item.branch)
        except WorktreeError:
            logger.warning("Worktree 已删除，但专用分支未能清理 branch=%s", item.branch)
        self.active.pop(item.name, None)
        metadata_path(self.metadata_dir, item.name).unlink(missing_ok=True)

    async def auto_cleanup(self, name: str) -> AutoCleanupReport:
        validate_slug(name)
        async with self.lock:
            item = self.active.get(name)
            if item is None:
                return AutoCleanupReport(False)
            if item.manual:
                return AutoCleanupReport(True, item.path, item.branch, "手动 Worktree 会被保留")
            if (
                self._current_session is not None
                and self._current_session.worktree_name == item.name
            ):
                return AutoCleanupReport(True, item.path, item.branch, "Worktree 当前正在使用")
            if await _has_worktree_changes(item.path, item.head_commit):
                return AutoCleanupReport(True, item.path, item.branch, "检测到修改或新提交")
            try:
                await self._remove_unlocked(item, ExitOptions(discard_changes=False))
            except Exception:
                return AutoCleanupReport(True, item.path, item.branch, "自动清理失败，已安全保留")
            return AutoCleanupReport(False)

    async def sweep_stale(self, cutoff: datetime) -> builtins.list[str]:
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("cutoff 必须包含时区")
        removed: builtins.list[str] = []
        for item in self.list():
            if (
                item.manual
                or _AGENT_NAME.fullmatch(item.name) is None
                or item.created >= cutoff
                or (
                    self._current_session is not None
                    and self._current_session.worktree_name == item.name
                )
            ):
                continue
            report = await self.auto_cleanup(item.name)
            if not report.kept:
                removed.append(item.name)
        return removed


__all__ = ["Manager", "random_agent_name"]
