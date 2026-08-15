"""Small presentation helpers for worktree-isolated subagents."""

from __future__ import annotations

from pathlib import Path

from codewright.tool import Result
from codewright.worktree import AutoCleanupReport, Worktree


def isolated_prompt(prompt: str, parent_cwd: str | Path, worktree: Worktree) -> str:
    parent = Path(parent_cwd).absolute()
    child = Path(worktree.path).absolute()
    notice = (
        "你正在独立 Git Worktree 中执行任务。\n"
        f"父工作目录：{parent}\n"
        f"当前 Worktree：{child}\n"
        "把任务中指向父目录的绝对路径前缀转换为当前 Worktree 前缀；"
        "使用相对路径时以当前 Worktree 为准。编辑任何文件前必须重新读取。"
    )
    return f"{notice}\n\n原始任务：\n{prompt}"


def append_cleanup_report(result: Result, report: AutoCleanupReport) -> Result:
    if not report.kept:
        return result
    suffix = (
        "\n\nWorktree 已安全保留："
        f"\npath: {report.path}\nbranch: {report.branch}\nreason: {report.reason}"
    )
    return Result(
        result.content + suffix,
        is_error=result.is_error,
        error_code=result.error_code,
        truncated=result.truncated,
        metadata=result.metadata,
    )


__all__ = ["append_cleanup_report", "isolated_prompt"]
