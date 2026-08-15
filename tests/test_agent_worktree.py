from __future__ import annotations

from datetime import UTC, datetime

from codewright.agent.agent_worktree import append_cleanup_report, isolated_prompt
from codewright.tool import Result
from codewright.worktree import AutoCleanupReport, Worktree


def test_prompt_and_cleanup_result_preserve_contract() -> None:
    item = Worktree("a", "/repo/w", "worktree-a", "HEAD", "a" * 40, datetime.now(UTC), False)
    prompt = isolated_prompt("do it", "/repo", item)
    assert "/repo/w" in prompt and "do it" in prompt and "重新读取" in prompt
    original = Result("failed", is_error=True, error_code="boom", truncated=True, metadata={"x": 1})
    updated = append_cleanup_report(
        original, AutoCleanupReport(True, item.path, item.branch, "dirty")
    )
    assert updated.is_error and updated.error_code == "boom" and updated.truncated
    assert updated.metadata == {"x": 1}
    assert item.branch in updated.content
