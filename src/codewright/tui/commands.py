"""Thin TUI adapter for slash commands and compaction presentation."""

from __future__ import annotations

import logging

from codewright.agent import CompactEvent, CompactPhase
from codewright.command import UI, Kind, Registry, parse_invocation

logger = logging.getLogger(__name__)


async def dispatch_slash(value: str, registry: Registry, ui: UI) -> bool:
    """Dispatch one slash-shaped input and isolate command failures."""
    invocation = parse_invocation(value)
    if not invocation.is_slash:
        return False
    command = registry.lookup(invocation.name) if invocation.valid else None
    if command is None or (invocation.args and not command.accepts_args):
        await ui.println(f"未知命令: {value.strip()}，请输入 /help 查看可用命令。")
        return True
    if command.kind is not Kind.LOCAL and not ui.idle():
        await ui.error("当前任务正在运行，请等待完成后再执行该命令。")
        return True
    try:
        await command.handler(ui, invocation.args)
    except Exception as error:
        logger.error(
            "Slash command failed command=%s error=%s",
            command.name,
            type(error).__name__,
        )
        await ui.error("命令执行失败，请稍后重试。")
    return True


def format_compact_notice(event: CompactEvent) -> str:
    """Render one safe, protocol-neutral compaction status message."""
    if event.phase is CompactPhase.BEFORE_AUTO:
        return "正在压缩上下文..."
    if event.phase is CompactPhase.BEFORE_EMERGENCY:
        return "上下文撞墙，自动压缩中..."
    if event.phase is CompactPhase.BEFORE_MANUAL:
        return "正在手动压缩上下文..."
    if event.error_message:
        return f"上下文压缩失败：{event.error_message}"
    return f"已压缩，token 从 {event.before} 降至 {event.after}"


__all__ = [
    "dispatch_slash",
    "format_compact_notice",
]
