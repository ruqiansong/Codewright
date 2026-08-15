"""Explicit construction of Codewright's fixed built-in command set."""

from codewright.command.builtin_local import (
    handle_hooks,
    handle_memory,
    handle_permission,
    handle_session,
    handle_status,
    make_help_handler,
)
from codewright.command.builtin_prompt import handle_do, handle_review
from codewright.command.builtin_skill import handle_skill
from codewright.command.builtin_team import handle_team
from codewright.command.builtin_ui import (
    handle_clear,
    handle_compact,
    handle_exit,
    handle_plan,
    handle_resume,
)
from codewright.command.builtin_worktree import handle_worktree
from codewright.command.models import Command, Kind
from codewright.command.registry import Registry


def register_builtins(registry: Registry) -> None:
    """Register the fixed built-ins; any collision aborts construction."""
    commands = (
        Command("clear", "清空当前对话并开始新会话", Kind.UI, handle_clear),
        Command("compact", "手动压缩当前上下文", Kind.UI, handle_compact),
        Command("do", "执行当前计划", Kind.PROMPT, handle_do),
        Command("exit", "退出 Codewright", Kind.UI, handle_exit),
        Command("help", "显示可用命令", Kind.LOCAL, make_help_handler(registry)),
        Command("hooks", "列出已加载的 hook 列表", Kind.LOCAL, handle_hooks),
        Command("memory", "列出已加载的记忆文件", Kind.LOCAL, handle_memory),
        Command("permission", "显示当前权限模式", Kind.LOCAL, handle_permission),
        Command("plan", "切换到只读计划模式", Kind.UI, handle_plan),
        Command("resume", "恢复历史会话", Kind.UI, handle_resume),
        Command("review", "请求审查当前代码上下文", Kind.PROMPT, handle_review),
        Command("session", "显示当前会话信息", Kind.LOCAL, handle_session),
        Command("skill", "管理和查看本地 Skills", Kind.UI, handle_skill, accepts_args=True),
        Command("status", "显示 Codewright 当前状态", Kind.LOCAL, handle_status),
        Command("team", "管理 Agent Team", Kind.UI, handle_team, accepts_args=True),
        Command(
            "worktree",
            "管理隔离的 Git Worktree",
            Kind.UI,
            handle_worktree,
            accepts_args=True,
        ),
    )
    for command in commands:
        registry.register(command)
