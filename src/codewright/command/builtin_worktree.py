"""Strict `/worktree` lifecycle command."""

from __future__ import annotations

import shlex

from codewright.command.ui import UI
from codewright.worktree import WorktreeError


async def handle_worktree(ui: UI, args: str) -> None:
    accessor = ui.worktree_accessor()
    if accessor is None:
        await ui.error("当前项目未启用 Worktree 管理。")
        return
    try:
        tokens = shlex.split(args)
    except ValueError:
        await ui.error("Worktree 参数中的引号不完整。")
        return
    if not tokens:
        await ui.error("用法：/worktree create|list|enter|exit|remove ...")
        return
    command, *tail = tokens
    try:
        if command == "create":
            name = _one_slug(tail, "用法：/worktree create <slug>")
            item = await accessor.create(name)
            await ui.println(f"已创建 Worktree：{item.name}\n{item.path}\n{item.branch}")
        elif command == "list":
            if tail:
                raise ValueError("用法：/worktree list")
            items = accessor.list()
            if not items:
                await ui.println("当前没有受管理的 Worktree。")
            else:
                await ui.println(
                    "\n".join(
                        f"{item.name}  {item.branch}  {item.path}  "
                        f"manual={str(item.manual).lower()} active={str(item.active).lower()}"
                        for item in items
                    )
                )
        elif command == "enter":
            name = _one_slug(tail, "用法：/worktree enter <slug>")
            path = await accessor.enter(name)
            await ui.println(f"已进入 Worktree：{path}")
        elif command == "exit":
            flags = _flags(tail, {"--remove", "--discard"})
            if "--discard" in flags and "--remove" not in flags:
                raise ValueError("--discard 只能与 --remove 一起使用")
            path = await accessor.exit(
                remove="--remove" in flags,
                discard="--discard" in flags,
            )
            await ui.println(f"已退出 Worktree，逻辑 cwd 已恢复：{path}")
        elif command == "remove":
            if not tail or tail[0].startswith("-"):
                raise ValueError("用法：/worktree remove <slug> [--discard]")
            name = tail[0]
            flags = _flags(tail[1:], {"--discard"})
            await accessor.remove(name, discard="--discard" in flags)
            await ui.println(f"已删除 Worktree：{name}")
        else:
            raise ValueError(f"未知 Worktree 子命令：{command}")
    except (ValueError, WorktreeError) as error:
        await ui.error(str(error))
    except Exception:
        await ui.error("Worktree 操作失败。")


def _one_slug(tokens: list[str], usage: str) -> str:
    if len(tokens) != 1 or tokens[0].startswith("-"):
        raise ValueError(usage)
    return tokens[0]


def _flags(tokens: list[str], allowed: set[str]) -> set[str]:
    flags = set(tokens)
    if len(flags) != len(tokens) or any(token not in allowed for token in tokens):
        raise ValueError("Worktree 参数包含未知、重复或多余选项。")
    return flags


__all__ = ["handle_worktree"]
