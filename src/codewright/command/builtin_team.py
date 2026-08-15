"""Safe `/team` lifecycle command using the existing command UI boundary."""

from __future__ import annotations

import shlex

from codewright.command.ui import UI


async def handle_team(ui: UI, args: str) -> None:
    accessor = ui.team_accessor()
    if accessor is None:
        await ui.error("当前项目未启用 Agent Team。")
        return
    try:
        tokens = shlex.split(args)
    except ValueError:
        await ui.error("Team 参数中的引号不完整。")
        return
    if not tokens:
        await ui.error("用法：/team list|info|use|delete|kill ...")
        return
    command, *tail = tokens
    try:
        if command == "list":
            if tail:
                raise ValueError("用法：/team list")
            active = getattr(accessor.active_team, "slug", "")
            items = accessor.list()
            await ui.println(
                "\n".join(
                    f"{item.slug}  {item.name}  members={len(item.members)}"
                    f"{'  active' if item.slug == active else ''}"
                    for item in items
                )
                or "当前没有 Agent Team。"
            )
        elif command == "info":
            name = _one(tail, "用法：/team info <name>")
            team = accessor.get(name)
            if team is None:
                raise ValueError(f"未知 Team：{name}")
            await ui.println(
                f"{team.name} ({team.slug})\nbackend={team.backend.value}\n"
                + "\n".join(f"{item.name}  {item.state.value}" for item in team.members)
            )
        elif command == "use":
            name = _one(tail, "用法：/team use <name>")
            team = accessor.use(name)
            await ui.println(f"已切换 Agent Team：{team.slug}")
        elif command == "delete":
            if not tail or tail[0].startswith("-"):
                raise ValueError("用法：/team delete <name> [--force] [--purge-sessions]")
            name = tail[0]
            flags = _flags(tail[1:], {"--force", "--purge-sessions"})
            report = await accessor.delete(
                name,
                force="--force" in flags,
                purge_sessions="--purge-sessions" in flags,
            )
            if not report.deleted:
                raise ValueError("Team 删除未完成：" + "; ".join(report.errors))
            await ui.println(f"已删除 Agent Team：{report.slug}")
        elif command == "kill":
            if len(tail) == 1:
                team = accessor.active_team
                if team is None:
                    raise ValueError("当前没有 active Team")
                team_name, member = team.slug, tail[0]
            elif len(tail) == 2:
                team_name, member = tail
            else:
                raise ValueError("用法：/team kill [team] <member>")
            await accessor.kill_member(team_name, member)
            await ui.println(f"已停止 Team 成员：{member}")
        else:
            raise ValueError(f"未知 Team 子命令：{command}")
    except ValueError as error:
        await ui.error(str(error))
    except Exception:
        await ui.error("Team 操作失败。")


def _one(tokens: list[str], usage: str) -> str:
    if len(tokens) != 1 or tokens[0].startswith("-"):
        raise ValueError(usage)
    return tokens[0]


def _flags(tokens: list[str], allowed: set[str]) -> set[str]:
    flags = set(tokens)
    if len(flags) != len(tokens) or any(token not in allowed for token in tokens):
        raise ValueError("Team 参数包含未知、重复或多余选项。")
    return flags


__all__ = ["handle_team"]
