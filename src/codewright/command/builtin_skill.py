"""Built-in Skill catalog management command."""

from codewright.command.ui import UI

USAGE = "用法: /skill [list | info <name> | reload]"


async def handle_skill(ui: UI, args: str) -> None:
    """List, inspect, or reload locally discovered Skills."""
    parts = args.split()
    if not parts or parts == ["list"]:
        skills = ui.list_skills()
        if not skills:
            await ui.println("未发现可用 Skill。")
            return
        lines = ["可用 Skills："]
        lines.extend(
            f"/{skill.name}  mode={skill.mode}  source={skill.source.value}  {skill.description}"
            for skill in skills
        )
        await ui.println("\n".join(lines))
        return
    if len(parts) == 2 and parts[0] == "info":
        skill = ui.get_skill(parts[1])
        if skill is None:
            await ui.error(f"未知 Skill: {parts[1]}")
            return
        model = skill.model or "(current provider)"
        await ui.println(
            "\n".join(
                (
                    f"name: {skill.name}",
                    f"description: {skill.description}",
                    f"mode: {skill.mode}",
                    f"context: {skill.context}",
                    f"model: {model}",
                    f"source: {skill.source.value}",
                    f"path: {skill.source_path}",
                    f"directory skill: {'yes' if skill.is_directory else 'no'}",
                    f"resource root: {skill.source_dir}",
                )
            )
        )
        return
    if parts == ["reload"]:
        skills = await ui.reload_skills()
        await ui.println(f"Skills 已重新加载：{len(skills)} 个。")
        return
    await ui.error(USAGE)


__all__ = ["USAGE", "handle_skill"]
