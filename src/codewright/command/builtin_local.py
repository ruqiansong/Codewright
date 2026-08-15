"""Purely local built-in slash-command handlers."""

from __future__ import annotations

from collections import OrderedDict

from codewright.command.models import Handler
from codewright.command.registry import Registry
from codewright.command.ui import UI
from codewright.hook import Event, Rule


def make_help_handler(registry: Registry) -> Handler:
    """Build a help handler backed only by the supplied registry."""

    async def handle_help(ui: UI, args: str) -> None:
        del args
        commands = registry.visible()
        width = max((len(command.name) for command in commands), default=0)
        await ui.println(
            "\n".join(
                f"/{command.name.ljust(width)}  {command.description}" for command in commands
            )
        )

    return handle_help


async def handle_status(ui: UI, args: str) -> None:
    del args
    input_tokens, output_tokens = ui.usage()
    project_files, user_files = ui.memory_files()
    rows = (
        ("Mode", str(ui.mode)),
        ("Tokens", f"{input_tokens} in / {output_tokens} out"),
        ("Tools", f"{ui.tool_count()} enabled"),
        ("Memories", f"{len(project_files) + len(user_files)} files"),
        ("Model", ui.model_name()),
        ("Directory", ui.cwd()),
    )
    width = max(len(key) for key, _ in rows)
    await ui.println("\n".join(f"{key.ljust(width)}: {value}" for key, value in rows))


async def handle_memory(ui: UI, args: str) -> None:
    del args
    project_files, user_files = ui.memory_files()
    lines = [*(f"[project] {name}" for name in project_files)]
    lines.extend(f"[user] {name}" for name in user_files)
    await ui.println("\n".join(lines) if lines else "无已加载的记忆文件")


async def handle_permission(ui: UI, args: str) -> None:
    del args
    await ui.println(str(ui.mode))


async def handle_session(ui: UI, args: str) -> None:
    del args
    await ui.println(f"Session: {ui.session_id()}\nPath: {ui.session_path()}")


async def handle_hooks(ui: UI, args: str) -> None:
    """List loaded lifecycle Hooks grouped by first-seen event."""
    del args
    rules = ui.hook_rules()
    if not rules:
        await ui.println("No hooks loaded.")
        return
    grouped: OrderedDict[Event, list[Rule]] = OrderedDict()
    for rule in rules:
        grouped.setdefault(rule.event, []).append(rule)
    lines: list[str] = []
    for event, event_rules in grouped.items():
        for rule in event_rules:
            flags = [
                label
                for enabled, label in ((rule.only_once, "[once]"), (rule.asyncio_mode, "[async]"))
                if enabled
            ]
            suffix = f"  {' '.join(flags)}" if flags else ""
            lines.append(f"  {rule.name}  {event.value}  {rule.action.type.value}{suffix}")
    sources = ui.hook_sources()
    lines.append(f"Loaded from: {', '.join(sources)}")
    await ui.println("\n".join(lines))
