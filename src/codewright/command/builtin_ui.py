"""Built-in handlers that change UI or session state."""

from codewright.command.ui import UI
from codewright.permission import Mode


async def handle_exit(ui: UI, args: str) -> None:
    del args
    await ui.request_exit()


async def handle_plan(ui: UI, args: str) -> None:
    del args
    await ui.set_mode(Mode.PLAN)
    await ui.println("已进入计划模式（仅使用只读工具）。")


async def handle_compact(ui: UI, args: str) -> None:
    del args
    await ui.force_compact()


async def handle_resume(ui: UI, args: str) -> None:
    del args
    await ui.open_resume_menu()


async def handle_clear(ui: UI, args: str) -> None:
    del args
    await ui.clear_and_new_session()
