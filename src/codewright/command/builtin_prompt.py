"""Built-in commands that inject fixed prompts into the conversation."""

from codewright.command.ui import UI
from codewright.permission import Mode
from codewright.prompt import EXECUTE_DIRECTIVE

REVIEW_DIRECTIVE = (
    "请审查当前上下文中的代码变更和已读取的文件，指出潜在 bug、可读性问题和可简化处。"
)


async def handle_do(ui: UI, args: str) -> None:
    del args
    await ui.set_mode(Mode.DEFAULT)
    await ui.println("已退出计划模式，开始执行上文计划。")
    await ui.inject_and_send("/do", EXECUTE_DIRECTIVE)


async def handle_review(ui: UI, args: str) -> None:
    del args
    await ui.inject_and_send("/review", REVIEW_DIRECTIVE)
