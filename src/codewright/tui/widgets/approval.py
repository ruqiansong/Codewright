"""Interactive, bounded presentation for one pending tool approval."""

import json

from rich.console import RenderableType
from rich.text import Text
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Static

from codewright.agent import ApprovalRequest
from codewright.permission import Outcome
from codewright.utils.logging import redact_sensitive

_MAX_ARGUMENT_CHARS = 600
_OPTIONS = (
    ("1. 允许本次", Outcome.ALLOW_ONCE),
    ("2. 永久允许（写入本地配置）", Outcome.ALLOW_FOREVER),
    ("3. 拒绝本次", Outcome.DENY_ONCE),
)


class ApprovalWidget(Static):
    """Render and operate the three-choice permission prompt."""

    can_focus = True
    BINDINGS = [
        Binding("up", "previous", "Previous", show=False),
        Binding("k", "previous", "Previous", show=False),
        Binding("down", "next", "Next", show=False),
        Binding("j", "next", "Next", show=False),
        Binding("enter", "select", "Select", show=False),
        Binding("1", "choose(0)", "Allow once", show=False),
        Binding("2", "choose(1)", "Allow forever", show=False),
        Binding("3", "choose(2)", "Deny once", show=False),
    ]

    class CursorChanged(Message):
        """Notify the screen that the visible selection moved."""

        def __init__(self, cursor: int) -> None:
            super().__init__()
            self.cursor = cursor

    class Selected(Message):
        """Submit exactly one permission outcome to the screen."""

        def __init__(self, outcome: Outcome) -> None:
            super().__init__()
            self.outcome = outcome

    def __init__(
        self,
        request: ApprovalRequest,
        *,
        cursor: int = 0,
        source: str = "",
    ) -> None:
        super().__init__(classes="approval-request")
        if not 0 <= cursor < len(_OPTIONS):
            raise ValueError("cursor is outside the approval menu")
        self.request = request
        self.cursor = cursor
        self.source = source
        self.argument_preview = _argument_preview(request.arguments_json)

    def action_previous(self) -> None:
        self._move(-1)

    def action_next(self) -> None:
        self._move(1)

    def action_select(self) -> None:
        self.post_message(self.Selected(_OPTIONS[self.cursor][1]))

    def action_choose(self, index: int) -> None:
        if 0 <= index < len(_OPTIONS):
            self.cursor = index
            self.post_message(self.CursorChanged(index))
            self.post_message(self.Selected(_OPTIONS[index][1]))

    def _move(self, offset: int) -> None:
        self.cursor = (self.cursor + offset) % len(_OPTIONS)
        self.post_message(self.CursorChanged(self.cursor))
        self.refresh()

    def render(self) -> RenderableType:
        heading = f"● {self.request.name}"
        if self.source:
            heading += f"  [来自 SubAgent {self.source}]"
        text = Text(heading, style="bold yellow")
        for line in self.argument_preview.splitlines() or ["{}"]:
            text.append(f"\n  {line}", style="dim")
        text.append(f"\n  原因：{redact_sensitive(self.request.reason)}", style="dim")
        text.append("\n\n是否继续?", style="bold")
        for index, (label, _) in enumerate(_OPTIONS):
            prefix = "> " if index == self.cursor else "  "
            style = "bold cyan" if index == self.cursor else ""
            text.append(f"\n{prefix}{label}", style=style)
        text.append("\n\n↑↓ 选择 · 回车确认 · Esc 取消", style="dim")
        return text


def _argument_preview(arguments_json: str) -> str:
    safe = redact_sensitive(arguments_json)
    try:
        parsed = json.loads(safe)
        rendered = json.dumps(parsed, ensure_ascii=False, indent=2)
    except (json.JSONDecodeError, TypeError):
        rendered = safe
    rendered = rendered.replace("\r", "")
    if len(rendered) > _MAX_ARGUMENT_CHARS:
        return rendered[:_MAX_ARGUMENT_CHARS].rstrip() + "\n[truncated]"
    return rendered


__all__ = ["ApprovalWidget"]
