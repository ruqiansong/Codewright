"""Session selection widgets and bounded presentation helpers."""

from __future__ import annotations

import asyncio
from datetime import datetime

from textual import on
from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from codewright.session import SessionInfo, list_sessions


class SessionItem:
    """Presentation wrapper for one resumable session."""

    def __init__(self, info: SessionInfo, *, now: datetime | None = None) -> None:
        if not isinstance(info, SessionInfo):
            raise TypeError("info must be a SessionInfo")
        self.info = info
        self.display_text = (
            f"{info.title} · {_relative_time(info.modified_at, now or datetime.now())}"
            f" · {info.model} · {_format_size(info.size)}"
        )


class ResumePanel(Static):
    """Searchable keyboard-driven session chooser."""

    can_focus = True

    class Selected(Message):
        def __init__(self, info: SessionInfo) -> None:
            super().__init__()
            self.info = info

    class Cancelled(Message):
        pass

    def __init__(self, sessions: list[SessionInfo]) -> None:
        super().__init__(id="resume-panel")
        self._items = [SessionItem(info) for info in sessions]
        self._visible = list(self._items)

    def compose(self) -> ComposeResult:
        yield Static("恢复历史会话（输入关键词过滤，↑/↓ 选择，Enter 确认，Esc 取消）")
        yield Input(placeholder="搜索标题、模型或会话 ID", id="resume-search")
        yield OptionList(id="resume-options")

    def on_mount(self) -> None:
        self._refresh_options("")
        self.query_one("#resume-search", Input).focus()

    @on(Input.Changed, "#resume-search")
    def filter_sessions(self, event: Input.Changed) -> None:
        self._refresh_options(event.value)

    @on(Input.Submitted, "#resume-search")
    def select_from_search(self) -> None:
        self._post_selected()

    @on(OptionList.OptionSelected, "#resume-options")
    def select_option(self, event: OptionList.OptionSelected) -> None:
        index = event.option_index
        if 0 <= index < len(self._visible):
            self.post_message(self.Selected(self._visible[index].info))

    def move_up(self) -> None:
        self.query_one("#resume-options", OptionList).action_cursor_up()

    def move_down(self) -> None:
        self.query_one("#resume-options", OptionList).action_cursor_down()

    def select_highlighted(self) -> None:
        self._post_selected()

    def _post_selected(self) -> None:
        options = self.query_one("#resume-options", OptionList)
        index = options.highlighted
        if index is not None and 0 <= index < len(self._visible):
            self.post_message(self.Selected(self._visible[index].info))

    def _refresh_options(self, query: str) -> None:
        needle = query.strip().casefold()
        self._visible = [
            item
            for item in self._items
            if not needle
            or needle in item.info.title.casefold()
            or needle in item.info.model.casefold()
            or needle in item.info.id.casefold()
        ]
        options = self.query_one("#resume-options", OptionList)
        options.clear_options()
        if self._visible:
            options.add_options(Option(item.display_text) for item in self._visible)
            options.highlighted = 0
        else:
            options.add_option(Option("没有匹配的会话", disabled=True))


async def begin_resume(sessions_dir: str) -> list[SessionInfo]:
    """Scan session metadata off the Textual event-loop thread."""
    return await asyncio.to_thread(list_sessions, sessions_dir)


def _relative_time(value: datetime, now: datetime) -> str:
    seconds = max(0, int((now - value).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60} minutes ago"
    if seconds < 86_400:
        return f"{seconds // 3600} hours ago"
    return f"{seconds // 86_400} days ago"


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


__all__ = ["ResumePanel", "SessionItem", "begin_resume"]
