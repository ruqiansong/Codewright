"""Framework-light slash-command completion state and Rich rendering."""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.text import Text

from codewright.command import Command, Registry

MAX_ROWS = 8
_ASCII_WHITESPACE = frozenset(" \t\n\r\f\v")


@dataclass(slots=True)
class CompletionMenu:
    """Track filtered commands, keyboard selection, and scroll position."""

    items: list[Command] = field(default_factory=list)
    cursor: int = 0
    offset: int = 0
    active: bool = False

    def update(self, input_text: str, registry: Registry) -> None:
        if not isinstance(input_text, str):
            raise TypeError("input_text must be a string")
        if not isinstance(registry, Registry):
            raise TypeError("registry must be a Registry")
        if any(
            character in _ASCII_WHITESPACE for character in input_text
        ) or not input_text.startswith("/"):
            self.hide()
            return
        self.items = list(registry.prefix_match(input_text))
        self.active = True
        if not self.items:
            self.cursor = 0
            self.offset = 0
            return
        self.cursor = min(self.cursor, len(self.items) - 1)
        self._ensure_visible()

    def move_up(self) -> None:
        if self.items:
            self.cursor = max(0, self.cursor - 1)
            self._ensure_visible()

    def move_down(self) -> None:
        if self.items:
            self.cursor = min(len(self.items) - 1, self.cursor + 1)
            self._ensure_visible()

    def selected(self) -> Command | None:
        if not self.active or not self.items:
            return None
        return self.items[self.cursor]

    def hide(self) -> None:
        self.items.clear()
        self.cursor = 0
        self.offset = 0
        self.active = False

    def render(self, width: int) -> Text:
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("width must be a positive integer")
        if not self.active:
            return Text()
        if not self.items:
            value = Text("无匹配", style="dim")
            value.truncate(width, overflow="ellipsis")
            return value

        visible = self.items[self.offset : self.offset + MAX_ROWS]
        name_width = max(len(command.name) for command in visible) + 1
        output = Text()
        for visible_index, command in enumerate(visible):
            absolute_index = self.offset + visible_index
            has_more_above = visible_index == 0 and self.offset > 0
            has_more_below = visible_index == len(visible) - 1 and self.offset + len(visible) < len(
                self.items
            )
            prefix = "↑ " if has_more_above else "↓ " if has_more_below else "  "
            line = Text(f"{prefix}/{command.name.ljust(name_width)} {command.description}")
            line.truncate(width, overflow="ellipsis")
            if absolute_index == self.cursor:
                line.stylize("reverse")
            output.append(line)
            if visible_index < len(visible) - 1:
                output.append("\n")
        return output

    def _ensure_visible(self) -> None:
        if self.cursor < self.offset:
            self.offset = self.cursor
        elif self.cursor >= self.offset + MAX_ROWS:
            self.offset = self.cursor - MAX_ROWS + 1
