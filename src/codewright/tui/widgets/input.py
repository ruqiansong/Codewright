"""User input widget for Codewright."""

from textual.widgets import Input


class MessageInput(Input):
    """Single-line input for user messages and slash commands."""

    def __init__(
        self,
        *,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            placeholder="Ask Codewright... (/help for commands)",
            id=id,
            classes=classes,
            disabled=disabled,
        )
