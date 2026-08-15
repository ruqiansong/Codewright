"""Conversation message rendering with Markdown fallback."""

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.widgets import Static

from codewright.llm import MessageRole


class ConversationMessage(Static):
    """Render one user or assistant conversation message."""

    def __init__(self, role: MessageRole, content: str, *, streaming: bool = False) -> None:
        super().__init__(classes=f"message {role.value}-message")
        self.role = role
        self._message_content = content
        self._streaming = streaming

    @property
    def message_content(self) -> str:
        """Return the currently visible message text."""
        return self._message_content

    def append_delta(self, text: str) -> None:
        """Append one provider delta and refresh the plain-text stream view."""
        self._message_content += text
        self._streaming = True
        self.refresh()

    def finalize(self) -> None:
        """Switch a complete assistant response to Markdown rendering."""
        self._streaming = False
        self.refresh()

    def mark_incomplete(self, reason: str) -> None:
        """Keep partial text visible and label it as incomplete."""
        separator = "\n\n" if self._message_content else ""
        self._message_content = f"{self._message_content}{separator}[{reason}]"
        self._streaming = True
        self.refresh()

    def render(self) -> RenderableType:
        """Render assistant Markdown, falling back to lossless plain text."""
        if self.role is MessageRole.ASSISTANT and not self._streaming:
            try:
                return Markdown(self._message_content)
            except Exception:
                return Text(self._message_content)
        return Text(self._message_content)
