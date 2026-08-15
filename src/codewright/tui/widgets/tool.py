"""Claude Code style tool lifecycle presentation."""

from rich.console import RenderableType
from rich.text import Text
from textual.widgets import Static

from codewright.agent import ToolEvent

_LABELS = {
    "read_file": "Read",
    "write_file": "Write",
    "edit_file": "Edit",
    "bash": "Bash",
    "glob": "Glob",
    "grep": "Grep",
}


class ToolCallWidget(Static):
    """Display one tool call and update it in place when execution ends."""

    def __init__(self, event: ToolEvent) -> None:
        super().__init__(classes="tool-call tool-running")
        self.call_id = event.call_id
        self.tool_name = event.name
        self.argument_summary = event.argument_summary
        self.result_summary = ""
        self.is_error = False
        self.is_complete = False

    @property
    def heading(self) -> str:
        """Return the stable Claude Code style invocation heading."""
        label = _LABELS.get(self.tool_name, self.tool_name)
        return f"● {label}({self.argument_summary})"

    def complete(self, event: ToolEvent) -> None:
        """Replace Running status with the bounded execution summary."""
        self.result_summary = event.summary
        self.is_error = event.is_error
        self.is_complete = True
        self.remove_class("tool-running")
        self.add_class("tool-error" if event.is_error else "tool-success")
        self.refresh()

    def render(self) -> RenderableType:
        """Render the invocation followed by an indented status or result."""
        text = Text(self.heading, style="bold")
        if not self.is_complete:
            text.append("\n  Running…", style="dim")
            return text
        style = "red" if self.is_error else "green"
        for line in self.result_summary.splitlines() or ["(empty result)"]:
            text.append(f"\n  {line}", style=style)
        return text
