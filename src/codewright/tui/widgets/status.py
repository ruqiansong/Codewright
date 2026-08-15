"""Request status, Agent mode, iteration, and token accounting widget."""

from textual.widgets import Static

from codewright.llm import TokenUsage

_MODE_LABELS = {
    "default": "DEFAULT",
    "acceptEdits": "ACCEPT EDITS",
    "plan": "PLAN",
    "bypassPermissions": "BYPASS",
}


class StatusWidget(Static):
    """Display request state together with persistent Agent context."""

    def __init__(
        self,
        *,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        self._message = "Ready"
        self._mode = "DEFAULT"
        self._iteration = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._coordinator = False
        super().__init__("", id=id, classes=classes, disabled=disabled)
        self._refresh_status()

    @property
    def input_tokens(self) -> int:
        return self._input_tokens

    @property
    def output_tokens(self) -> int:
        return self._output_tokens

    def set_mode(self, mode: str) -> None:
        """Update the persistent four-level permission mode badge."""
        self._mode = _MODE_LABELS.get(mode, mode.upper())
        self._refresh_status()

    def set_iteration(self, iteration: int) -> None:
        """Update the current Agent iteration; zero hides it."""
        self._iteration = iteration
        self._refresh_status()

    def set_coordinator(self, enabled: bool) -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean")
        self._coordinator = enabled
        self._refresh_status()

    def add_usage(self, usage: TokenUsage) -> None:
        """Accumulate one model request's token usage."""
        self._input_tokens += usage.input_tokens
        self._output_tokens += usage.output_tokens
        self._refresh_status()

    def reset_usage(self) -> None:
        """Reset session-scoped token and iteration counters."""
        self._input_tokens = 0
        self._output_tokens = 0
        self._iteration = 0
        self._refresh_status()

    def show_ready(self, message: str = "Ready") -> None:
        self._set_message(message)

    def show_waiting(self) -> None:
        """Show the state before the first text event arrives."""
        self._set_message("Thinking...")

    def show_streaming(self) -> None:
        """Show active response generation."""
        self._set_message("Generating...")

    def show_tool(self, name: str) -> None:
        """Show active local tool execution."""
        self._set_message(f"Running {name}...")

    def show_approving(self, name: str) -> None:
        """Show that one tool call is awaiting user approval."""
        self._set_message(f"Approval required for {name}")

    def show_cancelling(self) -> None:
        """Show that the active Agent turn is being cancelled."""
        self._set_message("Cancelling...")

    def show_complete(self, elapsed_seconds: float) -> None:
        """Show successful completion and total request duration."""
        self._set_message(f"Completed in {elapsed_seconds:.2f}s")

    def show_error(self, message: str) -> None:
        """Show a safe provider error."""
        self._set_message(f"Error: {message}")

    def show_cancelled(self, elapsed_seconds: float) -> None:
        """Show cancellation and the elapsed duration before it occurred."""
        self._set_message(f"Cancelled after {elapsed_seconds:.2f}s")

    def _set_message(self, message: str) -> None:
        self._message = message
        self._refresh_status()

    def _refresh_status(self) -> None:
        iteration = f" | Iteration {self._iteration}" if self._iteration else ""
        usage = f"↑{_compact(self._input_tokens)} ↓{_compact(self._output_tokens)} tok"
        coordinator = "[COORDINATOR] | " if self._coordinator else ""
        self.update(f"{coordinator}{self._message} | {self._mode}{iteration} | {usage}")


def _compact(value: int) -> str:
    if value < 1_000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1_000:.1f}k".replace(".0k", "k")
    return f"{value / 1_000_000:.1f}m".replace(".0m", "m")
