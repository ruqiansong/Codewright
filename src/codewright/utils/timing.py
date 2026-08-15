"""Monotonic request timing helpers."""

from dataclasses import dataclass, field
from time import monotonic


@dataclass(slots=True)
class RequestTimer:
    """Measure one request without depending on wall-clock changes."""

    _started_at: float = field(default_factory=monotonic)

    @property
    def elapsed_seconds(self) -> float:
        """Return a non-negative elapsed duration in seconds."""
        return max(0.0, monotonic() - self._started_at)
