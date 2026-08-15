"""Environment plus capability based Team backend selection."""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping

from codewright.team.backend.iterm2 import ItermController
from codewright.team.types import BackendType


async def detect_backend(
    *,
    environ: Mapping[str, str] | None = None,
    iterm_controller: ItermController | None = None,
) -> BackendType:
    values = os.environ if environ is None else environ
    tmux_available = shutil.which("tmux") is not None
    if values.get("TMUX") and tmux_available:
        return BackendType.TMUX
    if (
        values.get("TERM_PROGRAM") == "iTerm.app"
        and iterm_controller is not None
        and await iterm_controller.probe()
    ):
        return BackendType.ITERM2
    if tmux_available:
        return BackendType.TMUX
    return BackendType.IN_PROCESS


__all__ = ["detect_backend"]
