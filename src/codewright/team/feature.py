"""Double-lock feature switches for optional Team behavior."""

from __future__ import annotations

import os
from collections.abc import Mapping

from codewright.config import Config

_TRUTHY = {"1", "true", "yes", "on"}


def fork_teammate_enabled(
    config: Config,
    environ: Mapping[str, str] | None = None,
) -> bool:
    values = os.environ if environ is None else environ
    return (
        config.enable_fork_teammate
        and values.get("CODEWRIGHT_FORK_TEAMMATE", "").casefold() in _TRUTHY
    )


__all__ = ["fork_teammate_enabled"]
