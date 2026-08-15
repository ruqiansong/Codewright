"""Stable public interface for Codewright permission decisions."""

from codewright.permission.engine import Engine, new_engine
from codewright.permission.matcher import (
    ExactMatcher,
    GlobMatcher,
    Matcher,
    NotMatcher,
    RegexMatcher,
    compile_matcher,
)
from codewright.permission.models import (
    Category,
    Decision,
    Mode,
    Outcome,
    PermissionSetupError,
    parse_mode,
)

__all__ = [
    "Category",
    "Decision",
    "Engine",
    "ExactMatcher",
    "GlobMatcher",
    "Matcher",
    "Mode",
    "NotMatcher",
    "Outcome",
    "PermissionSetupError",
    "RegexMatcher",
    "compile_matcher",
    "new_engine",
    "parse_mode",
]
