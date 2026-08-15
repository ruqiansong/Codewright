"""Validated value types for persistent Codewright notes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class NoteType(StrEnum):
    """Supported semantic note categories."""

    USER_PREFERENCE = "user_preference"
    CORRECTION_FEEDBACK = "correction_feedback"
    PROJECT_KNOWLEDGE = "project_knowledge"
    REFERENCE_MATERIAL = "reference_material"


@dataclass(frozen=True, slots=True)
class Note:
    """One parsed Markdown memory note."""

    type: NoteType
    title: str
    slug: str
    content: str
    filename: str
    created: datetime
    updated: datetime


@dataclass(frozen=True, slots=True)
class UpdateAction:
    """One untrusted create, update, or delete instruction returned by an LLM."""

    action: str
    level: str
    type: str = ""
    title: str = ""
    slug: str = ""
    content: str = ""
    filename: str = ""
