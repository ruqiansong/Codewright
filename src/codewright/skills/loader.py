"""Deterministic project and user Skill discovery with hot reload."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from codewright.skills.models import SkillDef, SkillSource
from codewright.skills.parser import SkillParseError, parse_skill_file

logger = logging.getLogger(__name__)

PROJECT_SKILLS_DIR = Path(".codewright/skills")
USER_SKILLS_DIR = Path(".codewright/skills")


class SkillLoader:
    """Load a stable, project-first catalog and hot-reload selected files."""

    def __init__(self, work_dir: str | Path, user_home: str | Path | None = None) -> None:
        self._project_dir = (Path(work_dir).resolve() / PROJECT_SKILLS_DIR).resolve()
        selected_home = Path.home() if user_home is None else Path(user_home)
        self._user_dir = (selected_home.resolve() / USER_SKILLS_DIR).resolve()
        self._skills: dict[str, SkillDef] = {}
        self._cache: dict[str, SkillDef] = {}
        self._lock = threading.RLock()

    @property
    def project_dir(self) -> Path:
        return self._project_dir

    @property
    def user_dir(self) -> Path:
        return self._user_dir

    def load_all(self) -> tuple[SkillDef, ...]:
        """Scan both layers and atomically replace the complete catalog."""
        loaded: dict[str, SkillDef] = {}
        for directory, source in (
            (self._project_dir, SkillSource.PROJECT),
            (self._user_dir, SkillSource.USER),
        ):
            for skill in self._scan_directory(directory, source):
                key = skill.name.casefold()
                if key in loaded:
                    logger.warning(
                        "Duplicate skill ignored name=%s source=%s",
                        skill.name,
                        source.value,
                    )
                    continue
                loaded[key] = skill

        ordered = dict(sorted(loaded.items()))
        with self._lock:
            self._skills = ordered
            self._cache = dict(ordered)
            return tuple(self._skills.values())

    def reload(self) -> tuple[SkillDef, ...]:
        """Rescan both layers using the same atomic load behavior."""
        return self.load_all()

    def get(self, name: str) -> SkillDef | None:
        """Return a freshly parsed Skill, falling back to its last good value."""
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        key = name.casefold()
        with self._lock:
            selected = self._skills.get(key)
            if selected is None:
                return None
            try:
                refreshed = parse_skill_file(
                    selected.source_path,
                    selected.source,
                    is_directory=selected.is_directory,
                )
            except SkillParseError as error:
                logger.warning(
                    "Skill hot reload failed name=%s error=%s",
                    selected.name,
                    type(error).__name__,
                )
                return self._cache.get(key)
            self._skills[key] = refreshed
            self._cache[key] = refreshed
            return refreshed

    def list(self) -> tuple[SkillDef, ...]:
        """Return a name-sorted immutable catalog snapshot."""
        with self._lock:
            return tuple(self._skills.values())

    def get_catalog(self) -> tuple[tuple[str, str], ...]:
        """Return the bounded catalog fields needed by later prompt integration."""
        with self._lock:
            return tuple((skill.name, skill.description) for skill in self._skills.values())

    def get_source_label(self, name: str) -> str | None:
        """Return the selected source label for one known Skill."""
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        with self._lock:
            skill = self._skills.get(name.casefold())
            return skill.source.value if skill is not None else None

    def _scan_directory(self, directory: Path, source: SkillSource) -> tuple[SkillDef, ...]:
        try:
            if not directory.exists():
                return ()
            if directory.is_symlink() or not directory.is_dir():
                logger.warning("Unsafe skill directory ignored source=%s", source.value)
                return ()
            entries = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as error:
            logger.warning(
                "Skill directory could not be scanned source=%s error=%s",
                source.value,
                type(error).__name__,
            )
            return ()

        loaded: list[SkillDef] = []
        for entry in entries:
            candidate = _candidate_file(entry)
            if candidate is None:
                continue
            path, is_directory = candidate
            try:
                loaded.append(parse_skill_file(path, source, is_directory=is_directory))
            except SkillParseError as error:
                logger.warning(
                    "Skipping invalid skill path=%s source=%s error=%s",
                    path,
                    source.value,
                    type(error).__name__,
                )
        return tuple(loaded)


def _candidate_file(entry: Path) -> tuple[Path, bool] | None:
    try:
        if entry.is_symlink():
            return None
        if entry.is_file() and entry.suffix.casefold() == ".md":
            return entry, False
        if not entry.is_dir():
            return None
        skill_file = entry / "SKILL.md"
        if skill_file.is_symlink() or not skill_file.is_file():
            return None
        return skill_file, True
    except OSError as error:
        logger.warning(
            "Skill candidate could not be inspected path=%s error=%s",
            entry,
            type(error).__name__,
        )
        return None


__all__ = ["PROJECT_SKILLS_DIR", "USER_SKILLS_DIR", "SkillLoader"]
