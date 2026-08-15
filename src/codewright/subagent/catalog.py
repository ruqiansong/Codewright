"""Deterministic built-in, user, and project subagent discovery."""

from __future__ import annotations

import logging
import threading
from importlib.resources import files
from pathlib import Path

from codewright.permission import Mode
from codewright.subagent.definition import DEFAULT_MAX_TURNS, Definition, Source
from codewright.subagent.parser import DefinitionParseError, parse_definition, parse_file

logger = logging.getLogger(__name__)

AGENTS_DIR = Path(".codewright/agents")


class Catalog:
    """Thread-safe effective definitions plus source-level diagnostics."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._definitions: dict[str, Definition] = {}
        self._by_source: dict[Source, list[Definition]] = {source: [] for source in Source}

    def add_all(self, definitions: tuple[Definition, ...], source: Source) -> None:
        """Add one stable layer, warning while allowing later entries to win."""
        if not isinstance(source, Source):
            raise TypeError("source must be a Source")
        with self._lock:
            for definition in definitions:
                if definition.source is not source:
                    raise ValueError("definition source does not match catalog layer")
                key = definition.name.casefold()
                previous = self._definitions.get(key)
                if previous is not None:
                    logger.warning(
                        "Subagent definition overridden name=%s old_source=%s new_source=%s",
                        definition.name,
                        previous.source.value,
                        source.value,
                    )
                self._definitions[key] = definition
                self._by_source[source].append(definition)

    def resolve(self, name: str) -> Definition | None:
        """Resolve a normalized role name to its highest-precedence definition."""
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        key = name.strip().casefold()
        if not key:
            return None
        with self._lock:
            return self._definitions.get(key)

    def list(self) -> tuple[Definition, ...]:
        """Return effective definitions sorted by normalized name."""
        with self._lock:
            return tuple(self._definitions[key] for key in sorted(self._definitions))

    def list_by_source(self, source: Source) -> tuple[Definition, ...]:
        """Return every successfully parsed definition from one source layer."""
        if not isinstance(source, Source):
            raise TypeError("source must be a Source")
        with self._lock:
            return tuple(self._by_source[source])

    def fork_definition(self) -> Definition:
        """Return the internal definition used for a forced-background Fork."""
        return Definition(
            name="__fork__",
            description="Fork-based subagent",
            model="inherit",
            max_turns=DEFAULT_MAX_TURNS,
            permission_mode=Mode.DEFAULT,
            system_prompt="",
            file_path="internal:fork",
            source=Source.BUILTIN,
        )


def builtin_definitions() -> tuple[Definition, ...]:
    """Load packaged roles, failing fast because invalid built-ins are code bugs."""
    package = files("codewright.subagent.builtin")
    definitions: list[Definition] = []
    for resource in sorted(package.iterdir(), key=lambda item: item.name):
        if not resource.name.casefold().endswith(".md"):
            continue
        definitions.append(
            parse_definition(
                resource.read_bytes(),
                f"builtin:{resource.name}",
                Source.BUILTIN,
            )
        )
    return tuple(definitions)


def load_catalog(root: str | Path, *, user_home: str | Path | None = None) -> Catalog:
    """Load builtin, user, then project layers so higher precedence wins."""
    project_root = Path(root).resolve()
    selected_home = Path.home() if user_home is None else Path(user_home)
    catalog = Catalog()
    catalog.add_all(builtin_definitions(), Source.BUILTIN)
    catalog.add_all(_load_directory(selected_home.resolve() / AGENTS_DIR, Source.USER), Source.USER)
    catalog.add_all(_load_directory(project_root / AGENTS_DIR, Source.PROJECT), Source.PROJECT)
    return catalog


def _load_directory(directory: Path, source: Source) -> tuple[Definition, ...]:
    try:
        if not directory.exists():
            return ()
        if directory.is_symlink() or not directory.is_dir():
            logger.warning("Unsafe subagent directory ignored source=%s", source.value)
            return ()
        entries = sorted(directory.iterdir(), key=lambda path: path.name)
    except OSError as error:
        logger.warning(
            "Subagent directory could not be scanned source=%s error=%s",
            source.value,
            type(error).__name__,
        )
        return ()

    loaded: list[Definition] = []
    for path in entries:
        if path.suffix.casefold() != ".md":
            continue
        try:
            loaded.append(parse_file(path, source))
        except DefinitionParseError as error:
            logger.warning(
                "Skipping invalid subagent path=%s source=%s error=%s",
                path,
                source.value,
                type(error).__name__,
            )
    return tuple(loaded)


__all__ = ["AGENTS_DIR", "Catalog", "builtin_definitions", "load_catalog"]
