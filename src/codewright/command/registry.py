"""Collision-safe registry for slash-command definitions."""

from __future__ import annotations

from collections.abc import Iterable

from codewright.command.models import Command, CommandSource


class Registry:
    """Store command names and aliases with deterministic visible ordering."""

    def __init__(self) -> None:
        self._by_name: dict[str, Command] = {}
        self._commands: list[Command] = []

    def register(self, command: Command) -> None:
        """Register one complete command or fail without partial mutation."""
        if not isinstance(command, Command):
            raise TypeError("command must be a Command")
        keys = (command.name, *command.aliases)
        for key in keys:
            if key in self._by_name:
                raise RuntimeError(f"command conflict: {key}")
        for key in keys:
            self._by_name[key] = command
        self._commands.append(command)
        self._commands.sort(key=lambda item: item.name)

    def lookup(self, name: str) -> Command | None:
        """Return a command by case-insensitive main name or alias."""
        if not isinstance(name, str):
            raise TypeError("name must be a string")
        return self._by_name.get(name.casefold())

    def visible(self) -> tuple[Command, ...]:
        """Return visible commands sorted by main name."""
        return tuple(command for command in self._commands if not command.hidden)

    def prefix_match(self, prefix: str) -> tuple[Command, ...]:
        """Match visible main names by a case-insensitive slash prefix."""
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        normalized = prefix[1:] if prefix.startswith("/") else prefix
        normalized = normalized.casefold()
        return tuple(command for command in self.visible() if command.name.startswith(normalized))

    def count(self) -> int:
        """Return the number of primary command definitions."""
        return len(self._commands)

    def replace_source(self, source: CommandSource, commands: Iterable[Command]) -> None:
        """Atomically replace every command from one dynamic source."""
        if not isinstance(source, str) or source not in {"builtin", "skill"}:
            raise ValueError("source must be builtin or skill")
        replacements = tuple(commands)
        if not all(isinstance(command, Command) for command in replacements):
            raise TypeError("commands must contain only Command values")
        if any(command.source != source for command in replacements):
            raise ValueError(f"replacement command source must be {source}")

        retained = tuple(command for command in self._commands if command.source != source)
        by_name: dict[str, Command] = {}
        for command in (*retained, *replacements):
            for key in (command.name, *command.aliases):
                if key in by_name:
                    raise RuntimeError(f"command conflict: {key}")
                by_name[key] = command

        self._by_name = by_name
        self._commands = sorted((*retained, *replacements), key=lambda item: item.name)
