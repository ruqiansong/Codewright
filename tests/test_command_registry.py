"""Tests for collision-safe slash-command registration."""

from __future__ import annotations

from typing import cast

import pytest

from codewright.command import Command, CommandSource, Kind, Registry
from codewright.command.models import Handler


async def noop_handler(ui: object, args: str) -> None:
    del ui, args


def command(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    hidden: bool = False,
    source: CommandSource = "builtin",
) -> Command:
    return Command(
        name,
        f"Describe {name}",
        Kind.LOCAL,
        cast(Handler, noop_handler),
        aliases,
        hidden,
        source=source,
    )


def test_registry_registers_aliases_and_returns_sorted_visible_snapshot() -> None:
    registry = Registry()
    registry.register(command("status", aliases=("stat",)))
    registry.register(command("help"))

    assert registry.lookup("STATUS") is registry.lookup("Stat")
    assert [item.name for item in registry.visible()] == ["help", "status"]
    assert registry.count() == 2


def test_registry_rejects_name_and_alias_conflicts_without_partial_registration() -> None:
    registry = Registry()
    registry.register(command("help", aliases=("h",)))

    with pytest.raises(RuntimeError, match="command conflict: help"):
        registry.register(command("help"))
    with pytest.raises(RuntimeError, match="command conflict: h"):
        registry.register(command("history", aliases=("h", "hist")))

    assert registry.count() == 1
    assert registry.lookup("history") is None
    assert registry.lookup("hist") is None


def test_hidden_command_is_dispatchable_but_not_visible_or_completed() -> None:
    registry = Registry()
    secret = command("secret", aliases=("s",), hidden=True)
    registry.register(secret)
    registry.register(command("status"))

    assert registry.lookup("s") is secret
    assert [item.name for item in registry.visible()] == ["status"]
    assert [item.name for item in registry.prefix_match("/s")] == ["status"]


def test_prefix_match_uses_only_one_optional_slash_and_primary_names() -> None:
    registry = Registry()
    registry.register(command("session", aliases=("find",)))
    registry.register(command("status"))

    assert [item.name for item in registry.prefix_match("/S")] == ["session", "status"]
    assert registry.prefix_match("/find") == ()
    assert registry.prefix_match("//s") == ()
    assert registry.prefix_match("") == registry.visible()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"name": "Help"}, "lowercase"),
        ({"name": "/help"}, "slash"),
        ({"name": "two words"}, "whitespace"),
        ({"name": "help", "aliases": ("h", "h")}, "duplicates"),
        ({"name": "help", "aliases": ("help",)}, "command name"),
    ],
)
def test_command_rejects_invalid_identifiers(
    kwargs: dict[str, object],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        Command(
            cast(str, kwargs["name"]),
            "Description",
            Kind.LOCAL,
            cast(Handler, noop_handler),
            cast(tuple[str, ...], kwargs.get("aliases", ())),
        )


def test_command_validates_argument_and_source_metadata() -> None:
    value = command("skill-command", source="skill")

    assert value.source == "skill"
    assert value.accepts_args is False
    with pytest.raises(TypeError, match="accepts_args"):
        Command("bad", "Bad", Kind.LOCAL, cast(Handler, noop_handler), accepts_args=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source"):
        command("bad-source", source="external")  # type: ignore[arg-type]


def test_replace_source_atomically_adds_removes_and_sorts_commands() -> None:
    registry = Registry()
    registry.register(command("help"))
    registry.replace_source(
        "skill",
        (command("zeta", source="skill"), command("alpha", source="skill")),
    )

    assert [item.name for item in registry.visible()] == ["alpha", "help", "zeta"]
    registry.replace_source("skill", (command("beta", source="skill"),))
    assert [item.name for item in registry.visible()] == ["beta", "help"]
    assert registry.lookup("alpha") is None


def test_replace_source_conflict_rolls_back_without_partial_changes() -> None:
    registry = Registry()
    original = command("original", aliases=("old",), source="skill")
    registry.register(command("help", aliases=("h",)))
    registry.register(original)

    with pytest.raises(RuntimeError, match="command conflict: h"):
        registry.replace_source(
            "skill",
            (
                command("fresh", source="skill"),
                command("collision", aliases=("h",), source="skill"),
            ),
        )

    assert registry.lookup("old") is original
    assert registry.lookup("fresh") is None
    assert registry.count() == 2


def test_replace_source_state_is_isolated_between_registries() -> None:
    first = Registry()
    second = Registry()
    first.replace_source("skill", (command("only-first", source="skill"),))

    assert first.lookup("only-first") is not None
    assert second.lookup("only-first") is None


def test_replace_source_rejects_mismatched_command_source() -> None:
    registry = Registry()

    with pytest.raises(ValueError, match="replacement command source"):
        registry.replace_source("skill", (command("builtin"),))

    assert registry.count() == 0
