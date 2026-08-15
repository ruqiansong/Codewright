"""Tests for the framework-light slash completion state model."""

from __future__ import annotations

import pytest

from codewright.command import Command, Kind, NopUI, Registry, register_builtins
from codewright.tui.complete import MAX_ROWS, CompletionMenu


async def hidden_handler(ui: NopUI, args: str) -> None:
    del ui, args


def builtins() -> Registry:
    registry = Registry()
    register_builtins(registry)
    return registry


def test_completion_activates_for_slash_and_filters_primary_names() -> None:
    registry = builtins()
    menu = CompletionMenu()

    menu.update("/", registry)
    assert menu.active is True
    assert len(menu.items) == 16

    menu.update("/s", registry)
    assert [command.name for command in menu.items] == ["session", "skill", "status"]
    assert menu.selected() is menu.items[0]


@pytest.mark.parametrize("value", ["", "hello", " /help", "/help\n/status"])
def test_completion_hides_for_non_leading_or_multiline_input(value: str) -> None:
    menu = CompletionMenu()
    menu.update("/", builtins())

    menu.update(value, builtins())

    assert menu.active is False
    assert menu.items == []
    assert menu.selected() is None


@pytest.mark.parametrize("value", ["/skill ", "/skill info", "/status\tvalue"])
def test_completion_hides_after_first_whitespace(value: str) -> None:
    menu = CompletionMenu()

    menu.update(value, builtins())

    assert menu.active is False
    assert menu.items == []


def test_completion_zero_match_stays_active_and_renders_notice() -> None:
    menu = CompletionMenu()
    menu.update("/zzz", builtins())

    assert menu.active is True
    assert menu.items == []
    assert menu.selected() is None
    assert menu.render(20).plain == "无匹配"


def test_completion_navigation_clamps_and_scrolls_to_max_rows() -> None:
    menu = CompletionMenu()
    menu.update("/", builtins())

    menu.move_up()
    assert menu.cursor == 0
    for _ in range(20):
        menu.move_down()

    assert menu.cursor == 15
    assert menu.offset == 15 - MAX_ROWS + 1
    assert len(menu.render(100).plain.splitlines()) == MAX_ROWS
    menu.move_up()
    assert menu.cursor == 14


def test_completion_render_is_width_bounded_and_marks_highlight() -> None:
    menu = CompletionMenu()
    menu.update("/", builtins())
    rendered = menu.render(24)

    assert all(len(line) <= 24 for line in rendered.plain.splitlines())
    assert any(span.style == "reverse" for span in rendered.spans)
    assert "↓" in rendered.plain


def test_completion_excludes_hidden_commands() -> None:
    registry = builtins()
    registry.register(
        Command(
            "secret",
            "Hidden command",
            Kind.LOCAL,
            hidden_handler,
            hidden=True,
        )
    )
    menu = CompletionMenu()

    menu.update("/s", registry)

    assert [command.name for command in menu.items] == ["session", "skill", "status"]


def test_completion_hide_resets_all_state_and_validates_width() -> None:
    menu = CompletionMenu()
    menu.update("/", builtins())
    menu.move_down()
    menu.hide()

    assert menu == CompletionMenu()
    with pytest.raises(ValueError, match="positive"):
        menu.render(0)
