"""Tests for packaged and layered subagent discovery."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from codewright.subagent import Catalog, Source, builtin_definitions, load_catalog


def write_definition(
    path: Path,
    name: str,
    *,
    description: str | None = None,
    body: str = "role body",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description or f'{name} role'}\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_builtin_definitions_are_packaged_and_valid() -> None:
    definitions = builtin_definitions()

    assert [definition.name for definition in definitions] == [
        "explore",
        "general-purpose",
        "plan",
    ]
    assert all(definition.source is Source.BUILTIN for definition in definitions)
    assert all(definition.model == "inherit" for definition in definitions)
    assert all(definition.file_path.startswith("builtin:") for definition in definitions)
    assert all(definition.system_prompt for definition in definitions)


def test_missing_local_directories_still_returns_builtins(tmp_path: Path) -> None:
    catalog = load_catalog(tmp_path / "project", user_home=tmp_path / "home")

    assert [definition.name for definition in catalog.list()] == [
        "explore",
        "general-purpose",
        "plan",
    ]
    assert catalog.resolve("unknown") is None
    assert catalog.resolve("   ") is None


def test_project_overrides_user_and_builtin_case_insensitively(tmp_path: Path) -> None:
    project = tmp_path / "project"
    home = tmp_path / "home"
    write_definition(
        home / ".codewright" / "agents" / "explore.md",
        "explore",
        body="user explore",
    )
    write_definition(
        project / ".codewright" / "agents" / "explore.md",
        "explore",
        body="project explore",
    )

    catalog = load_catalog(project, user_home=home)

    selected = catalog.resolve("  ExPlOrE  ")
    assert selected is not None
    assert selected.source is Source.PROJECT
    assert selected.system_prompt == "project explore"
    assert [item.source for item in catalog.list_by_source(Source.USER)] == [Source.USER]
    assert [item.source for item in catalog.list_by_source(Source.PROJECT)] == [Source.PROJECT]


def test_last_sorted_duplicate_in_same_layer_wins(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    directory = tmp_path / "project" / ".codewright" / "agents"
    write_definition(directory / "a.md", "duplicate", body="first")
    write_definition(directory / "b.md", "duplicate", body="second")

    with caplog.at_level(logging.WARNING):
        catalog = load_catalog(tmp_path / "project", user_home=tmp_path / "home")

    selected = catalog.resolve("duplicate")
    assert selected is not None and selected.system_prompt == "second"
    assert "Subagent definition overridden name=duplicate" in caplog.text


def test_invalid_file_is_skipped_without_logging_body(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "catalog-sensitive-body"
    directory = tmp_path / "project" / ".codewright" / "agents"
    write_definition(directory / "good.md", "good")
    bad = directory / "bad.md"
    bad.write_text(
        f"---\nname: BAD\ndescription: invalid\n---\n{secret}\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        catalog = load_catalog(tmp_path / "project", user_home=tmp_path / "home")

    assert catalog.resolve("good") is not None
    assert catalog.resolve("bad") is None
    assert "Skipping invalid subagent" in caplog.text
    assert secret not in caplog.text


def test_catalog_ignores_non_markdown_and_symlink_entries(tmp_path: Path) -> None:
    directory = tmp_path / "project" / ".codewright" / "agents"
    write_definition(directory / "good.md", "good")
    (directory / "ignored.txt").write_text("ignored", encoding="utf-8")
    target = write_definition(tmp_path / "outside.md", "outside")
    try:
        (directory / "linked.md").symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    catalog = load_catalog(tmp_path / "project", user_home=tmp_path / "home")

    assert catalog.resolve("good") is not None
    assert catalog.resolve("outside") is None


def test_unsafe_source_directory_is_ignored(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    outside = tmp_path / "outside"
    write_definition(outside / "worker.md", "worker")
    agents = tmp_path / "project" / ".codewright" / "agents"
    agents.parent.mkdir(parents=True)
    try:
        agents.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with caplog.at_level(logging.WARNING):
        catalog = load_catalog(tmp_path / "project", user_home=tmp_path / "home")

    assert catalog.resolve("worker") is None
    assert "Unsafe subagent directory ignored source=project" in caplog.text


def test_fork_definition_is_internal_and_not_catalogued() -> None:
    catalog = Catalog()

    definition = catalog.fork_definition()

    assert definition.is_fork() is True
    assert definition.name == "__fork__"
    assert definition.system_prompt == ""
    assert catalog.resolve("__fork__") is None


def test_catalog_rejects_invalid_api_types() -> None:
    catalog = Catalog()
    with pytest.raises(TypeError, match="name"):
        catalog.resolve(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="source"):
        catalog.list_by_source("project")  # type: ignore[arg-type]
