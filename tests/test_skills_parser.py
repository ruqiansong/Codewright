"""Tests for strict Skill parsing and session-scoped activation state."""

from __future__ import annotations

from pathlib import Path

import pytest

from codewright.skills import (
    ActiveSkills,
    SkillParseError,
    SkillSource,
    parse_frontmatter,
    parse_skill_file,
    substitute_arguments,
)


def skill_text(
    *,
    name: str = "code-review",
    description: str = "Review current changes",
    mode: str | None = None,
    context: str | None = None,
    model: str | None = None,
    body: str = "Review $ARGUMENTS carefully.",
) -> str:
    fields = [f"name: {name}", f"description: {description}"]
    if mode is not None:
        fields.append(f"mode: {mode}")
    if context is not None:
        fields.append(f"context: {context}")
    if model is not None:
        fields.append(f"model: {model}")
    return f"---\n{'\n'.join(fields)}\n---\n{body}\n"


def test_parse_skill_file_applies_defaults_and_absolute_metadata(tmp_path: Path) -> None:
    path = tmp_path / "review.md"
    path.write_text(skill_text(), encoding="utf-8")

    skill = parse_skill_file(path, SkillSource.PROJECT, is_directory=False)

    assert skill.name == "code-review"
    assert skill.description == "Review current changes"
    assert skill.prompt_body == "Review $ARGUMENTS carefully."
    assert skill.mode == "inline"
    assert skill.context == "full"
    assert skill.model is None
    assert skill.source is SkillSource.PROJECT
    assert skill.source_path == path.resolve()
    assert skill.source_dir == tmp_path.resolve()
    assert skill.is_directory is False


def test_parse_directory_skill_accepts_fork_fields(tmp_path: Path) -> None:
    directory = tmp_path / "review"
    directory.mkdir()
    path = directory / "SKILL.md"
    path.write_text(
        skill_text(mode="fork", context="recent", model="secondary"),
        encoding="utf-8",
    )

    skill = parse_skill_file(path, "user", is_directory=True)

    assert (skill.mode, skill.context, skill.model) == ("fork", "recent", "secondary")
    assert skill.source is SkillSource.USER
    assert skill.source_dir == directory.resolve()
    assert skill.is_directory is True


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("name: test\n---\nbody", "start"),
        ("---\nname: test\nbody", "not closed"),
        ("---\nname: [\n---\nbody", "invalid YAML"),
        ("---\n- name\n---\nbody", "mapping"),
        ("---\nname: test\ndescription: ok\n---\n   ", "body"),
    ],
)
def test_parse_frontmatter_rejects_malformed_documents(raw: str, message: str) -> None:
    with pytest.raises(SkillParseError, match=message):
        parse_frontmatter(raw)


@pytest.mark.parametrize(
    "frontmatter",
    [
        "description: Missing name",
        "name: test",
        "name: Test\ndescription: Invalid uppercase",
        "name: two_words\ndescription: Invalid underscore",
        "name: test\ndescription: ''",
        "name: test\ndescription: ' padded '",
        "name: test\ndescription: ok\nmode: invalid",
        "name: test\ndescription: ok\ncontext: invalid",
        "name: test\ndescription: ok\nmodel: ''",
        "name: test\ndescription: ok\nmodel: two words",
        "name: test\ndescription: ok\nunknown: value",
    ],
)
def test_parse_skill_file_rejects_invalid_metadata(
    tmp_path: Path,
    frontmatter: str,
) -> None:
    path = tmp_path / "invalid.md"
    path.write_text(f"---\n{frontmatter}\n---\nbody\n", encoding="utf-8")

    with pytest.raises(SkillParseError):
        parse_skill_file(path, "project", is_directory=False)


def test_parse_skill_file_rejects_missing_non_file_and_symlink(tmp_path: Path) -> None:
    with pytest.raises(SkillParseError, match="regular"):
        parse_skill_file(tmp_path / "missing.md", "project", is_directory=False)
    with pytest.raises(SkillParseError, match="regular"):
        parse_skill_file(tmp_path, "project", is_directory=False)

    target = tmp_path / "target.md"
    target.write_text(skill_text(), encoding="utf-8")
    link = tmp_path / "link.md"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    with pytest.raises(SkillParseError, match="symbolic"):
        parse_skill_file(link, "project", is_directory=False)


def test_parse_skill_file_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid.md"
    path.write_bytes(b"\xff\xfe")

    with pytest.raises(SkillParseError, match="UTF-8"):
        parse_skill_file(path, "project", is_directory=False)


def test_substitute_arguments_replaces_all_or_leaves_body_unchanged() -> None:
    assert substitute_arguments("$ARGUMENTS / $ARGUMENTS", "one two") == "one two / one two"
    assert substitute_arguments("No placeholder", "ignored") == "No placeholder"
    assert substitute_arguments("Value: $ARGUMENTS", "") == "Value: "


def test_active_skills_preserves_order_and_replaces_content(tmp_path: Path) -> None:
    active = ActiveSkills()
    source = tmp_path.resolve()

    active.activate("first", "old", source)
    original_snapshot = active.snapshot()
    active.activate("second", "body", source)
    active.activate("first", "new", source)

    assert active.names() == ("first", "second")
    assert [entry.body for entry in active.snapshot()] == ["new", "body"]
    assert original_snapshot[0].body == "old"
    assert len(active) == 2

    active.clear()
    assert active.snapshot() == ()
    assert active.names() == ()


@pytest.mark.parametrize(
    ("name", "body", "source"),
    [("", "body", Path("/tmp")), ("name", "", Path("/tmp")), ("name", "body", Path("x"))],
)
def test_active_skills_rejects_invalid_entries(name: str, body: str, source: Path) -> None:
    with pytest.raises(ValueError):
        ActiveSkills().activate(name, body, source)
