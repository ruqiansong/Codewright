"""Tests for deterministic layered Skill loading and hot reload."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from codewright.skills import SkillLoader, SkillSource


def write_skill(
    path: Path,
    name: str,
    *,
    description: str | None = None,
    body: str = "original body",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description or f'{name} description'}\n---\n{body}\n",
        encoding="utf-8",
    )
    return path


def test_loader_missing_directories_returns_empty_catalog(tmp_path: Path) -> None:
    loader = SkillLoader(tmp_path / "project", tmp_path / "home")

    assert loader.load_all() == ()
    assert loader.list() == ()
    assert loader.get_catalog() == ()
    assert loader.get("unknown") is None
    assert loader.get_source_label("unknown") is None


def test_loader_supports_file_and_directory_layouts_in_name_order(tmp_path: Path) -> None:
    project = tmp_path / "project"
    skills = project / ".codewright" / "skills"
    write_skill(skills / "z-file.md", "zeta")
    write_skill(skills / "a-directory" / "SKILL.md", "alpha")
    (skills / "ignored.txt").write_text("ignored", encoding="utf-8")
    (skills / "empty-directory").mkdir()
    loader = SkillLoader(project, tmp_path / "home")

    loaded = loader.load_all()

    assert [skill.name for skill in loaded] == ["alpha", "zeta"]
    assert loaded[0].is_directory is True
    assert loaded[1].is_directory is False
    assert all(skill.source is SkillSource.PROJECT for skill in loaded)
    assert loader.get_catalog() == (
        ("alpha", "alpha description"),
        ("zeta", "zeta description"),
    )


def test_project_skill_wins_over_user_skill(tmp_path: Path) -> None:
    project = tmp_path / "project"
    home = tmp_path / "home"
    write_skill(project / ".codewright" / "skills" / "shared.md", "shared", body="project")
    write_skill(home / ".codewright" / "skills" / "shared.md", "shared", body="user")
    write_skill(home / ".codewright" / "skills" / "user.md", "user-only")
    loader = SkillLoader(project, home)

    loader.load_all()

    shared = loader.get("SHARED")
    assert shared is not None
    assert shared.prompt_body == "project"
    assert loader.get_source_label("shared") == "project"
    assert loader.get_source_label("USER-ONLY") == "user"


def test_first_sorted_duplicate_within_layer_wins(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    skills = tmp_path / "project" / ".codewright" / "skills"
    write_skill(skills / "a.md", "duplicate", body="first")
    write_skill(skills / "b.md", "duplicate", body="second")
    loader = SkillLoader(tmp_path / "project", tmp_path / "home")

    with caplog.at_level(logging.WARNING):
        loader.load_all()

    selected = loader.get("duplicate")
    assert selected is not None and selected.prompt_body == "first"
    assert "Duplicate skill ignored name=duplicate" in caplog.text
    assert "second" not in caplog.text


def test_loader_skips_invalid_file_without_logging_body(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_body = "loader-sensitive-body"
    skills = tmp_path / "project" / ".codewright" / "skills"
    write_skill(skills / "good.md", "good")
    bad = skills / "bad.md"
    bad.write_text(f"---\nname: BAD\ndescription: invalid\n---\n{secret_body}\n")
    loader = SkillLoader(tmp_path / "project", tmp_path / "home")

    with caplog.at_level(logging.WARNING):
        loaded = loader.load_all()

    assert [skill.name for skill in loaded] == ["good"]
    assert "Skipping invalid skill" in caplog.text
    assert secret_body not in caplog.text


def test_get_hot_reloads_and_falls_back_to_last_good_value(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = write_skill(
        tmp_path / "project" / ".codewright" / "skills" / "hot.md",
        "hot",
        body="version one",
    )
    loader = SkillLoader(tmp_path / "project", tmp_path / "home")
    loader.load_all()

    write_skill(path, "hot", body="version two")
    refreshed = loader.get("HOT")
    assert refreshed is not None and refreshed.prompt_body == "version two"

    path.write_text("invalid frontmatter\nsecret failed body", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        fallback = loader.get("hot")

    assert fallback is not None and fallback.prompt_body == "version two"
    assert "Skill hot reload failed name=hot" in caplog.text
    assert "secret failed body" not in caplog.text


def test_reload_observes_additions_and_deletions(tmp_path: Path) -> None:
    skills = tmp_path / "project" / ".codewright" / "skills"
    old = write_skill(skills / "old.md", "old")
    loader = SkillLoader(tmp_path / "project", tmp_path / "home")
    loader.load_all()

    old.unlink()
    write_skill(skills / "new.md", "new")
    reloaded = loader.reload()

    assert [skill.name for skill in reloaded] == ["new"]
    assert loader.get("old") is None


def test_loader_ignores_symlink_files_and_directories(tmp_path: Path) -> None:
    project = tmp_path / "project"
    skills = project / ".codewright" / "skills"
    outside_file = write_skill(tmp_path / "outside.md", "outside-file")
    outside_dir = tmp_path / "outside-dir"
    write_skill(outside_dir / "SKILL.md", "outside-dir")
    skills.mkdir(parents=True)
    try:
        (skills / "linked.md").symlink_to(outside_file)
        (skills / "linked-dir").symlink_to(outside_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    loader = SkillLoader(project, tmp_path / "home")

    assert loader.load_all() == ()


def test_loader_handles_directory_scan_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    project = tmp_path / "project"
    skills = project / ".codewright" / "skills"
    skills.mkdir(parents=True)
    loader = SkillLoader(project, tmp_path / "home")
    original_iterdir = Path.iterdir

    def fail_selected_directory(path: Path):
        if path == skills.resolve():
            raise OSError("unsafe detail")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_selected_directory)
    with caplog.at_level(logging.WARNING):
        assert loader.load_all() == ()

    assert "Skill directory could not be scanned source=project error=OSError" in caplog.text
    assert "unsafe detail" not in caplog.text
