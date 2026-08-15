"""Tests for layered project instruction loading."""

from pathlib import Path

import pytest

from codewright.instructions import Loader


def write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def test_load_merges_three_layers_in_priority_order(tmp_path: Path) -> None:
    project = tmp_path / "project"
    home = tmp_path / "home"
    write(project / "codewright.md", "project")
    write(project / ".codewright" / "codewright.md", "project-config")
    write(home / ".codewright" / "codewright.md", "user")

    assert Loader(str(project), str(home)).load() == "project\n\nproject-config\n\nuser"


def test_missing_and_empty_layers_are_silent(tmp_path: Path) -> None:
    project = tmp_path / "project"
    home = tmp_path / "home"
    write(project / "codewright.md", "only project")
    write(project / ".codewright" / "codewright.md", "")

    assert Loader(str(project), str(home)).load() == "only project"


def test_include_expands_only_on_a_standalone_line(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write(project / "codewright.md", "before\n@include rules/style.md\nafter\n")
    write(project / "rules" / "style.md", "included")

    output = Loader(str(project), str(tmp_path / "home")).load()

    assert output == "before\nincluded\nafter\n"


def test_non_standalone_include_remains_literal(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write(project / "codewright.md", "Use @include rules/style.md here")

    assert Loader(str(project), str(tmp_path / "home")).load() == (
        "Use @include rules/style.md here"
    )


def test_nested_include_stops_after_five_file_levels(tmp_path: Path) -> None:
    project = tmp_path / "project"
    for level in range(1, 7):
        target = project / ("codewright.md" if level == 1 else f"level{level}.md")
        content = f"@include level{level + 1}.md" if level < 6 else "too deep"
        write(target, content)

    output = Loader(str(project), str(tmp_path / "home")).load()

    assert "超过最大嵌套深度" in output
    assert "too deep" not in output


def test_include_cycle_is_reported(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write(project / "codewright.md", "@include b.md")
    write(project / "b.md", "@include codewright.md")

    assert "检测到环路" in Loader(str(project), str(tmp_path / "home")).load()


def test_sibling_branches_can_include_same_file(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write(project / "codewright.md", "@include a.md\n@include b.md")
    write(project / "a.md", "@include common.md")
    write(project / "b.md", "@include common.md")
    write(project / "common.md", "shared")

    output = Loader(str(project), str(tmp_path / "home")).load()

    assert output.count("shared") == 2
    assert "检测到环路" not in output


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write(project / "codewright.md", "@include ../outside.md")
    write(tmp_path / "outside.md", "secret")

    output = Loader(str(project), str(tmp_path / "home")).load()

    assert "路径超出允许范围" in output
    assert "secret" not in output


def test_binary_and_invalid_utf8_includes_are_rejected(tmp_path: Path) -> None:
    project = tmp_path / "project"
    write(project / "codewright.md", "@include binary.dat\n@include invalid.dat")
    write(project / "binary.dat", b"text\x00more")
    write(project / "invalid.dat", b"\xff\xfe")

    output = Loader(str(project), str(tmp_path / "home")).load()

    assert "二进制文件不可读" in output
    assert "文件不是有效 UTF-8 文本" in output


def test_loader_validates_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Loader("")
    with pytest.raises(ValueError):
        Loader(str(tmp_path), max_depth=0)
    with pytest.raises(TypeError):
        Loader(str(tmp_path), max_depth=True)
