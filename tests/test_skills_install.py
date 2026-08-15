"""Tests for restricted remote Skill installation."""

from __future__ import annotations

import base64
from pathlib import Path

import httpx
import pytest

from codewright.skills import SkillInstaller, SkillInstallError
from codewright.skills.install import (
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_TOTAL_BYTES,
    parse_install_url,
)


def skill_body(name: str = "remote-skill") -> str:
    return f"---\nname: {name}\ndescription: Remote test Skill\n---\nDo remote work.\n"


def installer(
    tmp_path: Path,
    handler,
) -> SkillInstaller:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return SkillInstaller(tmp_path / "home" / ".codewright" / "skills", client=client)


@pytest.mark.parametrize(
    "url",
    [
        "http://skills.sh/o/r/s",
        "https://user@skills.sh/o/r/s",
        "https://skills.sh:443/o/r/s",
        "https://skills.sh/o/r/s?x=1",
        "https://skills.sh/o/r/s#fragment",
        "https://evil.example/o/r/s",
        "https://skills.sh/o/r",
        "https://github.com/o/r/tree/main",
        "https://github.com/o/r/blob/main/SKILL.md",
        "https://raw.githubusercontent.com/o/r/main/README.md",
        "https://raw.githubusercontent.com/o/r/main/%2e%2e/SKILL.md",
    ],
)
def test_parse_install_url_rejects_unsupported_or_ambiguous_urls(url: str) -> None:
    with pytest.raises(SkillInstallError) as caught:
        parse_install_url(url)
    assert caught.value.code == "invalid_url"


def test_parse_install_url_accepts_exact_supported_shapes() -> None:
    assert parse_install_url("https://skills.sh/owner/repo/review").kind == "skills_sh"
    tree = parse_install_url("https://github.com/owner/repo/tree/main/skills/review")
    assert (tree.kind, tree.ref, tree.path) == ("github_tree", "main", "skills/review")
    raw = parse_install_url(
        "https://raw.githubusercontent.com/owner/repo/main/skills/review/SKILL.md"
    )
    assert (raw.kind, raw.path) == ("raw", "skills/review/SKILL.md")


@pytest.mark.asyncio
async def test_raw_skill_installs_as_directory(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "raw.githubusercontent.com"
        return httpx.Response(200, text=skill_body(), request=request)

    value = installer(tmp_path, handler)
    result = await value.install(
        "https://raw.githubusercontent.com/owner/repo/main/review/SKILL.md"
    )

    assert result.name == "remote-skill"
    assert result.target_dir == tmp_path / "home/.codewright/skills/remote-skill"
    assert (result.target_dir / "SKILL.md").read_text() == skill_body()


@pytest.mark.asyncio
async def test_skills_sh_uses_detail_api_files(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/skills/owner/repo/review"
        return httpx.Response(
            200,
            json={
                "files": [
                    {"path": "SKILL.md", "content": skill_body()},
                    {"path": "references/guide.md", "content": "guide"},
                ]
            },
            request=request,
        )

    result = await installer(tmp_path, handler).install("https://skills.sh/owner/repo/review")

    assert (result.target_dir / "references/guide.md").read_text() == "guide"


@pytest.mark.asyncio
async def test_github_tree_recurses_contents_api_and_decodes_files(tmp_path: Path) -> None:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.path.endswith("/contents/skills/review"):
            return httpx.Response(
                200,
                json=[
                    {
                        "type": "file",
                        "path": "skills/review/SKILL.md",
                        "encoding": "base64",
                        "content": base64.b64encode(skill_body().encode()).decode(),
                    },
                    {"type": "dir", "path": "skills/review/references"},
                ],
                request=request,
            )
        return httpx.Response(
            200,
            json=[
                {
                    "type": "file",
                    "path": "skills/review/references/info.md",
                    "encoding": "base64",
                    "content": base64.b64encode(b"info").decode(),
                }
            ],
            request=request,
        )

    result = await installer(tmp_path, handler).install(
        "https://github.com/owner/repo/tree/main/skills/review"
    )

    assert len(requests) == 2
    assert all("api.github.com" in request and "ref=main" in request for request in requests)
    assert (result.target_dir / "references/info.md").read_bytes() == b"info"


@pytest.mark.asyncio
async def test_redirect_revalidates_host_and_limit(tmp_path: Path) -> None:
    def hostile(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "https://evil.example/SKILL.md"},
            request=request,
        )

    with pytest.raises(SkillInstallError) as caught:
        await installer(tmp_path, hostile).install(
            "https://raw.githubusercontent.com/o/r/main/SKILL.md"
        )
    assert caught.value.code == "unsafe_redirect"

    def loop(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": str(request.url)}, request=request)

    with pytest.raises(SkillInstallError) as caught:
        await installer(tmp_path, loop).install(
            "https://raw.githubusercontent.com/o/r/main/SKILL.md"
        )
    assert caught.value.code == "too_many_redirects"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("files", "code"),
    [
        ([{"path": "../SKILL.md", "content": "bad"}], "unsafe_path"),
        ([{"path": "/SKILL.md", "content": "bad"}], "unsafe_path"),
        ([{"path": "bad\\name", "content": "bad"}], "unsafe_path"),
        ([{"path": "bad\x00name", "content": "bad"}], "unsafe_path"),
        ([{"path": "a/b/c/d/SKILL.md", "content": "bad"}], "path_too_deep"),
        ([{"path": "SKILL.md", "content": "x" * (MAX_FILE_BYTES + 1)}], "file_too_large"),
        (
            [{"path": "SKILL.md", "content": skill_body()}]
            + [{"path": f"f-{index}.txt", "content": "x"} for index in range(MAX_FILES)],
            "too_many_files",
        ),
        (
            [
                {"path": "SKILL.md", "content": skill_body()},
                {"path": "nested/SKILL.md", "content": skill_body("other")},
            ],
            "invalid_skill_layout",
        ),
    ],
)
async def test_downloaded_files_enforce_layout_and_limits(
    tmp_path: Path,
    files: list[dict[str, str]],
    code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"files": files}, request=request)

    with pytest.raises(SkillInstallError) as caught:
        await installer(tmp_path, handler).install("https://skills.sh/o/r/s")

    assert caught.value.code == code
    parent = tmp_path / "home/.codewright"
    assert not parent.exists() or not any(parent.glob(".skills.install-*"))


@pytest.mark.asyncio
async def test_exact_file_total_count_and_depth_limits_are_accepted(tmp_path: Path) -> None:
    root = skill_body()
    contents = [
        {"path": "SKILL.md", "content": root},
        {"path": "a/b/c/file.bin", "content": "x" * MAX_FILE_BYTES},
    ]
    remaining = MAX_TOTAL_BYTES - len(root.encode()) - MAX_FILE_BYTES
    for index in range(6):
        contents.append({"path": f"full-{index}.bin", "content": "x" * MAX_FILE_BYTES})
        remaining -= MAX_FILE_BYTES
    contents.append({"path": "remainder.bin", "content": "x" * remaining})
    contents.extend(
        {"path": f"empty-{index}.txt", "content": ""} for index in range(MAX_FILES - len(contents))
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"files": contents}, request=request)

    result = await installer(tmp_path, handler).install("https://skills.sh/o/r/s")

    installed = tuple(path for path in result.target_dir.rglob("*") if path.is_file())
    assert len(installed) == MAX_FILES
    assert sum(path.stat().st_size for path in installed) == MAX_TOTAL_BYTES
    assert (result.target_dir / "a/b/c/file.bin").stat().st_size == MAX_FILE_BYTES


@pytest.mark.asyncio
async def test_total_size_one_byte_over_limit_is_rejected(tmp_path: Path) -> None:
    root = skill_body()
    contents = [{"path": "SKILL.md", "content": root}]
    remaining = MAX_TOTAL_BYTES - len(root.encode()) + 1
    index = 0
    while remaining:
        size = min(MAX_FILE_BYTES, remaining)
        contents.append({"path": f"file-{index}.bin", "content": "x" * size})
        remaining -= size
        index += 1

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"files": contents}, request=request)

    with pytest.raises(SkillInstallError) as caught:
        await installer(tmp_path, handler).install("https://skills.sh/o/r/s")
    assert caught.value.code == "skill_too_large"


@pytest.mark.asyncio
async def test_rejects_symlink_submodule_and_out_of_tree_github_entries(
    tmp_path: Path,
) -> None:
    entries = (
        {"type": "symlink", "path": "skills/review/SKILL.md"},
        {
            "type": "file",
            "path": "skills/review/SKILL.md",
            "submodule_git_url": "https://github.com/o/sub",
        },
        {
            "type": "file",
            "path": "outside/SKILL.md",
            "encoding": "base64",
            "content": base64.b64encode(skill_body().encode()).decode(),
        },
    )
    for entry in entries:

        def handler(request: httpx.Request, selected: dict[str, object] = entry) -> httpx.Response:
            return httpx.Response(200, json=[selected], request=request)

        with pytest.raises(SkillInstallError) as caught:
            await installer(tmp_path, handler).install(
                "https://github.com/o/r/tree/main/skills/review"
            )
        assert caught.value.code in {"unsafe_file", "unsafe_path"}


@pytest.mark.asyncio
async def test_http_invalid_parse_duplicate_and_staging_cleanup(tmp_path: Path) -> None:
    responses = iter(
        (
            httpx.Response(404),
            httpx.Response(200, json={"files": [{"path": "SKILL.md", "content": "bad"}]}),
            httpx.Response(
                200,
                json={"files": [{"path": "SKILL.md", "content": skill_body()}]},
            ),
            httpx.Response(
                200,
                json={"files": [{"path": "SKILL.md", "content": skill_body()}]},
            ),
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        response = next(responses)
        return httpx.Response(
            response.status_code,
            content=response.content,
            headers=response.headers,
            request=request,
        )

    value = installer(tmp_path, handler)
    for expected in ("http_error", "invalid_skill"):
        with pytest.raises(SkillInstallError) as caught:
            await value.install("https://skills.sh/o/r/s")
        assert caught.value.code == expected
    await value.install("https://skills.sh/o/r/s")
    with pytest.raises(SkillInstallError) as caught:
        await value.install("https://skills.sh/o/r/s")
    assert caught.value.code == "skill_exists"
    assert not list((tmp_path / "home/.codewright").glob(".skills.install-*"))


@pytest.mark.asyncio
async def test_total_timeout_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        import asyncio

        await asyncio.sleep(0.05)
        return httpx.Response(200, text=skill_body(), request=request)

    monkeypatch.setattr("codewright.skills.install.TOTAL_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(SkillInstallError) as caught:
        await installer(tmp_path, handler).install(
            "https://raw.githubusercontent.com/o/r/main/SKILL.md"
        )
    assert caught.value.code == "download_timeout"
