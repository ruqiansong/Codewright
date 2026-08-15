"""Tests for the model-facing load_skill tool."""

from pathlib import Path

import httpx
import pytest

from codewright.skills import SkillInstaller, SkillLoader
from codewright.tool import InstallSkillTool, LoadSkillTool, Registry


class RecordingAgent:
    def __init__(self) -> None:
        self.activations: list[tuple[str, str, Path]] = []

    def activate_skill(self, name: str, body: str, source_dir: Path) -> None:
        self.activations.append((name, body, source_dir))


def write_skill(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: review\ndescription: Review changes\n---\n{body}\n",
        encoding="utf-8",
    )


def loader_with_skill(tmp_path: Path, body: str = "version one") -> tuple[SkillLoader, Path]:
    path = tmp_path / "project" / ".codewright" / "skills" / "review.md"
    write_skill(path, body)
    loader = SkillLoader(tmp_path / "project", tmp_path / "home")
    loader.load_all()
    return loader, path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    ["not-json", "[]", "{}", '{"name":""}', '{"name":" review "}', '{"name":"review","extra":1}'],
)
async def test_load_skill_rejects_invalid_arguments(tmp_path: Path, arguments: str) -> None:
    loader, _ = loader_with_skill(tmp_path)

    result = await LoadSkillTool(loader).execute(arguments)

    assert result.is_error is True
    assert result.error_code == "invalid_arguments"


@pytest.mark.asyncio
async def test_load_skill_reports_unknown_and_uninitialized_states(tmp_path: Path) -> None:
    loader, _ = loader_with_skill(tmp_path)
    tool = LoadSkillTool(loader)

    unknown = await tool.execute('{"name":"missing"}')
    uninitialized = await tool.execute('{"name":"review"}')

    assert unknown.error_code == "unknown_skill"
    assert uninitialized.error_code == "not_initialized"


@pytest.mark.asyncio
async def test_load_skill_uses_hot_reloaded_body_and_returns_safe_result(
    tmp_path: Path,
) -> None:
    loader, path = loader_with_skill(tmp_path)
    agent = RecordingAgent()
    tool = LoadSkillTool(loader)
    tool.set_agent(agent)  # type: ignore[arg-type]
    write_skill(path, "latest secret SOP")

    result = await tool.execute('{"name":"REVIEW"}')

    assert result.is_error is False
    assert result.content == "Skill activated: review"
    assert "latest secret SOP" not in result.content
    assert agent.activations == [("review", "latest secret SOP", path.parent.resolve())]


def test_load_skill_is_read_only_and_registered_without_permission_gate(tmp_path: Path) -> None:
    loader, _ = loader_with_skill(tmp_path)
    registry = Registry()
    registry.register(LoadSkillTool(loader))

    assert registry.is_read_only("load_skill") is True
    assert [definition.name for definition in registry.read_only_definitions()] == ["load_skill"]


def install_tool(
    tmp_path: Path,
    handler,
) -> tuple[InstallSkillTool, SkillLoader]:
    loader = SkillLoader(tmp_path / "project", tmp_path / "home")
    loader.load_all()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    tool = InstallSkillTool(SkillInstaller(loader.user_dir, client=client), loader)
    return tool, loader


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    ["bad", "[]", "{}", '{"url":""}', '{"url":" padded "}', '{"url":"x","extra":1}'],
)
async def test_install_skill_rejects_invalid_arguments(
    tmp_path: Path,
    arguments: str,
) -> None:
    tool, _ = install_tool(
        tmp_path,
        lambda request: httpx.Response(500, request=request),
    )

    result = await tool.execute(arguments)

    assert result.is_error is True
    assert result.error_code == "invalid_arguments"


@pytest.mark.asyncio
async def test_install_skill_installs_reloads_and_invokes_refresh(tmp_path: Path) -> None:
    body = "---\nname: installed\ndescription: Installed Skill\n---\nPRIVATE BODY\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body, request=request)

    tool, loader = install_tool(tmp_path, handler)
    refreshed: list[tuple[str, ...]] = []
    tool.set_refresh_callback(
        lambda skills: refreshed.append(tuple(skill.name for skill in skills))
    )

    result = await tool.execute('{"url":"https://raw.githubusercontent.com/o/r/main/SKILL.md"}')

    assert result.content == "Skill installed: installed"
    assert result.is_error is False
    assert refreshed == [("installed",)]
    assert [skill.name for skill in loader.list()] == ["installed"]
    assert "PRIVATE BODY" not in result.content


@pytest.mark.asyncio
async def test_install_skill_reports_stable_download_and_refresh_failures(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="---\nname: installed\ndescription: Installed Skill\n---\nbody\n",
            request=request,
        )

    invalid_tool, _ = install_tool(
        tmp_path / "invalid",
        lambda request: httpx.Response(404, request=request),
    )
    invalid = await invalid_tool.execute(
        '{"url":"https://raw.githubusercontent.com/o/r/main/SKILL.md"}'
    )
    assert invalid.error_code == "http_error"

    tool, loader = install_tool(tmp_path / "refresh", handler)
    refresh_failed = await tool.execute(
        '{"url":"https://raw.githubusercontent.com/o/r/main/SKILL.md"}'
    )
    assert refresh_failed.error_code == "refresh_failed"
    assert "Run /skill reload" in refresh_failed.content
    assert [skill.name for skill in loader.list()] == ["installed"]


@pytest.mark.asyncio
async def test_install_network_failure_does_not_expose_response_or_url(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "REMOTE-SECRET-RESPONSE"
    url = "https://raw.githubusercontent.com/o/r/main/SKILL.md"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(secret, request=request)

    tool, _ = install_tool(tmp_path, handler)
    result = await tool.execute(f'{{"url":"{url}"}}')

    assert result.error_code == "download_failed"
    assert result.content == "Skill download failed."
    assert secret not in result.content + caplog.text
    assert url not in caplog.text


def test_install_skill_is_side_effecting_and_excluded_from_read_only_tools(
    tmp_path: Path,
) -> None:
    tool, _ = install_tool(
        tmp_path,
        lambda request: httpx.Response(500, request=request),
    )
    registry = Registry()
    registry.register(tool)

    assert registry.is_read_only("install_skill") is False
    assert registry.read_only_definitions() == ()
