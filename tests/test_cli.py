"""Tests for CLI dependency assembly and process behavior."""

import runpy
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml

from codewright import __version__, cli
from codewright.agent import Agent
from codewright.agent.runtime import SessionRuntime
from codewright.config import ProviderConfig
from codewright.conversation import Conversation
from codewright.hook import Engine as HookEngine
from codewright.llm import (
    ChatResult,
    Message,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolDefinition,
)
from codewright.memory import Manager as MemoryManager
from codewright.permission import Engine, PermissionSetupError
from codewright.session import Writer
from codewright.skills import SkillLoader
from codewright.subagent import Catalog
from codewright.task import Manager as TaskManager
from codewright.task import SubagentApprovalBroker
from codewright.tool import InstallSkillTool, LoadSkillTool, Registry, Result

SYNTHETIC_SECRET = "cli-test-key-not-a-real-secret"


class FakeProvider:
    """Provider replacement proving CLI assembly stays vendor neutral."""

    provider_name = "fake"
    model_name = "fake-model"

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del request_context
        raise AssertionError("CLI must not call Provider.chat")

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del messages, parameters, tools, request_context
        yield StreamEvent.completed()


class FakeApp:
    """Record TUI construction without opening a real terminal."""

    instances: list["FakeApp"] = []

    def __init__(
        self,
        provider: FakeProvider,
        conversation: Conversation,
        registry: Registry,
        *,
        engine: Engine,
        working_directory: Path,
        version: str,
        stream: bool,
        runtime: SessionRuntime,
        writer: Writer,
        memory_manager: MemoryManager,
        instruction_text: str,
        base_prompt: str | None,
        sessions_dir: str,
        cleanup_task: object,
        skill_loader: SkillLoader,
        load_skill_tool: LoadSkillTool,
        install_skill_tool: InstallSkillTool,
        provider_configs: tuple[ProviderConfig, ...],
        hook_engine: HookEngine,
        subagent_catalog: Catalog,
        task_manager: TaskManager,
        approval_broker: SubagentApprovalBroker,
        worktree_manager: object | None,
        team_manager: object | None,
        main_allowed_tools: frozenset[str] | None,
        coordinator_prompt_suffix: str,
        coordinator_mode: bool,
    ) -> None:
        self.provider = provider
        self.conversation = conversation
        self.registry = registry
        self.engine = engine
        self.working_directory = working_directory
        self.version = version
        self.stream = stream
        self.runtime = runtime
        self.writer = writer
        self.memory_manager = memory_manager
        self.instruction_text = instruction_text
        self.base_prompt = base_prompt
        self.sessions_dir = sessions_dir
        self.cleanup_task = cleanup_task
        self.skill_loader = skill_loader
        self.load_skill_tool = load_skill_tool
        self.install_skill_tool = install_skill_tool
        self.provider_configs = provider_configs
        self.hook_engine = hook_engine
        self.subagent_catalog = subagent_catalog
        self.task_manager = task_manager
        self.approval_broker = approval_broker
        self.worktree_manager = worktree_manager
        self.team_manager = team_manager
        self.main_allowed_tools = main_allowed_tools
        self.coordinator_prompt_suffix = coordinator_prompt_suffix
        self.coordinator_mode = coordinator_mode
        self.main_agent = Agent(
            provider,
            registry,
            engine,
            runtime=runtime,
            hook_engine=hook_engine,
        )
        self.ran = False
        self.session_end_calls = 0
        self.instances.append(self)

    async def run_async(self) -> None:
        self.ran = True

    async def dispatch_session_end(self) -> None:
        self.session_end_calls += 1


class FakeHookEngine:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


@dataclass(slots=True)
class FakeMcpTool:
    """One remotely discovered tool registered during CLI assembly."""

    name: str = "mcp__demo__echo"
    description: str = "Echo text through a test MCP server."
    parameters: Mapping[str, object] = field(default_factory=lambda: {"type": "object"})
    read_only: bool = True

    async def execute(self, arguments_json: str) -> Result:
        return Result(arguments_json)


class FakeMcpManager:
    """Record MCP discovery and shutdown without opening transports."""

    def __init__(self) -> None:
        self.closed = False

    def tools(self) -> list[FakeMcpTool]:
        return [FakeMcpTool()]

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class McpHarness:
    managers: list[FakeMcpManager] = field(default_factory=list)
    loaded_roots: list[Path] = field(default_factory=list)


def write_config(path: Path, **overrides: Any) -> Path:
    """Write a valid CLI configuration with optional top-level overrides."""
    data: dict[str, object] = {
        "providers": [
            {
                "name": "deepseek",
                "protocol": "openai-compatible",
                "api_key": SYNTHETIC_SECRET,
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
            }
        ],
        "default_provider": "deepseek",
    }
    data.update(overrides)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def isolate_cli_mcp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> McpHarness:
    FakeApp.instances.clear()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    harness = McpHarness()

    def fake_load_config(root: Path) -> object:
        harness.loaded_roots.append(root)
        return object()

    async def fake_new_manager(config: object, version: str) -> FakeMcpManager:
        del config
        assert version == __version__
        manager = FakeMcpManager()
        harness.managers.append(manager)
        return manager

    monkeypatch.setattr(cli.mcp_client, "load_config", fake_load_config)
    monkeypatch.setattr(cli.mcp_client, "new_manager", fake_new_manager)
    return harness


def test_main_assembles_config_provider_conversation_and_tui(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolate_cli_mcp: McpHarness,
) -> None:
    config_path = write_config(tmp_path / "config.yaml", system_prompt="Custom prompt")
    provider = FakeProvider()
    selected_names: list[str] = []

    def fake_create(provider_config: ProviderConfig) -> FakeProvider:
        selected_names.append(provider_config.name)
        return provider

    monkeypatch.setattr(cli, "create_provider", fake_create)
    monkeypatch.setattr(cli, "CodewrightApp", FakeApp)
    hook_engine = FakeHookEngine()
    monkeypatch.setattr(cli.hook, "load", lambda root: hook_engine)

    exit_code = cli.main(["--config", str(config_path), "--provider", "deepseek"])

    assert exit_code == 0
    assert selected_names == ["deepseek"]
    assert len(FakeApp.instances) == 1
    app = FakeApp.instances[0]
    assert app.provider is provider
    assert app.conversation.messages()[0].content == "Custom prompt"
    assert [definition.name for definition in app.registry.definitions()] == [
        "read_file",
        "write_file",
        "edit_file",
        "bash",
        "glob",
        "grep",
        "load_skill",
        "install_skill",
        "TaskList",
        "TaskGet",
        "TaskStop",
        "SendMessage",
        "TeamCreate",
        "TeamDelete",
        "TeamTaskCreate",
        "TeamTaskGet",
        "TeamTaskList",
        "TeamTaskUpdate",
        "TeamSendMessage",
        "Agent",
        "mcp__demo__echo",
    ]
    assert app.working_directory == Path.cwd()
    assert app.engine.root == Path.cwd().resolve()
    assert app.version == __version__
    assert app.stream is True
    assert app.runtime.context_window == 128_000
    assert app.hook_engine is hook_engine
    assert app.session_end_calls == 1
    assert hook_engine.closed
    assert app.writer.path.parent.parent == tmp_path / ".codewright" / "sessions"
    assert app.memory_manager.project_store.load_index() == ""
    assert app.sessions_dir == str(tmp_path / ".codewright" / "sessions")
    assert app.base_prompt == "Custom prompt"
    assert app.registry.get("load_skill") is app.load_skill_tool
    assert app.registry.get("install_skill") is app.install_skill_tool
    assert app.install_skill_tool.read_only is False
    assert app.skill_loader.list() == ()
    assert [item.name for item in app.provider_configs] == ["deepseek"]
    assert app.ran is True
    assert isinstance(app.subagent_catalog, Catalog)
    assert app.main_agent.registry.get("Agent") is not None
    assert isolate_cli_mcp.loaded_roots == [Path.cwd()]
    assert len(isolate_cli_mcp.managers) == 1
    assert isolate_cli_mcp.managers[0].closed is True


def test_main_uses_default_system_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = write_config(tmp_path / "config.yaml")
    monkeypatch.setattr(cli, "create_provider", lambda _: FakeProvider())
    monkeypatch.setattr(cli, "CodewrightApp", FakeApp)

    assert cli.main(["--config", str(config_path)]) == 0

    assert "Codewright" in FakeApp.instances[0].conversation.messages()[0].content


def test_main_passes_non_streaming_provider_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_config(tmp_path / "config.yaml")
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_config["providers"][0]["stream"] = False
    config_path.write_text(yaml.safe_dump(raw_config), encoding="utf-8")
    monkeypatch.setattr(cli, "create_provider", lambda _: FakeProvider())
    monkeypatch.setattr(cli, "CodewrightApp", FakeApp)

    assert cli.main(["--config", str(config_path)]) == 0

    assert FakeApp.instances[0].stream is False


def test_missing_configuration_returns_safe_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(["--config", str(tmp_path / "missing.yaml")])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Configuration error: Configuration file not found" in captured.err
    assert "Traceback" not in captured.err


def test_unknown_provider_returns_configuration_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_config(tmp_path / "config.yaml")

    exit_code = cli.main(["--config", str(config_path), "--provider", "unknown"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Provider is not configured: unknown" in captured.err
    assert SYNTHETIC_SECRET not in captured.err


def test_tui_initialization_failure_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    isolate_cli_mcp: McpHarness,
) -> None:
    config_path = write_config(tmp_path / "config.yaml")
    monkeypatch.setattr(cli, "create_provider", lambda _: FakeProvider())

    def fail_app(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError(f"unsafe detail {SYNTHETIC_SECRET}")

    monkeypatch.setattr(cli, "CodewrightApp", fail_app)

    exit_code = cli.main(["--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Codewright could not start. Check the configuration and try again." in captured.err
    assert SYNTHETIC_SECRET not in captured.err
    assert "Traceback" not in captured.err
    assert isolate_cli_mcp.managers[0].closed is True


def test_version_option_uses_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as captured:
        cli.main(["--version"])

    assert captured.value.code == 0
    assert capsys.readouterr().out == f"Codewright v{__version__}\n"


def test_permission_setup_failure_is_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = write_config(tmp_path / "config.yaml")
    monkeypatch.setattr(cli, "create_provider", lambda _: FakeProvider())

    def fail_engine(root: Path) -> Engine:
        del root
        raise PermissionSetupError("unsafe filesystem detail")

    monkeypatch.setattr(cli, "new_engine", fail_engine)

    assert cli.main(["--config", str(config_path)]) == 2
    captured = capsys.readouterr()
    assert "safe project root" in captured.err
    assert "unsafe filesystem detail" not in captured.err


def test_module_entry_uses_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_main() -> int:
        nonlocal calls
        calls += 1
        return 7

    monkeypatch.setattr(cli, "main", fake_main)

    with pytest.raises(SystemExit) as captured:
        runpy.run_module("codewright.__main__", run_name="__main__")

    assert captured.value.code == 7
    assert calls == 1
