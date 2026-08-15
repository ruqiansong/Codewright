"""Command-line parsing and application dependency assembly."""

import argparse
import asyncio
import inspect
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import codewright.mcp as mcp_client
from codewright import __version__, hook
from codewright.agent.agent_tool import AgentTool
from codewright.agent.factory import build_defined_agent_and_conversation
from codewright.agent.filter import TEAM_COLLABORATION_TOOLS, filter_tool_names
from codewright.agent.runtime import SessionRuntime
from codewright.compact import (
    CompactCircuitBreaker,
    ContentReplacementState,
    RecoveryState,
    SessionContext,
    new_session_context,
)
from codewright.config import ConfigError, effective_context_window, load, select_provider
from codewright.conversation import Conversation
from codewright.coordinator import (
    COORDINATOR_PROMPT_SUFFIX,
    coordinator_allowed_tools,
    coordinator_enabled,
)
from codewright.instructions import Loader
from codewright.llm.factory import create_provider
from codewright.memory import Manager as MemoryManager
from codewright.permission import PermissionSetupError, new_engine
from codewright.prompt import build_system_prompt, render_skill_catalog
from codewright.session import Writer, clean_expired
from codewright.skills import SkillInstaller, SkillLoader
from codewright.subagent import load_catalog
from codewright.task import (
    Manager as TaskManager,
)
from codewright.task import (
    SendMessageTool,
    SubagentApprovalBroker,
    TaskGetTool,
    TaskListTool,
    TaskStopTool,
)
from codewright.team.backend.inprocess import InProcessBackend
from codewright.team.manager import Manager as TeamManager
from codewright.team.runtime import TeammateRuntimeFactory
from codewright.team.spawn import Spawner
from codewright.team.tools import register_team_tools
from codewright.tool import InstallSkillTool, LoadSkillTool, new_default_registry
from codewright.tui import CodewrightApp
from codewright.utils.logging import configure_logging
from codewright.worktree import Manager as WorktreeManager

DEFAULT_CONFIG_PATH = Path(".codewright/config.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build the public Codewright command-line parser."""
    parser = argparse.ArgumentParser(description="Codewright terminal AI coding assistant")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--provider", help="configured provider name")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="override the configured log level",
    )
    parser.add_argument("--version", action="version", version=f"Codewright v{__version__}")
    parser.add_argument(
        "--team-member",
        nargs=2,
        metavar=("TEAM", "MEMBER"),
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Assemble and run Codewright, returning a process exit code."""
    return asyncio.run(_amain(argv))


async def _amain(argv: Sequence[str] | None = None) -> int:
    """Assemble asynchronous dependencies and own their complete lifecycle."""
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level or "INFO")
    logger = logging.getLogger(__name__)

    try:
        config = load(args.config)
        effective_log_level = args.log_level or config.log_level
        configure_logging(
            effective_log_level,
            sensitive_values=(provider.api_key.get_secret_value() for provider in config.providers),
        )
        provider_config = select_provider(config, args.provider)
        logger.info(
            "Starting Codewright version=%s provider=%s model=%s",
            __version__,
            provider_config.name,
            provider_config.model,
        )
        root = Path.cwd()
        engine = new_engine(root)
        hook_engine = hook.load(root)
        provider = None
        app = None
        writer: Writer | None = None
        cleanup_task: asyncio.Task[object] | None = None
        worktree_sweep_task: asyncio.Task[object] | None = None
        worktree_manager: WorktreeManager | None = None
        mcp_manager: mcp_client.Manager | None = None
        task_manager: TaskManager | None = None
        approval_broker: SubagentApprovalBroker | None = None
        team_manager: TeamManager | None = None
        try:
            provider = create_provider(provider_config)
            skill_loader = SkillLoader(root)
            loaded_skills = skill_loader.load_all()
            instruction_text = Loader(str(root)).load()
            memory_manager = MemoryManager(
                str(root / ".codewright" / "memory"),
                str(Path.home() / ".codewright" / "memory"),
                provider,
                provider.model_name,
            )
            memory_text = memory_manager.load_index()
            base_prompt = config.system_prompt or None
            system_prompt = build_system_prompt(
                instruction_text,
                memory_text,
                base_prompt,
                render_skill_catalog(loaded_skills),
            )
            session_context, writer = _create_session_writer(root, provider.model_name)
            sessions_dir = str(root / ".codewright" / "sessions")
            cleanup_task = asyncio.create_task(
                asyncio.to_thread(clean_expired, sessions_dir, timedelta(days=30))
            )
            conversation = Conversation(
                system_prompt,
                on_append=writer.on_append,
                on_replace=writer.on_replace,
            )
            registry = new_default_registry(working_directory=engine.root)
            load_skill_tool = LoadSkillTool(skill_loader)
            registry.register(load_skill_tool)
            install_skill_tool = InstallSkillTool(
                SkillInstaller(skill_loader.user_dir),
                skill_loader,
            )
            registry.register(install_skill_tool)
            subagent_catalog = load_catalog(root)
            try:
                worktree_manager = WorktreeManager(root)
                _warn_missing_worktree_ignores(root, logger)
                worktree_sweep_task = asyncio.create_task(
                    worktree_manager.sweep_stale(datetime.now(UTC) - timedelta(hours=24))
                )
            except Exception:
                logger.warning("当前项目未启用 Worktree 管理")
            task_manager = TaskManager()
            approval_broker = SubagentApprovalBroker()
            registry.register(TaskListTool(task_manager))
            registry.register(TaskGetTool(task_manager))
            registry.register(TaskStopTool(task_manager))
            registry.register(SendMessageTool(task_manager))
            parent_holder: dict[str, object] = {}

            def build_team_runtime(*, initial_prompt: str, request, **kwargs):
                del kwargs
                parent = parent_holder.get("agent")
                if parent is None:
                    raise RuntimeError("main Agent is not ready")
                definition = subagent_catalog.resolve(
                    request.subagent_type or "general-purpose"
                )
                if definition is None:
                    raise ValueError("unknown teammate definition")
                child, child_conversation, owned = build_defined_agent_and_conversation(
                    parent,
                    definition,
                    initial_prompt,
                    config.providers,
                    model=request.model,
                )
                names = tuple(item.name for item in registry.definitions())
                child.allowed_tools = frozenset(
                    filter_tool_names(
                        names,
                        source=definition.source,
                        tools=definition.tools,
                        disallowed_tools=definition.disallowed_tools,
                        background=True,
                        team_member=True,
                    )
                )
                child.dont_ask = True
                if request.plan_mode_required:
                    from codewright.permission import Mode

                    child.permission_mode = Mode.PLAN
                return child, child_conversation, owned

            team_runtime_factory = TeammateRuntimeFactory(str(root), build_team_runtime)
            try:
                team_manager = TeamManager(
                    home_dir=Path.home(),
                    project_root=root,
                    worktree_manager=worktree_manager,
                    task_manager=task_manager,
                    runtime_factory=team_runtime_factory,
                )
            except OSError:
                logger.warning("Agent Team persistence is unavailable")
                team_manager = None
            spawner = (
                Spawner(team_manager, lambda team: InProcessBackend(task_manager))
                if team_manager is not None
                else None
            )
            active_callback: dict[str, object] = {}

            def update_team_visibility(team) -> None:
                callback = active_callback.get("callback")
                if callable(callback):
                    callback(team)

            if team_manager is not None:
                register_team_tools(
                    registry,
                    team_manager,
                    on_active_change=update_team_visibility,
                )
            agent_tool = AgentTool(
                subagent_catalog,
                task_manager,
                approval_broker,
                config.providers,
                enable_subagent_background=config.enable_subagent_background,
                worktree_manager=worktree_manager,
                team_hook=spawner,
            )
            registry.register(agent_tool)
            runtime = SessionRuntime(
                replacement=ContentReplacementState(),
                recovery=RecoveryState(),
                auto_tracking=CompactCircuitBreaker(),
                session=session_context,
                context_window=effective_context_window(provider_config),
            )
            mcp_config = mcp_client.load_config(root)
            mcp_manager = await mcp_client.new_manager(mcp_config, version=__version__)
            for tool in mcp_manager.tools():
                registry.register(tool)
            coordinator_mode = coordinator_enabled(config)
            registry_names = tuple(item.name for item in registry.definitions())
            initial_allowed = frozenset(
                name for name in registry_names if name not in TEAM_COLLABORATION_TOOLS
            )
            if team_manager is not None and team_manager.active_team is not None:
                initial_allowed = frozenset(registry_names)
            if coordinator_mode:
                initial_allowed &= coordinator_allowed_tools(registry_names)
            optional_app_arguments = {
                "team_manager": team_manager,
                "main_allowed_tools": initial_allowed,
                "coordinator_prompt_suffix": (
                    COORDINATOR_PROMPT_SUFFIX if coordinator_mode else ""
                ),
                "coordinator_mode": coordinator_mode,
            }
            app_parameters = inspect.signature(CodewrightApp).parameters.values()
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in app_parameters
            )
            if not accepts_kwargs:
                accepted_names = {parameter.name for parameter in app_parameters}
                optional_app_arguments = {
                    name: value
                    for name, value in optional_app_arguments.items()
                    if name in accepted_names
                }
            app = CodewrightApp(
                provider,
                conversation,
                registry,
                engine=engine,
                working_directory=engine.root,
                version=__version__,
                stream=provider_config.stream,
                runtime=runtime,
                writer=writer,
                memory_manager=memory_manager,
                instruction_text=instruction_text,
                base_prompt=base_prompt,
                sessions_dir=sessions_dir,
                cleanup_task=cleanup_task,
                skill_loader=skill_loader,
                load_skill_tool=load_skill_tool,
                install_skill_tool=install_skill_tool,
                provider_configs=config.providers,
                hook_engine=hook_engine,
                subagent_catalog=subagent_catalog,
                task_manager=task_manager,
                approval_broker=approval_broker,
                worktree_manager=worktree_manager,
                **optional_app_arguments,
            )
            main_agent = getattr(app, "main_agent", None)
            if main_agent is not None:
                agent_tool.set_parent(main_agent)
                parent_holder["agent"] = main_agent

                def apply_team_visibility(team) -> None:
                    allowed = frozenset(registry_names)
                    if team is None:
                        allowed -= TEAM_COLLABORATION_TOOLS
                    if coordinator_mode:
                        allowed &= coordinator_allowed_tools(registry_names)
                    main_agent.allowed_tools = allowed

                active_callback["callback"] = apply_team_visibility
            await app.run_async()
        finally:
            if app is not None:
                dispatch_session_end = getattr(app, "dispatch_session_end", None)
                if callable(dispatch_session_end):
                    try:
                        await dispatch_session_end()
                    except Exception as error:
                        logger.warning("SessionEnd hook failed error=%s", type(error).__name__)
                stop_consumers = getattr(app, "stop_subagent_consumers", None)
                if callable(stop_consumers):
                    try:
                        await stop_consumers()
                    except Exception as error:
                        logger.warning(
                            "Subagent consumers shutdown failed error=%s",
                            type(error).__name__,
                        )
            if task_manager is not None:
                try:
                    await task_manager.aclose()
                except Exception as error:
                    logger.warning("Task manager shutdown failed error=%s", type(error).__name__)
            if team_manager is not None:
                try:
                    await team_manager.aclose()
                except Exception as error:
                    logger.warning("Team manager shutdown failed error=%s", type(error).__name__)
            if approval_broker is not None:
                try:
                    await approval_broker.aclose()
                except Exception as error:
                    logger.warning("Approval broker shutdown failed error=%s", type(error).__name__)
            try:
                await hook_engine.aclose()
            except Exception as error:
                logger.warning("Hook engine shutdown failed error=%s", type(error).__name__)
            if mcp_manager is not None:
                try:
                    await mcp_manager.close()
                except Exception as error:
                    logger.warning("MCP shutdown failed error=%s", type(error).__name__)
            if writer is not None:
                try:
                    writer.close()
                except Exception as error:
                    logger.warning("Session writer shutdown failed error=%s", type(error).__name__)
            if cleanup_task is not None:
                try:
                    await _finish_cleanup_task(cleanup_task)
                except Exception as error:
                    logger.warning("Session cleanup shutdown failed error=%s", type(error).__name__)
            if worktree_sweep_task is not None:
                try:
                    await _finish_cleanup_task(worktree_sweep_task)
                except Exception as error:
                    logger.warning("Worktree sweep shutdown failed error=%s", type(error).__name__)
            if provider is not None:
                try:
                    await _close_provider(provider)
                except Exception as error:
                    logger.warning("Provider shutdown failed error=%s", type(error).__name__)
    except ConfigError as error:
        logger.error("Startup failed category=configuration")
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except PermissionSetupError:
        logger.error("Startup failed category=permission setup")
        print(
            "Permission setup error: Codewright could not establish a safe project root.",
            file=sys.stderr,
        )
        return 2
    except Exception as error:
        if logger.isEnabledFor(logging.DEBUG):
            logger.exception("Startup failed category=internal error=%s", type(error).__name__)
        else:
            logger.error("Startup failed category=internal error=%s", type(error).__name__)
        print("Codewright could not start. Check the configuration and try again.", file=sys.stderr)
        return 1

    logger.info("Codewright stopped normally")
    return 0


def _create_session_writer(root: Path, model: str) -> tuple[SessionContext, Writer]:
    """Create one collision-free session context and durable writer."""
    for _ in range(3):
        context = new_session_context(str(root))
        try:
            return context, Writer(context.session_dir, model)
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique session directory")


async def _finish_cleanup_task(task: asyncio.Task[object]) -> None:
    """Bound cleanup shutdown for failures before Textual takes ownership."""
    if task.done():
        await asyncio.gather(task, return_exceptions=True)
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=1.0)
    except TimeoutError:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def _close_provider(provider: object) -> None:
    """Best-effort close a Provider after normal or failed startup."""
    close = getattr(provider, "close", None)
    if close is None:
        return
    result = close()
    if result is not None:
        await result


def _warn_missing_worktree_ignores(root: Path, logger: logging.Logger) -> None:
    try:
        lines = {
            line.strip() for line in (root / ".gitignore").read_text(encoding="utf-8").splitlines()
        }
    except OSError:
        lines = set()
    required = {".codewright/worktrees/", ".codewright/worktree_session.json"}
    if not required.issubset(lines):
        logger.warning(".gitignore 缺少 Codewright Worktree 运行时路径")
