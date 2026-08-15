"""Textual application lifecycle for Codewright."""

import asyncio
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from textual.app import App

from codewright import __version__
from codewright.agent import Agent
from codewright.agent.runtime import SessionRuntime
from codewright.command import Registry as CommandRegistry
from codewright.command import build_skill_commands, register_builtins
from codewright.config import ProviderConfig
from codewright.conversation import Conversation
from codewright.hook import DispatchResult, Payload
from codewright.hook import Engine as HookEngine
from codewright.hook import Event as HookEvent
from codewright.hook import Rule as HookRule
from codewright.llm import Provider
from codewright.memory import Manager
from codewright.permission import Engine, Mode
from codewright.prompt import render_skill_catalog
from codewright.session import Writer
from codewright.skills import SkillDef, SkillExecutor, SkillLoader
from codewright.subagent import Catalog
from codewright.task import Manager as TaskManager
from codewright.task import SubagentApprovalBroker
from codewright.tool import InstallSkillTool, LoadSkillTool, Registry
from codewright.tui.screens.chat import ChatScreen
from codewright.worktree import Manager as WorktreeManager

logger = logging.getLogger(__name__)


@runtime_checkable
class _AsyncCloseable(Protocol):
    async def close(self) -> None:
        """Release asynchronous resources."""
        ...


class CodewrightApp(App[None]):
    """Codewright terminal application with Agent-driven tool execution."""

    CSS_PATH = "styles.tcss"
    TITLE = "Codewright"

    def __init__(
        self,
        provider: Provider,
        conversation: Conversation,
        registry: Registry | None = None,
        *,
        engine: Engine,
        working_directory: Path | None = None,
        version: str = __version__,
        stream: bool = True,
        runtime: SessionRuntime | None = None,
        writer: Writer | None = None,
        memory_manager: Manager | None = None,
        instruction_text: str = "",
        base_prompt: str | None = None,
        sessions_dir: str | None = None,
        cleanup_task: asyncio.Task[object] | None = None,
        command_registry: CommandRegistry | None = None,
        skill_loader: SkillLoader | None = None,
        load_skill_tool: LoadSkillTool | None = None,
        install_skill_tool: InstallSkillTool | None = None,
        provider_configs: tuple[ProviderConfig, ...] = (),
        hook_engine: HookEngine | None = None,
        subagent_catalog: Catalog | None = None,
        task_manager: TaskManager | None = None,
        approval_broker: SubagentApprovalBroker | None = None,
        worktree_manager: WorktreeManager | None = None,
        team_manager: object | None = None,
        main_allowed_tools: frozenset[str] | None = None,
        coordinator_prompt_suffix: str = "",
        coordinator_mode: bool = False,
    ) -> None:
        super().__init__()
        selected_directory = (working_directory or Path.cwd()).resolve()
        self._provider = provider
        self._conversation = conversation
        self._tool_registry = registry or Registry()
        combined_base_prompt = "\n\n".join(
            part for part in (base_prompt, coordinator_prompt_suffix) if part
        ) or None
        self._agent = Agent(
            provider,
            self._tool_registry,
            engine,
            version=version,
            runtime=runtime,
            memory_manager=memory_manager,
            instruction_text=instruction_text,
            base_prompt=combined_base_prompt,
            hook_engine=hook_engine,
            allowed_tools=main_allowed_tools,
        )
        self.runtime = self._agent.runtime
        self._engine = engine
        self._hook_engine = hook_engine
        self._subagent_catalog = subagent_catalog
        self._task_manager = task_manager
        self._approval_broker = approval_broker
        self._worktree_manager = worktree_manager
        self._team_manager = team_manager
        self._coordinator_mode = coordinator_mode
        self._working_directory = selected_directory
        self._version = version
        self._stream = stream
        self._current_writer = writer
        self._memory_manager = memory_manager
        self._instruction_text = instruction_text
        self._base_prompt = combined_base_prompt
        self._sessions_dir = sessions_dir
        self._cleanup_task = cleanup_task
        self._skill_loader = skill_loader
        self._skill_executor = SkillExecutor(
            self._agent,
            provider_configs,
            stream=stream,
        )
        selected_load_tool = load_skill_tool
        if selected_load_tool is None:
            registered = self._tool_registry.get("load_skill")
            if isinstance(registered, LoadSkillTool):
                selected_load_tool = registered
        if selected_load_tool is not None:
            selected_load_tool.set_agent(self._agent)
        if install_skill_tool is not None:
            install_skill_tool.set_refresh_callback(self._sync_skills)
        if command_registry is None:
            command_registry = CommandRegistry()
        register_builtins(command_registry)
        self._command_registry = command_registry
        self._reserved_command_names = frozenset(
            name
            for command in command_registry.visible()
            if command.source == "builtin"
            for name in (command.name, *command.aliases)
        )
        self._sync_skills(skill_loader.list() if skill_loader is not None else ())

    def _sync_skills(self, skills: tuple[SkillDef, ...]) -> None:
        """Atomically refresh dynamic commands, then publish the matching catalog."""
        commands = build_skill_commands(skills, self._reserved_command_names)
        self._command_registry.replace_source("skill", commands)
        self._agent.set_skill_catalog(render_skill_catalog(skills))
        self._agent.refresh_system_prompt(self._conversation)

    async def reload_skills(self) -> tuple[SkillDef, ...]:
        """Reload disk state and synchronize commands and Agent catalog."""
        if self._skill_loader is None:
            self._sync_skills(())
            return ()
        skills = await asyncio.to_thread(self._skill_loader.reload)
        self._sync_skills(skills)
        return skills

    def set_current_writer(self, writer: Writer) -> Writer | None:
        """Transfer application ownership to a successfully resumed writer."""
        if not isinstance(writer, Writer):
            raise TypeError("writer must be a Writer")
        previous = self._current_writer
        self._current_writer = writer
        return previous

    async def on_mount(self) -> None:
        """Install the chat screen after the application starts."""
        await self.push_screen(
            ChatScreen(
                self._agent,
                self._conversation,
                model_name=self._provider.model_name,
                working_directory=self._working_directory,
                version=self._version,
                initial_mode=self._engine.default_mode,
                stream=self._stream,
                writer=self._current_writer,
                memory_manager=self._memory_manager,
                instruction_text=self._instruction_text,
                base_prompt=self._base_prompt,
                sessions_dir=self._sessions_dir,
                command_registry=self._command_registry,
                skill_loader=self._skill_loader,
                skill_executor=self._skill_executor,
                skill_reloader=self.reload_skills,
                task_manager=self._task_manager,
                approval_broker=self._approval_broker,
                worktree_manager=self._worktree_manager,
                team_manager=self._team_manager,
                coordinator_mode=self._coordinator_mode,
            )
        )
        await self.dispatch_session_start(self._engine.default_mode)

    @property
    def main_agent(self) -> Agent:
        """Expose the constructed main Agent for late-bound AgentTool wiring."""
        return self._agent

    async def stop_subagent_consumers(self) -> None:
        """Stop ChatScreen queue consumers when the screen is still mounted."""
        try:
            current = self.screen
        except Exception:
            return
        if isinstance(current, ChatScreen):
            await current.stop_subagent_consumers()

    def _base_hook_payload(self, event: HookEvent, mode: Mode) -> Payload:
        return {
            "event": event.value,
            "session_id": self.runtime.session.session_id,
            "cwd": str(self._working_directory),
            "mode": str(mode),
        }

    async def dispatch_hook(
        self,
        event: HookEvent,
        payload: Payload,
        mode: Mode,
    ) -> DispatchResult:
        """Dispatch one TUI-owned event without consuming injected prompts."""
        if self._hook_engine is None:
            return DispatchResult()
        complete = self._base_hook_payload(event, mode)
        complete.update(payload)
        return await self._hook_engine.dispatch(event, complete, self.runtime)

    async def dispatch_session_start(self, mode: Mode) -> None:
        result = await self.dispatch_hook(HookEvent.SESSION_START, {}, mode)
        self.runtime.append_reminders(result.injected_prompts)

    async def dispatch_session_resume(self, mode: Mode) -> None:
        result = await self.dispatch_hook(HookEvent.SESSION_RESUME, {}, mode)
        self.runtime.append_reminders(result.injected_prompts)

    async def dispatch_session_end(self, mode: Mode | None = None) -> None:
        if not self.runtime.claim_session_end():
            return
        selected_mode = mode
        if selected_mode is None:
            try:
                current_screen = self.screen
            except Exception:
                current_screen = None
            selected_mode = (
                current_screen.mode
                if isinstance(current_screen, ChatScreen)
                else self._engine.default_mode
            )
        result = await self.dispatch_hook(HookEvent.SESSION_END, {}, selected_mode)
        self.runtime.append_reminders(result.injected_prompts)

    def hook_sources(self) -> list[str]:
        return self._hook_engine.sources if self._hook_engine is not None else []

    def hook_rules(self) -> list[HookRule]:
        return self._hook_engine.rules if self._hook_engine is not None else []

    async def on_unmount(self) -> None:
        """Close optional Provider resources when Textual shuts down."""
        try:
            await self.dispatch_session_end()
        except Exception as error:
            logger.warning("SessionEnd hook failed error=%s", type(error).__name__)
        try:
            await self.stop_subagent_consumers()
        except Exception as error:
            logger.warning("Subagent consumers shutdown failed error=%s", type(error).__name__)
        if self._task_manager is not None:
            try:
                await self._task_manager.aclose()
            except Exception as error:
                logger.warning("Task manager shutdown failed error=%s", type(error).__name__)
        if self._approval_broker is not None:
            try:
                await self._approval_broker.aclose()
            except Exception as error:
                logger.warning("Approval broker shutdown failed error=%s", type(error).__name__)
        if self._hook_engine is not None:
            try:
                await self._hook_engine.aclose()
            except Exception as error:
                logger.warning("Hook engine shutdown failed error=%s", type(error).__name__)
        await self._agent.shutdown_memory_updates()
        if isinstance(self._provider, _AsyncCloseable):
            try:
                await self._provider.close()
            except Exception as error:
                logger.error(
                    "Provider close failed provider=%s error=%s",
                    self._provider.provider_name,
                    type(error).__name__,
                )
        if self._current_writer is not None:
            try:
                self._current_writer.close()
            except Exception as error:
                logger.warning("Session writer close failed error=%s", type(error).__name__)
        cleanup_task = self._cleanup_task
        if cleanup_task is not None and not cleanup_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=1.0)
            except TimeoutError:
                cleanup_task.cancel()
                await asyncio.gather(cleanup_task, return_exceptions=True)
            except Exception as error:
                logger.warning("Session cleanup failed error=%s", type(error).__name__)
