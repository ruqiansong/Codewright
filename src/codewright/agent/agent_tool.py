"""Model-facing entry point for foreground, background, and forked subagents."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING

from codewright.agent.agent_worktree import append_cleanup_report, isolated_prompt
from codewright.agent.completion import MaxTurnsReached
from codewright.agent.context import current_execution_context
from codewright.agent.factory import (
    ProviderFactory,
    SubagentFactoryError,
    build_defined_agent_and_conversation,
    build_fork_agent_and_conversation,
)
from codewright.agent.fork import is_fork_context
from codewright.agent.team_hook import TeamHook, TeamSpawnRequest
from codewright.config import ProviderConfig
from codewright.conversation import Conversation
from codewright.llm import LLMError, Provider
from codewright.llm.factory import create_provider
from codewright.permission import Outcome
from codewright.subagent import Catalog, Definition
from codewright.task import Manager, SubagentApprovalBroker
from codewright.tool import Result, cwd_from_ctx, with_cwd
from codewright.worktree import AutoCleanupReport, random_agent_name
from codewright.worktree import Manager as WorktreeManager

if TYPE_CHECKING:
    from codewright.agent import Agent, ApprovalRequest, ApprovalUpgrader, Event

AUTO_BACKGROUND_SECONDS = 120.0
PROVIDER_CLOSE_TIMEOUT = 5.0


class AgentTool:
    """Launch one isolated subagent using the active Agent execution context."""

    name = "Agent"
    description = "Delegate a focused task to an isolated subagent."
    read_only = False
    execution_timeout = None
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Complete task for the subagent.",
            },
            "description": {
                "type": "string",
                "description": "Short task description for status displays.",
            },
            "subagent_type": {
                "type": "string",
                "description": "Configured role name; empty or omitted requests a Fork.",
            },
            "model": {
                "type": "string",
                "description": "inherit or a configured Provider name.",
            },
            "run_in_background": {
                "type": "boolean",
                "description": "Launch asynchronously when true.",
            },
            "name": {
                "type": "string",
                "description": "Optional stable name used by SendMessage.",
            },
            "team_name": {
                "type": "string",
                "description": "Team receiving this long-lived teammate.",
            },
            "plan_mode_required": {
                "type": "boolean",
                "description": "Require Lead approval before implementation.",
            },
        },
        "required": ["prompt", "description"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        catalog: Catalog,
        task_manager: Manager,
        approval_broker: SubagentApprovalBroker,
        provider_configs: Sequence[ProviderConfig] = (),
        *,
        provider_factory: ProviderFactory = create_provider,
        enable_subagent_background: bool = True,
        foreground_timeout: float = AUTO_BACKGROUND_SECONDS,
        worktree_manager: WorktreeManager | None = None,
        team_hook: TeamHook | None = None,
    ) -> None:
        if not isinstance(catalog, Catalog):
            raise TypeError("catalog must be a Catalog")
        if not isinstance(task_manager, Manager):
            raise TypeError("task_manager must be a Manager")
        if not isinstance(approval_broker, SubagentApprovalBroker):
            raise TypeError("approval_broker must be a SubagentApprovalBroker")
        configs = tuple(provider_configs)
        if not all(isinstance(config, ProviderConfig) for config in configs):
            raise TypeError("provider_configs must contain only ProviderConfig values")
        if len({config.name for config in configs}) != len(configs):
            raise ValueError("provider config names must be unique")
        if not callable(provider_factory):
            raise TypeError("provider_factory must be callable")
        if not isinstance(enable_subagent_background, bool):
            raise TypeError("enable_subagent_background must be a boolean")
        if (
            not isinstance(foreground_timeout, (int, float))
            or isinstance(foreground_timeout, bool)
            or foreground_timeout <= 0
        ):
            raise ValueError("foreground_timeout must be a positive number")
        self._catalog = catalog
        self._task_manager = task_manager
        self._approval_broker = approval_broker
        self._provider_configs = configs
        self._provider_factory = provider_factory
        self._enable_background = enable_subagent_background
        self._foreground_timeout = float(foreground_timeout)
        self._worktree_manager = worktree_manager
        if team_hook is not None and not isinstance(team_hook, TeamHook):
            raise TypeError("team_hook must satisfy TeamHook")
        self._team_hook = team_hook
        self._parent: Agent | None = None

    def set_parent(self, parent: Agent) -> None:
        """Inject the main Agent after the Registry and Agent are both constructed."""
        from codewright.agent import Agent

        if not isinstance(parent, Agent):
            raise TypeError("parent must be an Agent")
        self._parent = parent

    async def execute(self, arguments_json: str) -> Result:
        """Validate, construct, and run or launch one subagent invocation."""
        arguments = _parse_arguments(arguments_json)
        if isinstance(arguments, Result):
            return arguments
        context = current_execution_context()
        if context is None or self._parent is None:
            return _error("not_initialized", "Agent delegation is not initialized.")
        parent = context.agent
        if parent is not self._parent:
            return _error("not_initialized", "Agent delegation is not initialized.")
        if parent.subagent_kind != "main" or is_fork_context(context.conversation):
            return _error("nested_subagent", "Nested subagent delegation is not allowed.")

        prompt = arguments["prompt"]
        description = arguments["description"]
        subagent_type = arguments.get("subagent_type", "")
        model = arguments.get("model")
        requested_background = arguments.get("run_in_background", False)
        task_name = arguments.get("name", "")
        team_name = arguments.get("team_name", "")
        plan_mode_required = arguments.get("plan_mode_required", False)
        if not isinstance(prompt, str) or not isinstance(description, str):
            raise RuntimeError("validated arguments became inconsistent")
        if not isinstance(subagent_type, str) or not isinstance(task_name, str):
            raise RuntimeError("validated arguments became inconsistent")
        if not isinstance(team_name, str) or not isinstance(plan_mode_required, bool):
            raise RuntimeError("validated arguments became inconsistent")
        if model is not None and not isinstance(model, str):
            raise RuntimeError("validated arguments became inconsistent")
        if not isinstance(requested_background, bool):
            raise RuntimeError("validated arguments became inconsistent")

        if plan_mode_required and not team_name:
            return _error(
                "invalid_arguments", "plan_mode_required is only valid with team_name."
            )
        if team_name:
            if self._team_hook is None:
                return _error("team_unavailable", "Agent Team delegation is not available.")
            try:
                return await self._team_hook.spawn_teammate(
                    TeamSpawnRequest(
                        team_name=team_name,
                        member_name=task_name or subagent_type or "teammate",
                        prompt=prompt,
                        description=description,
                        subagent_type=subagent_type,
                        model=model,
                        plan_mode_required=plan_mode_required,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return _error("team_spawn_failed", "The Team teammate could not start.")

        is_fork = not subagent_type
        definition = (
            self._catalog.fork_definition() if is_fork else self._catalog.resolve(subagent_type)
        )
        if definition is None:
            return _error("unknown_subagent", f"Unknown subagent type: {subagent_type}")
        isolated = not is_fork and definition.isolation == "worktree"
        background = False if isolated else is_fork or requested_background or definition.background
        if background and not self._enable_background:
            return _error("background_disabled", "Background subagents are disabled.")

        def approval_upgrader(request: ApprovalRequest) -> Awaitable[Outcome]:
            return self._approval_broker.request(
                definition.name if not is_fork else "fork", request
            )

        if isolated:
            return await self._execute_isolated(
                parent,
                context.conversation,
                definition,
                prompt,
                description,
                model,
                task_name,
                approval_upgrader,
            )

        try:
            child, conversation, owned_provider = self._build_child(
                parent,
                context.conversation,
                definition,
                prompt,
                model,
                background,
                approval_upgrader,
                is_fork,
            )
        except SubagentFactoryError as error:
            return _error(error.error_code, error.safe_message)
        except (TypeError, ValueError):
            return _error("subagent_creation_failed", "The subagent could not be created.")

        if background:
            try:
                task = await self._task_manager.launch(
                    child,
                    conversation,
                    prompt,
                    description,
                    name=task_name,
                    owned_provider=owned_provider,
                )
            except asyncio.CancelledError:
                await _close_owned_provider(owned_provider)
                raise
            except Exception:
                await _close_owned_provider(owned_provider)
                return _error("background_launch_failed", "The background task could not start.")
            return _json_result({"task_id": task.id, "status": "async_launched"})

        return await self._run_foreground(
            child,
            conversation,
            prompt,
            description,
            task_name,
            owned_provider,
        )

    async def _execute_isolated(
        self,
        parent: Agent,
        parent_conversation: Conversation,
        definition: Definition,
        prompt: str,
        description: str,
        model: str | None,
        task_name: str,
        approval_upgrader: ApprovalUpgrader,
    ) -> Result:
        manager = self._worktree_manager
        if manager is None:
            return _error("worktree_unavailable", "Worktree isolation is not available.")
        name = random_agent_name()
        try:
            worktree = await manager.create(name, manual=False)
        except Exception:
            return _error("worktree_creation_failed", "The isolated Worktree could not be created.")
        parent_cwd = cwd_from_ctx() or manager.repo_root
        effective_prompt = isolated_prompt(prompt, parent_cwd, worktree)
        result: Result | None = None
        try:
            try:
                child, conversation, owned_provider = self._build_child(
                    parent,
                    parent_conversation,
                    definition,
                    effective_prompt,
                    model,
                    False,
                    approval_upgrader,
                    False,
                )
            except SubagentFactoryError as error:
                result = _error(error.error_code, error.safe_message)
            except (TypeError, ValueError):
                result = _error("subagent_creation_failed", "The subagent could not be created.")
            if result is None:
                result = await self._run_foreground(
                    child,
                    conversation,
                    effective_prompt,
                    description,
                    task_name,
                    owned_provider,
                    execution_cwd=worktree.path,
                    allow_adoption=False,
                )
        finally:
            try:
                cleanup = await manager.auto_cleanup(name)
            except Exception:
                cleanup = AutoCleanupReport(
                    True,
                    worktree.path,
                    worktree.branch,
                    "自动清理异常，已安全保留",
                )
        if result is None:
            result = _error("subagent_failed", "The subagent failed unexpectedly.")
        return append_cleanup_report(result, cleanup)

    def _build_child(
        self,
        parent: Agent,
        parent_conversation: Conversation,
        definition: Definition,
        prompt: str,
        model: str | None,
        background: bool,
        approval_upgrader: ApprovalUpgrader,
        is_fork: bool,
    ) -> tuple[Agent, Conversation, Provider | None]:
        if is_fork:
            return build_fork_agent_and_conversation(
                parent,
                definition,
                parent_conversation,
                prompt,
                self._provider_configs,
                model=model,
                approval_upgrader=approval_upgrader,
                provider_factory=self._provider_factory,
            )
        return build_defined_agent_and_conversation(
            parent,
            definition,
            prompt,
            self._provider_configs,
            model=model,
            background=background,
            approval_upgrader=approval_upgrader,
            provider_factory=self._provider_factory,
        )

    async def _run_foreground(
        self,
        child: Agent,
        conversation: Conversation,
        prompt: str,
        description: str,
        task_name: str,
        owned_provider: Provider | None,
        *,
        execution_cwd: str | None = None,
        allow_adoption: bool = True,
    ) -> Result:
        sink: asyncio.Queue[Event | None] = asyncio.Queue()
        if execution_cwd is None:
            handle = asyncio.create_task(child.run_to_completion(conversation, "", event_sink=sink))
        else:
            with with_cwd(execution_cwd):
                handle = asyncio.create_task(
                    child.run_to_completion(conversation, "", event_sink=sink)
                )
        adopted = False
        try:
            if allow_adoption:
                done, _ = await asyncio.wait({handle}, timeout=self._foreground_timeout)
            else:
                done, _ = await asyncio.wait({handle})
            if handle in done:
                try:
                    completion = handle.result()
                except MaxTurnsReached as error:
                    return _error(
                        "max_turns_reached",
                        error.last_text or "The subagent reached its maximum turn limit.",
                    )
                except LLMError as error:
                    return _error("subagent_failed", error.safe_message)
                except Exception:
                    return _error("subagent_failed", "The subagent failed unexpectedly.")
                return Result(completion.text)

            adoption = asyncio.create_task(
                self._task_manager.adopt_running(
                    child,
                    conversation,
                    prompt,
                    description,
                    handle,
                    name=task_name,
                    owned_provider=owned_provider,
                    event_sink=sink,
                )
            )
            try:
                task = await asyncio.shield(adoption)
            except asyncio.CancelledError:
                try:
                    await adoption
                    adopted = True
                except Exception:
                    pass
                raise
            adopted = True
            return _json_result({"task_id": task.id, "status": "timed_out_to_background"})
        except asyncio.CancelledError:
            if not adopted:
                handle.cancel()
                await asyncio.gather(handle, return_exceptions=True)
            raise
        except Exception:
            if not adopted:
                handle.cancel()
                await asyncio.gather(handle, return_exceptions=True)
            return _error("background_launch_failed", "The background task could not start.")
        finally:
            if not adopted:
                await _close_owned_provider(owned_provider)


def _parse_arguments(arguments_json: str) -> dict[str, object] | Result:
    if not isinstance(arguments_json, str):
        return _error("invalid_arguments", "Arguments must be a JSON string.")
    try:
        parsed = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return _error("invalid_arguments", "Arguments must be valid JSON.")
    if not isinstance(parsed, dict):
        return _error("invalid_arguments", "Arguments must be a JSON object.")
    allowed = {
        "prompt",
        "description",
        "subagent_type",
        "model",
        "run_in_background",
        "name",
        "team_name",
        "plan_mode_required",
    }
    if set(parsed) - allowed or not {"prompt", "description"}.issubset(parsed):
        return _error(
            "invalid_arguments",
            "prompt and description are required; unknown arguments are not allowed.",
        )
    for key in ("prompt", "description"):
        value = parsed[key]
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            return _error("invalid_arguments", f"{key} must be a non-empty trimmed string.")
    for key in ("subagent_type", "model", "name", "team_name"):
        if key not in parsed:
            continue
        value = parsed[key]
        if not isinstance(value, str) or value != value.strip():
            return _error("invalid_arguments", f"{key} must be a trimmed string.")
        if key == "model" and not value:
            return _error("invalid_arguments", "model must not be empty.")
    if "run_in_background" in parsed and not isinstance(parsed["run_in_background"], bool):
        return _error("invalid_arguments", "run_in_background must be a boolean.")
    if "plan_mode_required" in parsed and not isinstance(parsed["plan_mode_required"], bool):
        return _error("invalid_arguments", "plan_mode_required must be a boolean.")
    return parsed


async def _close_owned_provider(provider: Provider | None) -> None:
    if provider is None:
        return
    close = getattr(provider, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await asyncio.wait_for(result, timeout=PROVIDER_CLOSE_TIMEOUT)
    except asyncio.CancelledError:
        raise
    except Exception:
        pass


def _json_result(payload: object) -> Result:
    return Result(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _error(code: str, message: str) -> Result:
    return Result(message, is_error=True, error_code=code)


__all__ = ["AUTO_BACKGROUND_SECONDS", "AgentTool"]
