"""Transactional Team teammate spawning with Worktree compensation."""

from __future__ import annotations

from collections.abc import Callable

from codewright.agent.team_hook import TeamHook, TeamSpawnRequest
from codewright.team.backend import Backend
from codewright.team.manager import Manager, TeamError
from codewright.team.types import RuntimeHandle, SpawnRequest, TeammateInfo
from codewright.tool import Result
from codewright.worktree import ExitOptions

type BackendFactory = Callable[[object], Backend]


class Spawner(TeamHook):
    def __init__(self, manager: Manager, backend_factory: BackendFactory) -> None:
        self._manager = manager
        self._backend_factory = backend_factory

    async def spawn_teammate(self, request: TeamSpawnRequest) -> Result:
        team = self._manager.get(request.team_name)
        if team is None:
            return _error("unknown_team", f"Unknown team: {request.team_name}")
        seed = TeammateInfo(
            name=request.member_name,
            agent_type=request.subagent_type or "general-purpose",
            model=request.model or "inherit",
            backend_type=team.backend,
            plan_mode_required=request.plan_mode_required,
        )
        try:
            await self._manager.reserve_member(team.slug, seed)
        except TeamError as error:
            return _error(error.code, error.safe_message)

        worktree = None
        runtime = None
        result = None
        cleanup_errors: list[str] = []
        backend = self._backend_factory(team)
        try:
            worktree = await self._manager.worktree_manager.create(
                f"team-{team.slug}/{request.member_name}",
                base_ref="HEAD",
                manual=True,
            )
            runtime = self._manager.runtime_factory.create(
                initial_prompt=request.prompt,
                description=request.description,
                model=request.model or "inherit",
                request=request,
                team=team,
                worktree=worktree,
            )
            result = await backend.spawn(
                SpawnRequest(team.slug, request.member_name, request.prompt, runtime)
            )
            member = await self._manager.commit_member_start(
                team.slug,
                request.member_name,
                result,
                worktree_name=worktree.name,
                worktree_path=worktree.path,
                branch=worktree.branch,
                session_dir=str(runtime.writer.path.parent),
            )
            return Result(
                f'{{"agent_id":"{member.agent_id}","status":"running"}}'
            )
        except Exception as error:
            if result is not None:
                try:
                    await backend.kill(
                        RuntimeHandle(
                            team.slug,
                            request.member_name,
                            result.agent_id,
                            result.pane_id,
                            result.runtime_task_id,
                        )
                    )
                except Exception as cleanup_error:
                    cleanup_errors.append(f"runtime: {type(cleanup_error).__name__}")
            if runtime is not None and result is None:
                try:
                    await runtime.aclose()
                except Exception as cleanup_error:
                    cleanup_errors.append(f"session: {type(cleanup_error).__name__}")
            if worktree is not None:
                try:
                    await self._manager.worktree_manager.remove(
                        worktree.name, ExitOptions(discard_changes=True)
                    )
                except Exception as cleanup_error:
                    cleanup_errors.append(f"worktree: {type(cleanup_error).__name__}")
            await self._manager.fail_member_start(
                team.slug,
                request.member_name,
                type(error).__name__,
                tuple(cleanup_errors),
            )
            return _error("team_spawn_failed", "The Team teammate could not start.")


def _error(code: str, message: str) -> Result:
    return Result(message, is_error=True, error_code=code)


__all__ = ["Spawner"]
