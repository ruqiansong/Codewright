"""Project-scoped Agent Team lifecycle and member state management."""

from __future__ import annotations

import asyncio
import logging
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from codewright.team.mailbox import Mailbox, Message, MessageType
from codewright.team.persistence import (
    FileLock,
    LockTimeoutError,
    atomic_write_json,
    contained_team_dir,
    read_json,
    sanitize_team_name,
)
from codewright.team.types import (
    BackendType,
    DeleteReport,
    MemberState,
    SpawnResult,
    Team,
    TeammateInfo,
    utc_now,
)

logger = logging.getLogger(__name__)
_ACTIVE_STATES = {MemberState.STARTING, MemberState.RUNNING, MemberState.STOPPING}


class TeamError(RuntimeError):
    """Stable, user-displayable Team lifecycle failure."""

    def __init__(self, code: str, safe_message: str) -> None:
        self.code = code
        self.safe_message = safe_message
        super().__init__(safe_message)


class Manager:
    def __init__(
        self,
        *,
        home_dir: Path,
        project_root: Path,
        worktree_manager: object,
        task_manager: object,
        runtime_factory: object,
        backend_detector: Callable[[], BackendType] | None = None,
    ) -> None:
        self.home_dir = Path(home_dir).resolve()
        self.project_root = Path(project_root).resolve()
        self.teams_root = self.home_dir / ".codewright" / "teams"
        self.teams_root.mkdir(parents=True, exist_ok=True)
        self.worktree_manager = worktree_manager
        self.task_manager = task_manager
        self.runtime_factory = runtime_factory
        self._backend_detector = backend_detector or (lambda: BackendType.IN_PROCESS)
        self._teams: dict[str, Team] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._active_slug: str | None = None
        self._load_existing()
        add_listener = getattr(self.task_manager, "add_completion_listener", None)
        if callable(add_listener):
            add_listener(self._on_task_completion)

    @property
    def active_team(self) -> Team | None:
        return self._teams.get(self._active_slug) if self._active_slug is not None else None

    def get(self, name_or_slug: str) -> Team | None:
        direct = self._teams.get(name_or_slug)
        if direct is not None:
            return direct
        folded = name_or_slug.casefold()
        matches = [
            team
            for team in self._teams.values()
            if team.slug.casefold() == folded or team.name.casefold() == folded
        ]
        return matches[0] if len(matches) == 1 else None

    def list(self) -> tuple[Team, ...]:
        return tuple(sorted(self._teams.values(), key=lambda team: team.created_at))

    def use(self, name_or_slug: str) -> Team:
        team = self.get(name_or_slug)
        if team is None:
            raise TeamError("unknown_team", f"Unknown team: {name_or_slug}")
        self._active_slug = team.slug
        return team

    async def create(
        self,
        name: str,
        description: str = "",
        *,
        lead_permission_mode: str = "default",
    ) -> Team:
        base = sanitize_team_name(name)
        try:
            async with FileLock(self.teams_root / ".create.lock"):
                slug = self._allocate_slug(base)
                team_dir = contained_team_dir(self.teams_root, slug)
                team_dir.mkdir(mode=0o700)
                (team_dir / "mailbox").mkdir(mode=0o700)
                now = utc_now()
                team = Team(
                    name=name.strip(),
                    slug=slug,
                    description=description,
                    project_root=str(self.project_root),
                    lead_permission_mode=lead_permission_mode,
                    backend=self._backend_detector(),
                    created_at=now,
                    updated_at=now,
                    config_dir=str(team_dir),
                )
                atomic_write_json(team.tasks_path, [])
                atomic_write_json(team.config_path, team.to_dict())
        except LockTimeoutError as error:
            raise TeamError("team_lock_timeout", "Timed out creating the team") from error
        self._teams[team.slug] = team
        self._locks.setdefault(team.slug, asyncio.Lock())
        self._active_slug = team.slug
        return team

    async def reserve_member(
        self,
        team_slug: str,
        seed: TeammateInfo | None = None,
        **fields: object,
    ) -> TeammateInfo:
        if seed is None:
            seed = TeammateInfo(**fields)  # type: ignore[arg-type]
        if seed.state is not MemberState.STARTING:
            seed = replace(seed, state=MemberState.STARTING, updated_at=utc_now())

        def mutate(team: Team) -> TeammateInfo:
            if any(member.name.casefold() == seed.name.casefold() for member in team.members):
                raise TeamError("duplicate_member", f"Team member already exists: {seed.name}")
            team.members.append(seed)
            return seed

        return await self._mutate(team_slug, mutate)

    async def commit_member_start(
        self,
        team_slug: str,
        member_name: str,
        result: SpawnResult,
        *,
        worktree_name: str | None = None,
        worktree_path: str | None = None,
        branch: str | None = None,
        session_dir: str | None = None,
    ) -> TeammateInfo:
        def mutate(team: Team) -> TeammateInfo:
            index, member = self._member_index(team, member_name)
            if member.state is not MemberState.STARTING:
                raise TeamError("invalid_member_state", "Team member is not starting")
            updated = replace(
                member,
                agent_id=result.agent_id,
                pane_id=result.pane_id,
                runtime_task_id=result.runtime_task_id,
                worktree_name=member.worktree_name if worktree_name is None else worktree_name,
                worktree_path=member.worktree_path if worktree_path is None else worktree_path,
                branch=member.branch if branch is None else branch,
                session_dir=member.session_dir if session_dir is None else session_dir,
                state=MemberState.RUNNING,
                last_error="",
                updated_at=utc_now(),
            )
            team.members[index] = updated
            return updated

        return await self._mutate(team_slug, mutate)

    async def fail_member_start(
        self,
        team_slug: str,
        member_name: str,
        error: str,
        resources: tuple[str, ...] = (),
    ) -> TeammateInfo:
        def mutate(team: Team) -> TeammateInfo:
            index, member = self._member_index(team, member_name)
            updated = replace(
                member,
                state=MemberState.FAILED,
                last_error=error.strip(),
                updated_at=utc_now(),
            )
            team.members[index] = updated
            team.cleanup_errors.extend(item for item in resources if item)
            return updated

        return await self._mutate(team_slug, mutate)

    async def update_member_state(
        self,
        team_slug: str,
        member_name: str,
        state: MemberState,
        *,
        last_error: str | None = None,
    ) -> TeammateInfo:
        if not isinstance(state, MemberState):
            raise TypeError("state must be a MemberState")

        def mutate(team: Team) -> TeammateInfo:
            index, member = self._member_index(team, member_name)
            updated = replace(
                member,
                state=state,
                last_error=member.last_error if last_error is None else last_error,
                updated_at=utc_now(),
            )
            team.members[index] = updated
            return updated

        return await self._mutate(team_slug, mutate)

    async def set_pending_plan_request(
        self,
        team_slug: str,
        member_name: str,
        request_id: str,
    ) -> TeammateInfo:
        def mutate(team: Team) -> TeammateInfo:
            index, member = self._member_index(team, member_name)
            if not member.plan_mode_required:
                raise TeamError("plan_not_required", "This member does not require plan approval")
            if member.pending_plan_request_id:
                raise TeamError("plan_request_pending", "A plan request is already pending")
            updated = replace(
                member,
                pending_plan_request_id=request_id,
                updated_at=utc_now(),
            )
            team.members[index] = updated
            return updated

        return await self._mutate(team_slug, mutate)

    async def consume_plan_response(
        self,
        team_slug: str,
        member_name: str,
        request_id: str,
    ) -> TeammateInfo:
        def mutate(team: Team) -> TeammateInfo:
            index, member = self._member_index(team, member_name)
            if member.pending_plan_request_id != request_id:
                raise TeamError("invalid_plan_request", "Plan requestId is stale or does not match")
            updated = replace(
                member,
                pending_plan_request_id="",
                updated_at=utc_now(),
            )
            team.members[index] = updated
            return updated

        return await self._mutate(team_slug, mutate)

    def resolve_member(self, team_slug: str, name_or_id: str) -> TeammateInfo | None:
        team = self._teams.get(team_slug)
        if team is None:
            return None
        folded = name_or_id.casefold()
        matches = [
            member
            for member in team.members
            if member.name.casefold() == folded
            or (member.agent_id and member.agent_id.casefold() == folded)
        ]
        return matches[0] if len(matches) == 1 else None

    async def dispatch_member(
        self,
        team_slug: str,
        member_name: str,
        message: str,
        *,
        approved: bool = False,
    ) -> TeammateInfo:
        """Continue a retained task or recover its persisted session after restart."""
        team = self.get(team_slug)
        member = self.resolve_member(team_slug, member_name)
        if team is None or member is None:
            raise TeamError("unknown_member", f"Unknown team member: {member_name}")
        if member.backend_type is not BackendType.IN_PROCESS:
            raise TeamError("teammate_unavailable", "Pane teammate wake is unavailable")
        task = (
            await self.task_manager.get(member.runtime_task_id)
            if member.runtime_task_id
            else None
        )
        if task is not None and getattr(task.status, "value", "") == "completed":
            if approved:
                from codewright.permission import parse_mode

                mode, _ = parse_mode(team.lead_permission_mode)
                task.sub_agent.permission_mode = mode
            continued = await self.task_manager.send_message(
                f"team:{team.slug}:{member.name}", message
            )
            return await self._set_member_runtime(
                team.slug, member.name, continued.id, MemberState.RUNNING
            )
        if task is not None and getattr(task.status, "value", "") == "running":
            raise TeamError("teammate_busy", "Team member is still running")
        try:
            runtime = self.runtime_factory.resume(
                member,
                system_prompt=f"You are Team member {member.name} in Team {team.name}.",
                description=f"Resume Team member {member.name}",
            )
            if approved:
                from codewright.permission import parse_mode

                mode, _ = parse_mode(team.lead_permission_mode)
                runtime.agent.permission_mode = mode
            runtime.conversation.add_user(message)
            launched = await self.task_manager.launch(
                runtime.agent,
                runtime.conversation,
                runtime.initial_prompt,
                runtime.description,
                name=f"team:{team.slug}:{member.name}",
                owned_provider=runtime.owned_provider,
            )
        except Exception as error:
            await self.update_member_state(
                team.slug,
                member.name,
                MemberState.FAILED,
                last_error=type(error).__name__,
            )
            raise TeamError("teammate_resume_failed", "Team member could not resume") from error
        return await self._set_member_runtime(
            team.slug, member.name, launched.id, MemberState.RUNNING
        )

    async def kill_member(self, team_slug: str, member_name: str) -> TeammateInfo:
        member = self.resolve_member(team_slug, member_name)
        if member is None:
            raise TeamError("unknown_member", f"Unknown team member: {member_name}")
        if member.runtime_task_id:
            task = await self.task_manager.get(member.runtime_task_id)
            if task is not None and getattr(task.status, "value", "") == "running":
                await self.task_manager.stop(member.runtime_task_id)
        return await self.update_member_state(team_slug, member.name, MemberState.STOPPED)

    async def _set_member_runtime(
        self,
        team_slug: str,
        member_name: str,
        task_id: str,
        state: MemberState,
    ) -> TeammateInfo:
        def mutate(team: Team) -> TeammateInfo:
            index, member = self._member_index(team, member_name)
            updated = replace(
                member,
                agent_id=task_id,
                runtime_task_id=task_id,
                state=state,
                last_error="",
                updated_at=utc_now(),
            )
            team.members[index] = updated
            return updated

        return await self._mutate(team_slug, mutate)

    async def delete(
        self,
        team_slug: str,
        *,
        force: bool = False,
        purge_sessions: bool = False,
    ) -> DeleteReport:
        team = self.get(team_slug)
        if team is None:
            raise TeamError("unknown_team", f"Unknown team: {team_slug}")
        if not force and any(member.state in _ACTIVE_STATES for member in team.members):
            raise TeamError("team_active", "Team has active members")
        errors: list[str] = []
        retained_sessions: list[str] = []
        for member in tuple(team.members):
            if force and member.runtime_task_id:
                try:
                    task = await self.task_manager.get(member.runtime_task_id)
                    if task is not None and task.status.value == "running":
                        await self.task_manager.stop(member.runtime_task_id)
                except Exception as error:
                    errors.append(f"{member.name}: runtime {type(error).__name__}")
                    continue
            if member.worktree_name:
                try:
                    from codewright.worktree import ExitOptions

                    await self.worktree_manager.remove(
                        member.worktree_name,
                        ExitOptions(discard_changes=force),
                    )
                except Exception as error:
                    errors.append(f"{member.name}: worktree {type(error).__name__}")
                    continue
            if member.session_dir and purge_sessions:
                try:
                    await asyncio.to_thread(shutil.rmtree, member.session_dir)
                except FileNotFoundError:
                    pass
                except Exception as error:
                    errors.append(f"{member.name}: session {type(error).__name__}")
                    continue
            elif member.session_dir:
                retained_sessions.append(member.session_dir)
            await self._remove_member(team.slug, member.name)
        if errors:
            await self._record_cleanup_errors(team.slug, errors)
            return DeleteReport(False, team.slug, tuple(errors), tuple(retained_sessions))
        team_dir = Path(team.config_dir)
        try:
            await asyncio.to_thread(shutil.rmtree, team_dir)
        except OSError as error:
            return DeleteReport(False, team.slug, (str(error),), tuple(retained_sessions))
        self._teams.pop(team.slug, None)
        self._locks.pop(team.slug, None)
        if self._active_slug == team.slug:
            self._active_slug = None
        return DeleteReport(True, team.slug, retained_sessions=tuple(retained_sessions))

    async def _remove_member(self, team_slug: str, member_name: str) -> TeammateInfo:
        def mutate(team: Team) -> TeammateInfo:
            index, member = self._member_index(team, member_name)
            team.members.pop(index)
            return member

        return await self._mutate(team_slug, mutate)

    async def _record_cleanup_errors(self, team_slug: str, errors: list[str]) -> TeammateInfo:
        def mutate(team: Team) -> TeammateInfo:
            team.cleanup_errors.extend(errors)
            if team.members:
                return team.members[0]
            return TeammateInfo(name="cleanup", state=MemberState.FAILED)

        return await self._mutate(team_slug, mutate)

    async def _on_task_completion(self, task: object) -> None:
        task_id = getattr(task, "id", "")
        status = getattr(getattr(task, "status", None), "value", "failed")
        for team in tuple(self._teams.values()):
            member = next(
                (item for item in team.members if item.runtime_task_id == task_id),
                None,
            )
            if member is None:
                continue
            state = MemberState.IDLE if status == "completed" else MemberState.FAILED
            await self.update_member_state(team.slug, member.name, state)
            if state is MemberState.IDLE:
                await Mailbox(Path(team.config_dir)).append(
                    team.lead_agent_id,
                    Message(
                        sender=member.name,
                        type=MessageType.IDLE_NOTIFICATION,
                        content=f"{member.name} is idle",
                    ),
                )
            return

    async def aclose(self) -> None:
        remove_listener = getattr(self.task_manager, "remove_completion_listener", None)
        if callable(remove_listener):
            remove_listener(self._on_task_completion)

    async def poll_lead_mailboxes(self) -> tuple[Message, ...]:
        team = self.active_team
        if team is None:
            return ()
        return await Mailbox(Path(team.config_dir)).consume_unread(team.lead_agent_id)

    async def _mutate(
        self,
        team_slug: str,
        operation: Callable[[Team], TeammateInfo],
    ) -> TeammateInfo:
        cached = self.get(team_slug)
        if cached is None:
            raise TeamError("unknown_team", f"Unknown team: {team_slug}")
        local_lock = self._locks.setdefault(cached.slug, asyncio.Lock())
        try:
            async with local_lock:
                async with FileLock(Path(cached.config_dir) / "config.lock"):
                    team = self._read_team(Path(cached.config_dir))
                    result = operation(team)
                    team.updated_at = utc_now()
                    atomic_write_json(team.config_path, team.to_dict())
                    self._teams[team.slug] = team
                    return result
        except LockTimeoutError as error:
            raise TeamError("team_lock_timeout", "Timed out updating the team") from error

    def _load_existing(self) -> None:
        for path in sorted(self.teams_root.glob("*/config.json")):
            try:
                team = self._read_team(path.parent)
                if Path(team.project_root) != self.project_root:
                    continue
                if contained_team_dir(self.teams_root, team.slug) != path.parent:
                    raise ValueError("team slug does not match its directory")
                self._teams[team.slug] = team
                self._locks[team.slug] = asyncio.Lock()
            except Exception as error:
                logger.warning(
                    "Skipping invalid Team config path=%s error=%s",
                    path,
                    type(error).__name__,
                )

    def _read_team(self, directory: Path) -> Team:
        return Team.from_dict(read_json(directory / "config.json"), config_dir=str(directory))

    def _allocate_slug(self, base: str) -> str:
        candidate = base
        suffix = 1
        existing = {path.name.casefold() for path in self.teams_root.iterdir() if path.is_dir()}
        while candidate.casefold() in existing:
            suffix += 1
            marker = f"-{suffix}"
            candidate = f"{base[: 48 - len(marker)].rstrip('-_')}{marker}"
        return candidate

    @staticmethod
    def _member_index(team: Team, name_or_id: str) -> tuple[int, TeammateInfo]:
        folded = name_or_id.casefold()
        matches = [
            (index, member)
            for index, member in enumerate(team.members)
            if member.name.casefold() == folded
            or (member.agent_id and member.agent_id.casefold() == folded)
        ]
        if len(matches) != 1:
            raise TeamError("unknown_member", f"Unknown team member: {name_or_id}")
        return matches[0]


__all__ = ["Manager", "TeamError"]
