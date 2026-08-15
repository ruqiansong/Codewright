"""Model-facing tools for Agent Team lifecycle and collaboration."""

from __future__ import annotations

import inspect
import json
import uuid
from collections.abc import Callable, Mapping

from codewright.agent.context import current_execution_context
from codewright.team.mailbox import Mailbox, Message, MessageType
from codewright.team.manager import Manager, TeamError
from codewright.team.tasks import Store, TaskStatus, TeamTask
from codewright.team.types import Team
from codewright.tool import Registry, Result

_EMPTY_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}
_TASK_ID_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {"task_id": {"type": "string"}},
    "required": ["task_id"],
    "additionalProperties": False,
}


class _TeamTool:
    execution_timeout = None

    def __init__(self, manager: Manager) -> None:
        if not isinstance(manager, Manager):
            raise TypeError("manager must be a Team Manager")
        self._manager = manager

    def _team(self) -> Team | Result:
        context = current_execution_context()
        if context is not None and context.team is not None:
            team = self._manager.get(context.team.team_slug)
        else:
            team = self._manager.active_team
        if team is None:
            return _error("no_active_team", "No Agent Team is active.")
        return team


class TeamCreateTool(_TeamTool):
    name = "TeamCreate"
    description = "Create an Agent Team and make it active for the Lead."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "team_name": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["team_name"],
        "additionalProperties": False,
    }
    read_only = False

    def __init__(
        self,
        manager: Manager,
        on_active_change: Callable[[Team | None], object] | None = None,
    ) -> None:
        super().__init__(manager)
        self._on_active_change = on_active_change

    async def execute(self, arguments_json: str) -> Result:
        parsed = _parse(arguments_json, required={"team_name"}, optional={"description"})
        if isinstance(parsed, Result):
            return parsed
        team_name = _required_text(parsed, "team_name")
        if isinstance(team_name, Result):
            return team_name
        description = _optional_text(parsed, "description")
        if isinstance(description, Result):
            return description
        try:
            team = await self._manager.create(team_name, description)
        except (TeamError, ValueError) as error:
            return _mapped_error(error)
        await _notify(self._on_active_change, team)
        return _json({"team": _team_summary(team), "status": "created"})


class TeamDeleteTool(_TeamTool):
    name = "TeamDelete"
    description = "Delete an inactive Agent Team with explicit cleanup options."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "team_name": {"type": "string"},
            "force": {"type": "boolean"},
            "purge_sessions": {"type": "boolean"},
        },
        "required": ["team_name"],
        "additionalProperties": False,
    }
    read_only = False

    def __init__(
        self,
        manager: Manager,
        on_active_change: Callable[[Team | None], object] | None = None,
    ) -> None:
        super().__init__(manager)
        self._on_active_change = on_active_change

    async def execute(self, arguments_json: str) -> Result:
        parsed = _parse(
            arguments_json,
            required={"team_name"},
            optional={"force", "purge_sessions"},
        )
        if isinstance(parsed, Result):
            return parsed
        team_name = _required_text(parsed, "team_name")
        if isinstance(team_name, Result):
            return team_name
        flags = _booleans(parsed, "force", "purge_sessions")
        if isinstance(flags, Result):
            return flags
        try:
            report = await self._manager.delete(
                team_name,
                force=flags["force"],
                purge_sessions=flags["purge_sessions"],
            )
        except TeamError as error:
            return _mapped_error(error)
        if report.deleted:
            await _notify(self._on_active_change, self._manager.active_team)
        return _json(
            {
                "deleted": report.deleted,
                "team_slug": report.slug,
                "errors": list(report.errors),
            },
            error=not report.deleted,
            error_code="team_delete_incomplete",
        )


class TeamTaskCreateTool(_TeamTool):
    name = "TeamTaskCreate"
    description = "Create one task in the active Agent Team's shared task graph."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "description": {"type": "string"},
            "assignee": {"type": "string"},
            "blocked_by": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title"],
        "additionalProperties": False,
    }
    read_only = False

    async def execute(self, arguments_json: str) -> Result:
        parsed = _parse(
            arguments_json,
            required={"title"},
            optional={"description", "assignee", "blocked_by"},
        )
        if isinstance(parsed, Result):
            return parsed
        team = self._team()
        if isinstance(team, Result):
            return team
        title = _required_text(parsed, "title")
        description = _optional_text(parsed, "description")
        assignee = _optional_text(parsed, "assignee")
        dependencies = _string_list(parsed, "blocked_by")
        for value in (title, description, assignee, dependencies):
            if isinstance(value, Result):
                return value
        try:
            task = await Store(_team_dir(team)).create(
                title,
                description,
                assignee=assignee,
                blocked_by=dependencies,
            )
        except (KeyError, ValueError) as error:
            return _error("invalid_team_task", str(error))
        return _json({"task": _task_dict(task)})


class TeamTaskGetTool(_TeamTool):
    name = "TeamTaskGet"
    description = "Get one task from the active Agent Team's shared task graph."
    parameters = _TASK_ID_SCHEMA
    read_only = True

    async def execute(self, arguments_json: str) -> Result:
        parsed = _parse(arguments_json, required={"task_id"})
        if isinstance(parsed, Result):
            return parsed
        task_id = _required_text(parsed, "task_id")
        if isinstance(task_id, Result):
            return task_id
        team = self._team()
        if isinstance(team, Result):
            return team
        store = Store(_team_dir(team))
        task = await store.get(task_id)
        if task is None:
            return _error("unknown_team_task", f"Unknown Team task: {task_id}")
        return _json({"task": _task_dict(task, ready=await store.is_ready(task.id))})


class TeamTaskListTool(_TeamTool):
    name = "TeamTaskList"
    description = "List tasks in the active Agent Team's shared task graph."
    parameters = _EMPTY_SCHEMA
    read_only = True

    async def execute(self, arguments_json: str) -> Result:
        parsed = _parse(arguments_json)
        if isinstance(parsed, Result):
            return parsed
        team = self._team()
        if isinstance(team, Result):
            return team
        store = Store(_team_dir(team))
        tasks = await store.list()
        result = [
            _task_dict(task, ready=await store.is_ready(task.id)) for task in tasks
        ]
        return _json({"tasks": result})


class TeamTaskUpdateTool(_TeamTool):
    name = "TeamTaskUpdate"
    description = "Update one task in the active Agent Team's shared task graph."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "status": {"type": "string", "enum": [item.value for item in TaskStatus]},
            "assignee": {"type": "string"},
            "blocked_by": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    }
    read_only = False

    async def execute(self, arguments_json: str) -> Result:
        optional = {"title", "description", "status", "assignee", "blocked_by"}
        parsed = _parse(arguments_json, required={"task_id"}, optional=optional)
        if isinstance(parsed, Result):
            return parsed
        task_id = _required_text(parsed, "task_id")
        if isinstance(task_id, Result):
            return task_id
        team = self._team()
        if isinstance(team, Result):
            return team
        values: dict[str, object] = {}
        for name in ("title", "description", "assignee"):
            if name in parsed:
                value = (
                    _required_text(parsed, name)
                    if name == "title"
                    else _optional_text(parsed, name)
                )
                if isinstance(value, Result):
                    return value
                values[name] = value
        if "status" in parsed:
            status = _required_text(parsed, "status")
            if isinstance(status, Result):
                return status
            try:
                values["status"] = TaskStatus(status)
            except ValueError:
                return _error("invalid_arguments", "status is invalid.")
        if "blocked_by" in parsed:
            dependencies = _string_list(parsed, "blocked_by")
            if isinstance(dependencies, Result):
                return dependencies
            values["blocked_by"] = dependencies
        try:
            task = await Store(_team_dir(team)).update(task_id, **values)  # type: ignore[arg-type]
        except KeyError:
            return _error("unknown_team_task", f"Unknown Team task: {task_id}")
        except ValueError as error:
            return _error("invalid_team_task", str(error))
        return _json({"task": _task_dict(task)})


class TeamSendMessageTool(_TeamTool):
    name = "TeamSendMessage"
    description = "Send a scoped message to one member or all peers in the active Agent Team."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "content": {"type": "string"},
            "type": {"type": "string", "enum": [item.value for item in MessageType]},
            "requestId": {"type": "string"},
            "approve": {"type": "boolean"},
        },
        "required": ["to", "content"],
        "additionalProperties": False,
    }
    read_only = False

    async def execute(self, arguments_json: str) -> Result:
        parsed = _parse(
            arguments_json,
            required={"to", "content"},
            optional={"type", "requestId", "approve"},
        )
        if isinstance(parsed, Result):
            return parsed
        recipient = _required_text(parsed, "to")
        content = _optional_text(parsed, "content")
        if isinstance(recipient, Result) or isinstance(content, Result):
            return recipient if isinstance(recipient, Result) else content
        team = self._team()
        if isinstance(team, Result):
            return team
        context = current_execution_context()
        sender = (
            context.team.member_name
            if context is not None and context.team is not None and not context.team.is_lead
            else "lead"
        )
        try:
            message_type = MessageType(parsed.get("type", MessageType.TEXT.value))
        except (TypeError, ValueError):
            return _error("invalid_arguments", "type is invalid.")
        request_id = parsed.get("requestId", "")
        if message_type is MessageType.PLAN_APPROVAL_REQUEST and not request_id:
            request_id = f"plan-{uuid.uuid4().hex}"
        if not isinstance(request_id, str):
            return _error("invalid_arguments", "requestId must be a string.")
        approve = parsed.get("approve")
        if approve is not None and not isinstance(approve, bool):
            return _error("invalid_arguments", "approve must be a boolean.")
        if message_type is MessageType.PLAN_APPROVAL_REQUEST:
            if sender == "lead":
                return _error("invalid_plan_sender", "Only a teammate can request plan approval.")
            try:
                await self._manager.set_pending_plan_request(team.slug, sender, request_id)
            except TeamError as error:
                return _mapped_error(error)
        if message_type is MessageType.PLAN_APPROVAL_RESPONSE:
            if sender != "lead" or recipient in {"lead", "*"}:
                return _error(
                    "invalid_plan_sender", "Only the Lead can answer one teammate's plan request."
                )
            target_member = self._manager.resolve_member(team.slug, recipient)
            if target_member is None:
                return _error("unknown_member", f"Unknown Team member: {recipient}")
            try:
                await self._manager.consume_plan_response(
                    team.slug, target_member.name, request_id
                )
            except TeamError as error:
                return _mapped_error(error)
        try:
            message = Message(
                sender=sender,
                type=message_type,
                content=content,
                request_id=request_id,
                approve=approve,
            )
            targets = _resolve_targets(team, recipient, sender)
        except ValueError as error:
            return _error("invalid_team_message", str(error))
        mailbox = Mailbox(_team_dir(team))
        for target in targets:
            await mailbox.append(target, message)
        dispatch_status = "delivered"
        if recipient not in {"lead", "*"}:
            target_member = self._manager.resolve_member(team.slug, recipient)
            if target_member is not None and target_member.state.value in {
                "idle",
                "failed",
                "stopped",
            }:
                try:
                    await self._manager.dispatch_member(
                        team.slug,
                        target_member.name,
                        content,
                        approved=(
                            message_type is MessageType.PLAN_APPROVAL_RESPONSE
                            and approve is True
                        ),
                    )
                    dispatch_status = "resumed"
                except TeamError as error:
                    return _error(error.code, error.safe_message)
        return _json(
            {
                "message_id": message.id,
                "requestId": message.request_id,
                "recipients": list(targets),
                "status": dispatch_status,
            }
        )


def register_team_tools(
    registry: Registry,
    manager: Manager,
    *,
    on_active_change: Callable[[Team | None], object] | None = None,
) -> tuple[str, ...]:
    tools = (
        TeamCreateTool(manager, on_active_change),
        TeamDeleteTool(manager, on_active_change),
        TeamTaskCreateTool(manager),
        TeamTaskGetTool(manager),
        TeamTaskListTool(manager),
        TeamTaskUpdateTool(manager),
        TeamSendMessageTool(manager),
    )
    for tool in tools:
        registry.register(tool)
    return tuple(tool.name for tool in tools)


def _resolve_targets(team: Team, recipient: str, sender: str) -> tuple[str, ...]:
    identities = {member.name: member.agent_id or member.name for member in team.members}
    identities["lead"] = team.lead_agent_id
    if recipient == "*":
        targets = tuple(identifier for name, identifier in identities.items() if name != sender)
        if not targets:
            raise ValueError("broadcast has no recipients")
        return targets
    if recipient == "lead":
        return (team.lead_agent_id,)
    member = next(
        (
            item
            for item in team.members
            if item.name.casefold() == recipient.casefold()
            or (item.agent_id and item.agent_id == recipient)
        ),
        None,
    )
    if member is None:
        raise ValueError(f"unknown Team recipient: {recipient}")
    return (member.agent_id or member.name,)


def _parse(
    arguments_json: str,
    *,
    required: set[str] | None = None,
    optional: set[str] | None = None,
) -> dict[str, object] | Result:
    if not isinstance(arguments_json, str):
        return _error("invalid_arguments", "Arguments must be a JSON string.")
    try:
        parsed = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return _error("invalid_arguments", "Arguments must be valid JSON.")
    if not isinstance(parsed, dict):
        return _error("invalid_arguments", "Arguments must be a JSON object.")
    required = required or set()
    optional = optional or set()
    if set(parsed) - required - optional or not required.issubset(parsed):
        return _error("invalid_arguments", "Required fields are missing or unknown fields exist.")
    return parsed


def _required_text(parsed: dict[str, object], name: str) -> str | Result:
    value = parsed.get(name)
    if not isinstance(value, str) or not value or value != value.strip():
        return _error("invalid_arguments", f"{name} must be a non-empty trimmed string.")
    return value


def _optional_text(parsed: dict[str, object], name: str) -> str | Result:
    value = parsed.get(name, "")
    if not isinstance(value, str) or value != value.strip():
        return _error("invalid_arguments", f"{name} must be a trimmed string.")
    return value


def _string_list(parsed: dict[str, object], name: str) -> tuple[str, ...] | Result:
    value = parsed.get(name, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item or item != item.strip() for item in value
    ):
        return _error("invalid_arguments", f"{name} must be an array of trimmed strings.")
    return tuple(value)


def _booleans(parsed: dict[str, object], *names: str) -> dict[str, bool] | Result:
    values: dict[str, bool] = {}
    for name in names:
        value = parsed.get(name, False)
        if not isinstance(value, bool):
            return _error("invalid_arguments", f"{name} must be a boolean.")
        values[name] = value
    return values


def _task_dict(task: TeamTask, *, ready: bool | None = None) -> dict[str, object]:
    result = task.to_dict()
    if ready is not None:
        result["is_ready"] = ready
    return result


def _team_summary(team: Team) -> dict[str, object]:
    return {"name": team.name, "slug": team.slug, "backend": team.backend.value}


def _team_dir(team: Team):
    return team.config_path.parent


async def _notify(callback: Callable[[Team | None], object] | None, team: Team | None) -> None:
    if callback is None:
        return
    result = callback(team)
    if inspect.isawaitable(result):
        await result


def _mapped_error(error: Exception) -> Result:
    if isinstance(error, TeamError):
        return _error(error.code, error.safe_message)
    return _error("invalid_team", str(error))


def _json(
    payload: object,
    *,
    error: bool = False,
    error_code: str = "",
) -> Result:
    return Result(
        json.dumps(payload, ensure_ascii=False, sort_keys=True),
        is_error=error,
        error_code=error_code if error else None,
    )


def _error(code: str, message: str) -> Result:
    return Result(message, is_error=True, error_code=code)


__all__ = [
    "TeamCreateTool",
    "TeamDeleteTool",
    "TeamSendMessageTool",
    "TeamTaskCreateTool",
    "TeamTaskGetTool",
    "TeamTaskListTool",
    "TeamTaskUpdateTool",
    "register_team_tools",
]
