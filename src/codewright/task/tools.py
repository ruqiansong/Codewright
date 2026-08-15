"""Model-facing tools for inspecting and controlling background tasks."""

from __future__ import annotations

import json
from collections.abc import Mapping

from codewright.task.manager import BackgroundTask, Manager, ManagerError, Status
from codewright.tool import Result

_TASK_ID_SCHEMA: Mapping[str, object] = {
    "type": "object",
    "properties": {"task_id": {"type": "string", "description": "Background task id."}},
    "required": ["task_id"],
    "additionalProperties": False,
}


class TaskListTool:
    name = "TaskList"
    description = "List all background subagent tasks and their current status."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    read_only = True

    def __init__(self, manager: Manager) -> None:
        _validate_manager(manager)
        self._manager = manager

    async def execute(self, arguments_json: str) -> Result:
        parsed = _parse_object(arguments_json)
        if isinstance(parsed, Result):
            return parsed
        if parsed:
            return _error("invalid_arguments", "TaskList does not accept arguments.")
        tasks = await self._manager.list()
        return _json_result({"tasks": [_task_summary(task) for task in tasks]})


class TaskGetTool:
    name = "TaskGet"
    description = "Get safe details for one background subagent task."
    parameters = _TASK_ID_SCHEMA
    read_only = True

    def __init__(self, manager: Manager) -> None:
        _validate_manager(manager)
        self._manager = manager

    async def execute(self, arguments_json: str) -> Result:
        task_id = _parse_single_string(arguments_json, "task_id")
        if isinstance(task_id, Result):
            return task_id
        task = await self._manager.get(task_id)
        if task is None:
            return _error("unknown_task", f"Unknown task: {task_id}")
        return _json_result(_task_detail(task))


class TaskStopTool:
    name = "TaskStop"
    description = "Cancel one running background subagent task."
    parameters = _TASK_ID_SCHEMA
    read_only = False

    def __init__(self, manager: Manager) -> None:
        _validate_manager(manager)
        self._manager = manager

    async def execute(self, arguments_json: str) -> Result:
        task_id = _parse_single_string(arguments_json, "task_id")
        if isinstance(task_id, Result):
            return task_id
        try:
            task = await self._manager.stop(task_id)
        except ManagerError as error:
            return _error(error.code, error.safe_message)
        return _json_result({"task_id": task.id, "status": "cancellation_requested"})


class SendMessageTool:
    name = "SendMessage"
    description = "Continue a named completed background subagent task."
    parameters: Mapping[str, object] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Named task to continue."},
            "message": {"type": "string", "description": "New task message."},
        },
        "required": ["name", "message"],
        "additionalProperties": False,
    }
    read_only = False

    def __init__(self, manager: Manager) -> None:
        _validate_manager(manager)
        self._manager = manager

    async def execute(self, arguments_json: str) -> Result:
        parsed = _parse_object(arguments_json)
        if isinstance(parsed, Result):
            return parsed
        if set(parsed) != {"name", "message"}:
            return _error("invalid_arguments", "Exactly name and message arguments are required.")
        name = _trimmed_string(parsed["name"], "name")
        if isinstance(name, Result):
            return name
        message = _trimmed_string(parsed["message"], "message")
        if isinstance(message, Result):
            return message
        try:
            task = await self._manager.send_message(name, message)
        except ManagerError as error:
            return _error(error.code, error.safe_message)
        return _json_result({"task_id": task.id, "status": Status.RUNNING.value})


def _task_summary(task: BackgroundTask) -> dict[str, object]:
    return {
        "task_id": task.id,
        "name": task.name,
        "description": task.description,
        "status": task.status.value,
        "tool_count": task.tool_count,
        "last_activity": task.last_activity,
    }


def _task_detail(task: BackgroundTask) -> dict[str, object]:
    return {
        **_task_summary(task),
        "initial_prompt": task.initial_prompt,
        "result": task.result,
        "error_type": task.error_type,
        "error_message": task.error_message,
        "started_at": task.started_at,
        "ended_at": task.ended_at,
        "usage": {
            "input_tokens": task.usage.input_tokens,
            "output_tokens": task.usage.output_tokens,
            "total_tokens": task.usage.total_tokens,
            "cache_write_tokens": task.usage.cache_write_tokens,
            "cache_read_tokens": task.usage.cache_read_tokens,
        },
        "notification_generation": task.notification_generation,
    }


def _parse_single_string(arguments_json: str, key: str) -> str | Result:
    parsed = _parse_object(arguments_json)
    if isinstance(parsed, Result):
        return parsed
    if set(parsed) != {key}:
        return _error("invalid_arguments", f"Exactly one {key} argument is required.")
    return _trimmed_string(parsed[key], key)


def _parse_object(arguments_json: str) -> dict[str, object] | Result:
    if not isinstance(arguments_json, str):
        return _error("invalid_arguments", "Arguments must be a JSON string.")
    try:
        parsed = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return _error("invalid_arguments", "Arguments must be valid JSON.")
    if not isinstance(parsed, dict):
        return _error("invalid_arguments", "Arguments must be a JSON object.")
    return parsed


def _trimmed_string(value: object, name: str) -> str | Result:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        return _error("invalid_arguments", f"{name} must be a non-empty trimmed string.")
    return value


def _json_result(payload: object) -> Result:
    return Result(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _error(code: str, message: str) -> Result:
    return Result(message, is_error=True, error_code=code)


def _validate_manager(manager: Manager) -> None:
    if not isinstance(manager, Manager):
        raise TypeError("manager must be a Manager")


__all__ = ["SendMessageTool", "TaskGetTool", "TaskListTool", "TaskStopTool"]
