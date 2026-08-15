"""Tool-free LLM orchestration for long-term memory updates."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence

from codewright.llm import Message, MessageRole, Provider
from codewright.memory.prompts import MEMORY_UPDATE_SYSTEM_PROMPT
from codewright.memory.store import MAX_INDEX_BYTES, Store
from codewright.memory.types import UpdateAction

logger = logging.getLogger(__name__)

_INDEX_TRUNCATED = "\n(index truncated)"


class Manager:
    """Coordinate project and user stores using the active conversation provider."""

    def __init__(
        self,
        project_dir: str,
        user_dir: str,
        provider: Provider | None,
        model: str,
    ) -> None:
        if provider is not None and not isinstance(provider, Provider):
            raise TypeError("provider must satisfy Provider or be None")
        if not isinstance(model, str):
            raise TypeError("model must be a string")
        self.project_store = Store(project_dir)
        self.user_store = Store(user_dir)
        self._provider = provider
        self._model = model.strip()
        self._lock = asyncio.Lock()

    def load_index(self) -> str:
        """Return a labeled, project-first snapshot bounded by UTF-8 bytes."""
        sections: list[str] = []
        project = self.project_store.load_index().strip()
        user = self.user_store.load_index().strip()
        if project:
            sections.append(f"[project memory]\n{project}")
        if user:
            sections.append(f"[user memory]\n{user}")
        combined = "\n\n".join(sections)
        encoded = combined.encode("utf-8")
        if len(encoded) <= MAX_INDEX_BYTES:
            return combined
        available = MAX_INDEX_BYTES - len(_INDEX_TRUNCATED.encode())
        head = encoded[:available].decode("utf-8", errors="ignore")
        return head + _INDEX_TRUNCATED

    def list_files(self) -> tuple[list[str], list[str]]:
        """Return sorted project and user Markdown filenames without reading them."""
        return (
            _list_store_files(self.project_store, "project"),
            _list_store_files(self.user_store, "user"),
        )

    async def update_async(self, recent_messages: list[Message]) -> None:
        """Extract and apply memory updates; all failures are isolated from chat."""
        try:
            messages = _validated_recent_messages(recent_messages)
            if self._provider is None:
                raise RuntimeError("memory provider is unavailable")
            async with self._lock:
                request = (
                    Message(MessageRole.SYSTEM, MEMORY_UPDATE_SYSTEM_PROMPT),
                    Message(
                        MessageRole.USER,
                        _build_update_input(messages, self.load_index()),
                    ),
                )
                chunks: list[str] = []
                async for event in self._provider.stream_chat(request, tools=()):
                    if event.error is not None:
                        raise event.error
                    if event.text:
                        chunks.append(event.text)
                    if event.tool_calls:
                        raise ValueError("memory update must not request tools")
                actions = _parse_actions("".join(chunks))
                project_actions = [action for action in actions if action.level == "project"]
                user_actions = [action for action in actions if action.level == "user"]
                if project_actions:
                    await asyncio.to_thread(self.project_store.apply, project_actions)
                if user_actions:
                    await asyncio.to_thread(self.user_store.apply, user_actions)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.error("Memory update failed error=%s", type(error).__name__)


def _validated_recent_messages(messages: list[Message]) -> list[Message]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("recent_messages must be a non-empty list")
    if not all(isinstance(message, Message) for message in messages):
        raise TypeError("recent_messages must contain only Message values")
    return list(messages)


def _list_store_files(store: Store, level: str) -> list[str]:
    directory = store.directory
    try:
        if not directory.exists():
            return []
        if directory.is_symlink() or not directory.is_dir():
            logger.warning("Unsafe memory directory ignored level=%s", level)
            return []
        return sorted(
            path.name
            for path in directory.iterdir()
            if path.suffix == ".md" and path.is_file() and not path.is_symlink()
        )
    except OSError as error:
        logger.warning(
            "Memory directory could not be listed level=%s error=%s",
            level,
            type(error).__name__,
        )
        return []


def _build_update_input(messages: Sequence[Message], index: str) -> str:
    history: list[str] = []
    for message in messages:
        content = message.content
        if message.tool_calls:
            content = (
                f"{content}\n[tool calls: {', '.join(call.name for call in message.tool_calls)}]"
            )
        if message.tool_results:
            content = (
                f"[tool results: {', '.join(result.tool_name for result in message.tool_results)}]"
            )
        history.append(f"{message.role.value}: {content}")
    current_index = index or "(empty)"
    return (
        "现有记忆索引：\n"
        f"{current_index}\n\n"
        "最近一轮对话：\n" + "\n".join(history) + "\n\n请返回严格 JSON 数组。"
    )


def _parse_actions(raw: str) -> list[UpdateAction]:
    value = json.loads(raw.strip())
    if not isinstance(value, list):
        raise ValueError("memory response must be a JSON array")
    actions: list[UpdateAction] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("memory action must be an object")
        allowed = {"action", "level", "type", "title", "slug", "content", "filename"}
        if any(key not in allowed for key in item):
            raise ValueError("memory action contains unknown fields")
        actions.append(
            UpdateAction(
                action=_string_field(item, "action", required=True),
                level=_string_field(item, "level", required=True),
                type=_string_field(item, "type"),
                title=_string_field(item, "title"),
                slug=_string_field(item, "slug"),
                content=_string_field(item, "content"),
                filename=_string_field(item, "filename"),
            )
        )
    return actions


def _string_field(value: dict[str, object], name: str, *, required: bool = False) -> str:
    raw = value.get(name, "")
    if not isinstance(raw, str) or (required and not raw.strip()):
        raise ValueError(f"memory action field {name} must be a string")
    return raw.strip()
