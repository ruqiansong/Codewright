"""Crash-resistant append-only JSONL conversation writer."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

from codewright.compact import parse_session_time
from codewright.llm import Message, MessageRole

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Entry:
    """One serializable conversation or compaction record."""

    role: str = ""
    content: str = ""
    tool_calls: list[dict[str, object]] | None = None
    tool_results: list[dict[str, object]] | None = None
    ts: int = 0
    model: str | None = None
    type: str | None = None


class Writer:
    """Synchronously append complete durable JSONL records for one session."""

    def __init__(self, session_dir: str, model: str) -> None:
        directory = _validated_session_dir(session_dir)
        normalized_model = _validated_model(model)
        directory.parent.mkdir(parents=True, exist_ok=True)
        directory.mkdir(exist_ok=False)
        self._initialize(directory, normalized_model, has_messages=False)

    @classmethod
    def open_existing(cls, session_dir: str, model: str) -> Writer:
        """Open an existing conversation for append without truncating it."""
        directory = _validated_session_dir(session_dir)
        normalized_model = _validated_model(model)
        path = directory / "conversation.jsonl"
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError("session directory does not exist")
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("conversation.jsonl does not exist")
        writer = cls.__new__(cls)
        writer._initialize(directory, normalized_model, has_messages=path.stat().st_size > 0)
        return writer

    def _initialize(self, directory: Path, model: str, *, has_messages: bool) -> None:
        self._path = directory / "conversation.jsonl"
        self._model = model
        self._has_messages = has_messages
        self._lock = threading.Lock()
        self._closed = False
        try:
            self._file = self._path.open("ab")
        except Exception:
            if not has_messages:
                directory.rmdir()
            raise

    @property
    def path(self) -> Path:
        """Return the JSONL path for diagnostics and tests."""
        return self._path

    def append(self, message: Message) -> None:
        """Persist one non-system message and force it to disk."""
        if not isinstance(message, Message):
            raise TypeError("message must be a Message")
        if message.role is MessageRole.SYSTEM:
            return
        with self._lock:
            self._ensure_open_locked()
            self._write_entry_locked(self._message_entry_locked(message))
            self._sync_locked()

    def append_all(self, messages: list[Message]) -> None:
        """Persist non-system messages as one uninterrupted durable batch."""
        values = _validated_messages(messages)
        with self._lock:
            self._ensure_open_locked()
            for message in values:
                if message.role is not MessageRole.SYSTEM:
                    self._write_entry_locked(self._message_entry_locked(message))
            self._sync_locked()

    def write_compact_marker(self) -> None:
        """Persist one compaction boundary marker."""
        with self._lock:
            self._ensure_open_locked()
            self._write_entry_locked(Entry(type="compact", ts=int(time.time())))
            self._sync_locked()

    def on_append(self, message: Message) -> None:
        """Conversation callback that isolates persistence failures."""
        try:
            self.append(message)
        except Exception as error:
            logger.error("Session append failed error=%s", type(error).__name__)

    def on_replace(self, messages: list[Message]) -> None:
        """Conversation callback that atomically records a compacted history."""
        try:
            values = _validated_messages(messages)
            with self._lock:
                self._ensure_open_locked()
                self._write_entry_locked(Entry(type="compact", ts=int(time.time())))
                for message in values:
                    if message.role is not MessageRole.SYSTEM:
                        self._write_entry_locked(self._message_entry_locked(message))
                self._sync_locked()
        except Exception as error:
            logger.error("Session replacement write failed error=%s", type(error).__name__)

    def close(self) -> None:
        """Flush and close the file; repeated calls are harmless."""
        with self._lock:
            if self._closed:
                return
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            finally:
                self._file.close()
                self._closed = True

    def __enter__(self) -> Writer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _message_entry_locked(self, message: Message) -> Entry:
        calls: list[dict[str, object]] = [
            {"id": call.id, "name": call.name, "arguments_json": call.arguments_json}
            for call in message.tool_calls
        ]
        results: list[dict[str, object]] = [
            {
                "tool_call_id": result.tool_call_id,
                "tool_name": result.tool_name,
                "content": result.content,
                "is_error": result.is_error,
                "error_code": result.error_code,
                "truncated": result.truncated,
                "metadata": dict(result.metadata),
            }
            for result in message.tool_results
        ]
        model = None if self._has_messages else self._model
        self._has_messages = True
        return Entry(
            role=str(message.role),
            content=message.content,
            tool_calls=calls or None,
            tool_results=results or None,
            ts=int(time.time()),
            model=model,
        )

    def _write_entry_locked(self, entry: Entry) -> None:
        payload = {key: value for key, value in asdict(entry).items() if value is not None}
        encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        self._file.write(encoded)

    def _sync_locked(self) -> None:
        self._file.flush()
        os.fsync(self._file.fileno())

    def _ensure_open_locked(self) -> None:
        if self._closed:
            raise RuntimeError("session writer is closed")


def _validated_session_dir(value: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("session_dir must be a non-empty string")
    path = Path(value)
    if path.name != value.rstrip("/\\").split("/")[-1] and path.name != Path(value).name:
        raise ValueError("invalid session directory")
    parse_session_time(path.name)
    resolved_parent = path.parent.resolve()
    if resolved_parent.name != "sessions":
        raise ValueError("session directory must be directly below a sessions directory")
    resolved = (resolved_parent / path.name).resolve()
    if resolved.parent != resolved_parent:
        raise ValueError("session directory escapes the sessions root")
    return resolved


def _validated_model(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    return model.strip()


def _validated_messages(messages: list[Message]) -> list[Message]:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    if not all(isinstance(message, Message) for message in messages):
        raise TypeError("messages must contain only Message values")
    return list(messages)
