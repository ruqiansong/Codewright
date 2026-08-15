"""Per-recipient durable mailboxes for Agent Team collaboration."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from codewright.team.persistence import FileLock, atomic_write_json, read_json
from codewright.team.types import utc_now

_RECIPIENT = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class MessageType(StrEnum):
    TEXT = "text"
    IDLE_NOTIFICATION = "idle_notification"
    SHUTDOWN_REQUEST = "shutdown_request"
    SHUTDOWN_RESPONSE = "shutdown_response"
    PLAN_APPROVAL_REQUEST = "plan_approval_request"
    PLAN_APPROVAL_RESPONSE = "plan_approval_response"


@dataclass(frozen=True, slots=True)
class Message:
    sender: str
    type: MessageType
    content: str
    id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex}")
    request_id: str = ""
    approve: bool | None = None
    created_at: str = field(default_factory=utc_now)
    read_at: str | None = None

    def __post_init__(self) -> None:
        if not self.id.startswith("msg-") or len(self.id) <= 4:
            raise ValueError("invalid message id")
        if not self.sender or self.sender != self.sender.strip():
            raise ValueError("message sender must be non-empty and trimmed")
        if not isinstance(self.type, MessageType):
            raise TypeError("message type must be a MessageType")
        if not isinstance(self.content, str) or self.content != self.content.strip():
            raise ValueError("message content must be trimmed")
        if not isinstance(self.request_id, str) or self.request_id != self.request_id.strip():
            raise ValueError("request_id must be trimmed")
        if self.type is MessageType.PLAN_APPROVAL_RESPONSE:
            if not self.request_id or not isinstance(self.approve, bool):
                raise ValueError("plan response requires request_id and boolean approve")
        elif self.approve is not None:
            raise ValueError("approve is only valid for plan responses")
        if self.read_at is not None and not isinstance(self.read_at, str):
            raise ValueError("read_at must be null or a timestamp")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "from": self.sender,
            "type": self.type.value,
            "content": self.content,
            "requestId": self.request_id,
            "approve": self.approve,
            "createdAt": self.created_at,
            "readAt": self.read_at,
        }

    @classmethod
    def from_dict(cls, value: object) -> Message:
        keys = {"id", "from", "type", "content", "requestId", "approve", "createdAt", "readAt"}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("invalid message fields")
        try:
            message_type = MessageType(value["type"])
        except (TypeError, ValueError) as error:
            raise ValueError("invalid message type") from error
        strings = ("id", "from", "content", "requestId", "createdAt")
        if any(not isinstance(value[key], str) for key in strings):
            raise ValueError("invalid message string field")
        return cls(
            id=value["id"],
            sender=value["from"],
            type=message_type,
            content=value["content"],
            request_id=value["requestId"],
            approve=value["approve"],
            created_at=value["createdAt"],
            read_at=value["readAt"],
        )


class Mailbox:
    def __init__(self, team_dir: Path) -> None:
        self.directory = team_dir / "mailbox"
        self.directory.mkdir(parents=True, exist_ok=True)

    async def append(self, recipient_id: str, message: Message) -> None:
        path, lock_path = self._paths(recipient_id)
        if not isinstance(message, Message):
            raise TypeError("message must be a Message")
        async with FileLock(lock_path):
            messages = self._load(path)
            if any(item.id == message.id for item in messages):
                raise ValueError("duplicate message id")
            messages.append(message)
            atomic_write_json(path, [item.to_dict() for item in messages])

    async def peek(self, recipient_id: str) -> tuple[Message, ...]:
        path, lock_path = self._paths(recipient_id)
        async with FileLock(lock_path):
            return tuple(self._load(path))

    async def consume_unread(self, recipient_id: str) -> tuple[Message, ...]:
        path, lock_path = self._paths(recipient_id)
        async with FileLock(lock_path):
            messages = self._load(path)
            now = utc_now()
            unread = [message for message in messages if message.read_at is None]
            if not unread:
                return ()
            unread_ids = {message.id for message in unread}
            updated = [
                replace(message, read_at=now) if message.id in unread_ids else message
                for message in messages
            ]
            atomic_write_json(path, [item.to_dict() for item in updated])
            return tuple(replace(message, read_at=now) for message in unread)

    def _paths(self, recipient_id: str) -> tuple[Path, Path]:
        if not isinstance(recipient_id, str) or _RECIPIENT.fullmatch(recipient_id) is None:
            raise ValueError("invalid mailbox recipient")
        return (
            self.directory / f"{recipient_id}.json",
            self.directory / f"{recipient_id}.lock",
        )

    @staticmethod
    def _load(path: Path) -> list[Message]:
        if not path.exists():
            return []
        value = read_json(path)
        if not isinstance(value, list):
            raise ValueError("mailbox must contain a list")
        messages = [Message.from_dict(item) for item in value]
        if len(messages) != len({message.id for message in messages}):
            raise ValueError("duplicate message id")
        return messages


__all__ = ["Mailbox", "Message", "MessageType"]
