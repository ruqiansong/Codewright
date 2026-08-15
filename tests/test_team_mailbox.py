from __future__ import annotations

from pathlib import Path

from codewright.team.mailbox import Mailbox, Message, MessageType


async def test_mailbox_append_peek_and_atomic_consume(tmp_path: Path) -> None:
    mailbox = Mailbox(tmp_path)
    message = Message(sender="alice", type=MessageType.TEXT, content="hello")
    await mailbox.append("lead", message)
    assert (await mailbox.peek("lead"))[0].read_at is None
    consumed = await mailbox.consume_unread("lead")
    assert consumed[0].id == message.id
    assert consumed[0].read_at is not None
    assert await mailbox.consume_unread("lead") == ()
