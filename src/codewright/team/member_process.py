"""Pane teammate mailbox loop with stdin wake and polling fallback."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from codewright.team.mailbox import Mailbox, MessageType
from codewright.team.types import Team

type MessageHandler = Callable[[str], Awaitable[None]]


async def run_member_loop(
    team: Team,
    member_name: str,
    handler: MessageHandler,
    *,
    poll_seconds: float = 2.0,
) -> None:
    member = next((item for item in team.members if item.name == member_name), None)
    if member is None or member.state.value != "starting":
        raise ValueError("pane member reservation does not match")
    mailbox = Mailbox(Path(team.config_dir))
    recipient = member.agent_id or member.name
    while True:
        messages = await mailbox.consume_unread(recipient)
        for message in messages:
            if message.type is MessageType.SHUTDOWN_REQUEST:
                return
            await handler(message.content)
        await asyncio.sleep(poll_seconds)


__all__ = ["run_member_loop"]
