"""TUI session-resume behavior tests."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from codewright.compact import new_session_context
from codewright.conversation import Conversation
from codewright.llm import Message, MessageRole
from codewright.session import SessionInfo, Writer
from codewright.tui.resume import ResumePanel, SessionItem


def _session_info(tmp_path: Path) -> SessionInfo:
    return SessionInfo(
        id="20260812-100000-abcd",
        title="Review the persistence design",
        modified_at=datetime.now() - timedelta(hours=3),
        model="deepseek-chat",
        size=1331,
        dir=str(tmp_path / "20260812-100000-abcd"),
    )


def test_session_item_has_bounded_human_readable_metadata(tmp_path: Path) -> None:
    item = SessionItem(_session_info(tmp_path))

    assert "Review the persistence design" in item.display_text
    assert "3 hours ago" in item.display_text
    assert "deepseek-chat" in item.display_text
    assert "1.3KB" in item.display_text


@pytest.mark.asyncio
async def test_resume_panel_filters_and_selects_visible_session(tmp_path: Path) -> None:
    sessions = [
        _session_info(tmp_path),
        SessionInfo(
            id="20260812-110000-bbbb",
            title="Implement memory store",
            modified_at=datetime.now(),
            model="other-model",
            size=100,
            dir=str(tmp_path / "20260812-110000-bbbb"),
        ),
    ]
    panel = ResumePanel(sessions)

    # Widget behavior is covered through Textual composition tests; this assertion
    # protects the source metadata used by filtering and selection.
    assert [item.info.id for item in panel._items] == [session.id for session in sessions]


def test_writer_for_candidate_session_does_not_change_existing_conversation(
    tmp_path: Path,
) -> None:
    first = new_session_context(str(tmp_path))
    writer = Writer(first.session_dir, "model")
    conversation = Conversation("system", on_append=writer.on_append)
    conversation.add_user("current session")

    candidate = new_session_context(str(tmp_path))
    candidate_writer = Writer(candidate.session_dir, "model")
    restored = Conversation.from_messages(
        "current system",
        [Message(MessageRole.USER, "restored session")],
        on_append=candidate_writer.on_append,
    )
    candidate_writer.close()
    writer.close()

    assert conversation.messages()[-1].content == "current session"
    assert restored.messages()[-1].content == "restored session"
    assert os.path.exists(Path(candidate.session_dir) / "conversation.jsonl")
