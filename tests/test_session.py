"""Tests for durable session writing, discovery, loading, and cleanup."""

from __future__ import annotations

import json
import os
import threading
from datetime import timedelta
from pathlib import Path

import pytest

from codewright.llm import Message, MessageRole, ToolCall, ToolResult
from codewright.session import Writer, clean_expired, list_sessions, load_session


def session_dir(root: Path, session_id: str = "20260812-142305-a1b2") -> Path:
    return root / ".codewright" / "sessions" / session_id


def records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_writer_appends_messages_and_round_trips_tool_data(tmp_path: Path) -> None:
    directory = session_dir(tmp_path)
    call = ToolCall("call-1", "read_file", '{"path":"README.md"}')
    result = ToolResult(
        "call-1",
        "read_file",
        "content",
        metadata={"path": "README.md"},
    )
    with Writer(str(directory), "test-model") as writer:
        writer.append(Message(MessageRole.SYSTEM, "not persisted"))
        writer.append(Message(MessageRole.USER, "read it"))
        writer.append(Message(MessageRole.ASSISTANT, "", tool_calls=(call,)))
        writer.append(Message(MessageRole.TOOL, "", tool_results=(result,)))

    values = records(directory / "conversation.jsonl")
    assert [value["role"] for value in values] == ["user", "assistant", "tool"]
    assert values[0]["model"] == "test-model"
    assert "model" not in values[1]
    assert values[1]["tool_calls"] == [
        {"id": "call-1", "name": "read_file", "arguments_json": '{"path":"README.md"}'}
    ]
    assert values[2]["tool_results"][0]["metadata"] == {"path": "README.md"}

    loaded = load_session(str(directory))
    assert loaded.messages == [
        Message(MessageRole.USER, "read it"),
        Message(MessageRole.ASSISTANT, "", tool_calls=(call,)),
        Message(MessageRole.TOOL, "", tool_results=(result,)),
    ]
    assert loaded.model == "test-model"
    assert loaded.last_message_ts is not None


def test_writer_compact_callback_is_contiguous_and_skips_system(tmp_path: Path) -> None:
    directory = session_dir(tmp_path)
    writer = Writer(str(directory), "model")
    writer.on_append(Message(MessageRole.USER, "old"))
    writer.on_replace(
        [
            Message(MessageRole.SYSTEM, "current system"),
            Message(MessageRole.USER, "summary"),
            Message(MessageRole.ASSISTANT, "answer"),
        ]
    )
    writer.close()

    values = records(directory / "conversation.jsonl")
    assert [value.get("type") or value.get("role") for value in values] == [
        "user",
        "compact",
        "user",
        "assistant",
    ]
    assert all(value.get("role") != "system" for value in values)
    assert [message.content for message in load_session(str(directory)).messages] == [
        "summary",
        "answer",
    ]


def test_writer_open_existing_appends_without_truncating(tmp_path: Path) -> None:
    directory = session_dir(tmp_path)
    with Writer(str(directory), "original-model") as writer:
        writer.append(Message(MessageRole.USER, "first"))
    original_size = (directory / "conversation.jsonl").stat().st_size

    with Writer.open_existing(str(directory), "current-model") as writer:
        writer.append(Message(MessageRole.ASSISTANT, "second"))

    values = records(directory / "conversation.jsonl")
    assert (directory / "conversation.jsonl").stat().st_size > original_size
    assert [value["content"] for value in values] == ["first", "second"]
    assert values[0]["model"] == "original-model"
    assert "model" not in values[1]


def test_writer_close_is_idempotent_and_closed_write_is_controlled(tmp_path: Path) -> None:
    writer = Writer(str(session_dir(tmp_path)), "model")
    writer.close()
    writer.close()

    with pytest.raises(RuntimeError, match="closed"):
        writer.append(Message(MessageRole.USER, "late"))


def test_writer_callbacks_isolate_closed_writer_errors(tmp_path: Path, caplog: object) -> None:
    writer = Writer(str(session_dir(tmp_path)), "model")
    writer.close()

    writer.on_append(Message(MessageRole.USER, "safe"))
    writer.on_replace([Message(MessageRole.SYSTEM, "system")])


def test_writer_concurrent_appends_produce_complete_json_lines(tmp_path: Path) -> None:
    directory = session_dir(tmp_path)
    writer = Writer(str(directory), "model")
    threads = [
        threading.Thread(target=writer.append, args=(Message(MessageRole.USER, f"m-{i}"),))
        for i in range(30)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    writer.close()

    values = records(directory / "conversation.jsonl")
    assert len(values) == 30
    assert {value["content"] for value in values} == {f"m-{i}" for i in range(30)}
    assert sum("model" in value for value in values) == 1


def test_load_session_skips_bad_json_and_invalid_records(tmp_path: Path) -> None:
    directory = session_dir(tmp_path)
    directory.mkdir(parents=True)
    path = directory / "conversation.jsonl"
    path.write_text(
        "\n".join(
            (
                '{"role":"user","content":"valid","ts":1,"model":"m"}',
                "{bad json",
                '{"role":"assistant","content":42,"ts":2}',
                '{"role":"assistant","content":"ok","ts":3}',
            )
        )
        + "\n"
    )

    loaded = load_session(str(directory))

    assert [message.content for message in loaded.messages] == ["valid", "ok"]
    assert loaded.last_message_ts == 3


def test_load_session_skips_invalid_utf8_line(tmp_path: Path) -> None:
    directory = session_dir(tmp_path)
    directory.mkdir(parents=True)
    path = directory / "conversation.jsonl"
    path.write_bytes(
        b'{"role":"user","content":"before","ts":1,"model":"m"}\n'
        b"\xff\xfe\n"
        b'{"role":"assistant","content":"after","ts":2}\n'
    )

    loaded = load_session(str(directory))

    assert [message.content for message in loaded.messages] == ["before", "after"]
    assert loaded.last_message_ts == 2


def test_load_session_truncates_orphaned_tool_call_and_timestamp(tmp_path: Path) -> None:
    directory = session_dir(tmp_path)
    with Writer(str(directory), "model") as writer:
        writer.append(Message(MessageRole.USER, "before"))
        writer.append(
            Message(
                MessageRole.ASSISTANT,
                "",
                tool_calls=(ToolCall("call-1", "read_file"),),
            )
        )

    values = records(directory / "conversation.jsonl")
    loaded = load_session(str(directory))

    assert loaded.messages == [Message(MessageRole.USER, "before")]
    assert loaded.last_message_ts == values[0]["ts"]


def test_list_sessions_filters_and_orders_with_bounded_titles(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    identifiers = (
        "20260810-100000-a1b2",
        "20260811-100000-a1b3",
        "20260812-100000-a1b4",
    )
    for index, identifier in enumerate(identifiers):
        with Writer(str(root / identifier), f"model-{index}") as writer:
            writer.append(Message(MessageRole.USER, ("title " * 20) + str(index)))
        os.utime(root / identifier / "conversation.jsonl", (100 + index, 100 + index))
    old = root / "1717000000-abc12345"
    old.mkdir()
    (old / "conversation.jsonl").write_text("{}\n")
    empty = root / "20260812-100000-a1b5"
    empty.mkdir()
    (empty / "conversation.jsonl").touch()

    sessions = list_sessions(str(root))

    assert [session.id for session in sessions] == list(reversed(identifiers))
    assert all(len(session.title) <= 50 for session in sessions)
    assert sessions[0].model == "model-2"


def test_clean_expired_only_removes_old_current_format_directories(tmp_path: Path) -> None:
    root = tmp_path / "sessions"
    expired = root / "20200101-000000-a1b2"
    current = root / "20990101-000000-a1b3"
    legacy = root / "1717000000-abc12345"
    for directory in (expired, current, legacy):
        directory.mkdir(parents=True)
        (directory / "marker").write_text("x")

    clean_expired(str(root), timedelta(days=30))

    assert not expired.exists()
    assert current.exists()
    assert legacy.exists()


def test_missing_directories_degrade_safely(tmp_path: Path) -> None:
    missing = tmp_path / "missing"

    assert list_sessions(str(missing)) == []
    clean_expired(str(missing), timedelta(days=30))
    with pytest.raises(FileNotFoundError):
        load_session(str(missing))
    with pytest.raises(FileNotFoundError):
        Writer.open_existing(str(session_dir(tmp_path)), "model")
