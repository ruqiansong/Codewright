"""Tests for deterministic summary prompt construction."""

import logging

from codewright.compact.summary_prompt import (
    build_summary_prompt,
    extract_summary,
    serialize_conversation,
)
from codewright.llm import Message, MessageRole, ToolCall, ToolResult


def conversation() -> list[Message]:
    call = ToolCall("call-1", "read_file", '{"path":"README.md"}')
    result = ToolResult("call-1", "read_file", "contents")
    return [
        Message(MessageRole.SYSTEM, "You are Codewright."),
        Message(MessageRole.USER, "Read the README."),
        Message(MessageRole.ASSISTANT, "", tool_calls=(call,)),
        Message(MessageRole.TOOL, "", tool_results=(result,)),
    ]


def test_serialize_conversation_is_deterministic_and_complete() -> None:
    messages = conversation()

    serialized = serialize_conversation(messages)

    assert serialized == serialize_conversation(messages)
    assert "system: You are Codewright." in serialized
    assert "user: Read the README." in serialized
    assert '[call read_file id=call-1 args={"path":"README.md"}]' in serialized
    assert "[result id=call-1 name=read_file is_error=false] contents" in serialized


def test_build_summary_prompt_has_one_user_message_and_nine_sections() -> None:
    prompt = build_summary_prompt(conversation())

    assert len(prompt) == 1
    assert prompt[0].role is MessageRole.USER
    assert "<analysis>...</analysis>" in prompt[0].content
    assert "<summary>...</summary>" in prompt[0].content
    assert "不要调用任何工具" in prompt[0].content
    for section in range(1, 10):
        assert f"## {section} " in prompt[0].content


def test_extract_summary_uses_last_tagged_block() -> None:
    raw = "<summary>old</summary> text <summary> final </summary>"

    assert extract_summary(raw) == "final"


def test_extract_summary_falls_back_without_tags(caplog: object) -> None:
    raw = "untagged response"

    with caplog.at_level(logging.WARNING):  # type: ignore[attr-defined]
        result = extract_summary(raw)

    assert result == raw
    assert "Summary tags not found" in caplog.text  # type: ignore[attr-defined]
