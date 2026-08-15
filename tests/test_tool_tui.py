"""Textual Pilot tests for Agent-driven tool lifecycle presentation."""

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest

from codewright.conversation import Conversation
from codewright.llm import (
    ChatResult,
    Message,
    MessageRole,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)
from codewright.permission import Engine, Mode
from codewright.permission.rule import Rule, RuleSet
from codewright.prompt import SYSTEM_PROMPT
from codewright.tool import Registry, Result
from codewright.tui import ChatScreen, ChatState, CodewrightApp
from codewright.tui.widgets.input import MessageInput
from codewright.tui.widgets.message import ConversationMessage
from codewright.tui.widgets.tool import ToolCallWidget


class ToolFlowProvider:
    provider_name = "fake"
    model_name = "fake-tools"

    def __init__(self, call: ToolCall, final_text: str = "Final answer") -> None:
        self.call = call
        self.final_text = final_text
        self.requests: list[tuple[Message, ...]] = []
        self.closed = False

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del messages, parameters, tools, request_context
        raise AssertionError("tool TUI test uses streaming")

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters, request_context
        assert tools
        self.requests.append(tuple(messages))
        if len(self.requests) == 1:
            yield StreamEvent.tool_calls_ready((self.call,))
        else:
            yield StreamEvent.delta(self.final_text)
        yield StreamEvent.completed()

    async def close(self) -> None:
        self.closed = True


@dataclass(slots=True)
class ControlledTool:
    name: str
    result: Result
    pause: bool = False
    description: str = "Controlled tool for TUI tests."
    parameters: Mapping[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only: bool = True
    started: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def execute(self, arguments_json: str) -> Result:
        del arguments_json
        self.started.set()
        if self.pause:
            await self.release.wait()
        return self.result


def build_tool_app(
    call: ToolCall,
    tool: ControlledTool,
) -> tuple[CodewrightApp, Conversation, ToolFlowProvider]:
    provider = ToolFlowProvider(call)
    registry = Registry()
    registry.register(tool)
    conversation = Conversation(SYSTEM_PROMPT)
    root = Path.cwd().resolve()
    engine = Engine(
        root=root,
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(allow=[Rule("Write", None, True), Rule("Edit", None, True)]),
        local_path=root / ".codewright" / "settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )
    app = CodewrightApp(
        provider,
        conversation,
        registry,
        engine=engine,
        working_directory=Path("/workspace/codewright"),
        version="0.5.0",
    )
    return app, conversation, provider


def active_screen(app: CodewrightApp) -> ChatScreen:
    return cast(ChatScreen, app.screen)


@pytest.mark.asyncio
async def test_tool_line_appears_running_then_completes_and_remains_in_scrollback() -> None:
    call = ToolCall("call-1", "read_file", '{"path":"README.md"}')
    tool = ControlledTool("read_file", Result("first line\nsecond line"), pause=True)
    app, conversation, provider = build_tool_app(call, tool)

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press(*"read the readme", "enter")
        await tool.started.wait()
        await pilot.pause()

        screen = active_screen(app)
        tool_view = screen.query_one(ToolCallWidget)
        assert screen.state is ChatState.STREAMING
        assert tool_view.heading == "● Read(README.md)"
        assert "Running" in str(tool_view.render())
        assert screen.query_one(MessageInput).disabled is True

        tool.release.set()
        await pilot.pause()

        assert tool_view.is_complete is True
        assert tool_view.result_summary == "first line\nsecond line"
        assert "first line" in str(tool_view.render())
        assert screen.query(ConversationMessage).last().message_content == "Final answer"
        assert screen.query_one(MessageInput).disabled is False

    assert [message.role for message in conversation.messages()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert len(provider.requests) == 2


@pytest.mark.asyncio
async def test_tool_error_has_distinct_style_and_conversation_recovers() -> None:
    call = ToolCall(
        "call-1",
        "edit_file",
        '{"path":"sample.py","old_string":"old","new_string":"new"}',
    )
    tool = ControlledTool(
        "edit_file",
        Result("old_string matched 0 times", is_error=True, error_code="match_not_found"),
    )
    app, conversation, provider = build_tool_app(call, tool)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*"edit file", "enter")
        await pilot.pause()

        tool_view = active_screen(app).query_one(ToolCallWidget)
        assert tool_view.is_error is True
        assert tool_view.has_class("tool-error")
        assert "old_string matched 0 times" in str(tool_view.render())
        assert active_screen(app).state is ChatState.IDLE

        await pilot.press(*"continue normally", "enter")
        await pilot.pause()

        assert active_screen(app).state is ChatState.IDLE
        assert conversation.messages()[-1] == Message(MessageRole.ASSISTANT, "Final answer")
        assert len(provider.requests) == 3


@pytest.mark.asyncio
async def test_tool_summary_is_eight_lines_and_sensitive_content_is_not_in_heading() -> None:
    secret_content = "unique-sensitive-write-content"
    call = ToolCall(
        "call-1",
        "write_file",
        f'{{"path":"safe.txt","content":"{secret_content}"}}',
    )
    result = Result("\n".join(f"line {index}" for index in range(20)))
    tool = ControlledTool("write_file", result)
    app, _, _ = build_tool_app(call, tool)

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press(*"write file", "enter")
        await pilot.pause()

        tool_view = active_screen(app).query_one(ToolCallWidget)
        assert tool_view.heading == "● Write(safe.txt)"
        assert secret_content not in str(tool_view.render())
        assert len(tool_view.result_summary.splitlines()) == 8
        assert tool_view.result_summary.endswith("[summary truncated]")
