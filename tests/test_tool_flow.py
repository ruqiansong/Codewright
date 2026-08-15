"""Cross-protocol offline acceptance tests for the complete tool flow."""

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import pytest

from codewright.agent import Agent, Event, Phase
from codewright.config import ProviderConfig
from codewright.conversation import Conversation
from codewright.llm import MessageRole, Provider, ToolCall
from codewright.llm.anthropic_provider import AnthropicProvider
from codewright.llm.openai_provider import OpenAICompatibleProvider
from codewright.permission import Engine, Mode
from codewright.permission.rule import RuleSet
from codewright.prompt import SYSTEM_PROMPT
from codewright.tool import new_default_registry
from codewright.utils.logging import register_secrets


class _ClosableClient(Protocol):
    async def close(self) -> None: ...


def permission_engine(root: Path) -> Engine:
    return Engine(
        root=root.resolve(),
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=root / ".codewright" / "settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


class FakeAsyncStream:
    def __init__(self, items: Sequence[object]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> AsyncIterator[object]:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration from None


class OpenAICompletions:
    def __init__(self, responses: Sequence[object]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, object]] = []

    async def create(self, **parameters: object) -> object:
        self.requests.append(parameters)
        return next(self._responses)


class OpenAIClient:
    def __init__(self, responses: Sequence[object]) -> None:
        self.completions = OpenAICompletions(responses)
        self.chat = SimpleNamespace(completions=self.completions)

    async def close(self) -> None:
        return None


class AnthropicMessageStream(FakeAsyncStream):
    def __init__(self, items: Sequence[object], final_message: object) -> None:
        super().__init__(items)
        self._final_message = final_message

    async def get_final_message(self) -> object:
        return self._final_message


class AnthropicManager:
    def __init__(self, stream: AnthropicMessageStream) -> None:
        self._stream = stream

    async def __aenter__(self) -> AnthropicMessageStream:
        return self._stream

    async def __aexit__(self, *args: object) -> None:
        return None


class AnthropicMessages:
    def __init__(self, streams: Sequence[AnthropicMessageStream]) -> None:
        self._streams = iter(streams)
        self.requests: list[dict[str, object]] = []

    def stream(self, **parameters: object) -> AnthropicManager:
        self.requests.append(parameters)
        return AnthropicManager(next(self._streams))

    async def create(self, **parameters: object) -> object:
        raise AssertionError("acceptance flow uses streaming")


class AnthropicClient:
    def __init__(self, streams: Sequence[AnthropicMessageStream]) -> None:
        self.messages = AnthropicMessages(streams)

    async def close(self) -> None:
        return None


def provider_config(protocol: str) -> ProviderConfig:
    return ProviderConfig.model_validate(
        {
            "name": protocol,
            "protocol": protocol,
            "api_key": "offline-synthetic-key",
            "base_url": "https://example.invalid",
            "model": "offline-model",
            "max_tokens": 256,
        }
    )


def openai_chunk(*, text: str | None = None, tool_call: object | None = None) -> object:
    calls = None if tool_call is None else [tool_call]
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=calls))]
    )


def build_openai_provider(path: Path) -> tuple[Provider, OpenAICompletions]:
    raw_call = SimpleNamespace(
        index=0,
        id="call-1",
        function=SimpleNamespace(name="read_file", arguments=f'{{"path":"{path}"}}'),
    )
    client = OpenAIClient(
        [
            FakeAsyncStream([openai_chunk(tool_call=raw_call)]),
            FakeAsyncStream([openai_chunk(text="Summary: Codewright tool flow fixture.")]),
        ]
    )
    provider = OpenAICompatibleProvider(
        provider_config("openai-compatible"),
        client=client,
    )
    return provider, client.completions


def anthropic_final(*blocks: object) -> object:
    return SimpleNamespace(
        content=list(blocks),
        model="offline-model",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def build_anthropic_provider(path: Path) -> tuple[Provider, AnthropicMessages]:
    first_final = anthropic_final(
        SimpleNamespace(type="tool_use", id="call-1", name="read_file", input={"path": str(path)})
    )
    second_final = anthropic_final(
        SimpleNamespace(type="text", text="Summary: Codewright tool flow fixture.")
    )
    text_event = SimpleNamespace(
        delta=SimpleNamespace(type="text_delta", text="Summary: Codewright tool flow fixture.")
    )
    client = AnthropicClient(
        [
            AnthropicMessageStream([], first_final),
            AnthropicMessageStream([text_event], second_final),
        ]
    )
    provider = AnthropicProvider(provider_config("anthropic"), client=client)
    return provider, client.messages


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["openai-compatible", "anthropic"])
async def test_read_file_summary_flow_is_equivalent_across_protocols(
    protocol: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "fixture.txt"
    path.write_text("Codewright tool flow fixture.\n", encoding="utf-8")
    if protocol == "openai-compatible":
        provider, request_recorder = build_openai_provider(path)
    else:
        provider, request_recorder = build_anthropic_provider(path)
    registry = new_default_registry(working_directory=tmp_path)
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_user("Read the fixture and summarize it")

    events = [
        event
        async for event in Agent(provider, registry, permission_engine(tmp_path)).run(conversation)
    ]

    tool_events = [event.tool for event in events if event.tool]
    assert [event.phase for event in tool_events] == [Phase.START, Phase.END]
    assert tool_events[0].name == "read_file"
    assert "Codewright tool flow fixture." in tool_events[1].summary
    assert [event.text for event in events if event.text] == [
        "Summary: Codewright tool flow fixture."
    ]
    assert events[-1] == Event.completed()
    assert len(request_recorder.requests) == 2
    assert len(registry.definitions()) == 6
    assert [message.role for message in conversation.messages()] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert conversation.messages()[2].tool_calls == (
        ToolCall("call-1", "read_file", f'{{"path":"{path}"}}'),
    )
    assert "Codewright tool flow fixture." in conversation.messages()[3].tool_results[0].content
    assert conversation.messages()[-1].content == "Summary: Codewright tool flow fixture."

    if protocol == "openai-compatible":
        second_messages = cast(list[dict[str, object]], request_recorder.requests[1]["messages"])
        assert any(message["role"] == "tool" for message in second_messages)
    else:
        second_messages = cast(list[dict[str, object]], request_recorder.requests[1]["messages"])
        result_blocks = cast(list[dict[str, object]], second_messages[-1]["content"])
        assert result_blocks[0]["type"] == "tool_result"

    await cast(_ClosableClient, provider).close()


@pytest.mark.asyncio
async def test_registered_api_key_is_absent_from_tool_history_events_and_requests(
    tmp_path: Path,
) -> None:
    secret = "unique-stage-two-api-key-value"
    register_secrets((secret,))
    path = tmp_path / "credentials.txt"
    path.write_text(f"credential={secret}\n", encoding="utf-8")
    provider, recorder = build_openai_provider(path)
    conversation = Conversation(SYSTEM_PROMPT)
    conversation.add_user("Inspect the credentials fixture")

    events = [
        event
        async for event in Agent(
            provider,
            new_default_registry(working_directory=tmp_path),
            permission_engine(tmp_path),
        ).run(conversation)
    ]

    assert secret not in repr(events)
    assert secret not in repr(conversation.messages())
    assert secret not in repr(recorder.requests)
    assert "[REDACTED]" in conversation.messages()[3].tool_results[0].content
    await cast(_ClosableClient, provider).close()
