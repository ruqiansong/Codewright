"""Explicitly enabled real DeepSeek API smoke tests."""

import os

import pytest

from codewright.config import ProviderConfig
from codewright.llm import Message, MessageRole
from codewright.llm.deepseek import DeepSeekProvider

_RUN_INTEGRATION = os.getenv("CODEWRIGHT_RUN_DEEPSEEK_INTEGRATION") == "1"
_API_KEY = os.getenv("DEEPSEEK_API_KEY")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _RUN_INTEGRATION or not _API_KEY,
        reason="requires explicit DeepSeek integration opt-in and DEEPSEEK_API_KEY",
    ),
]


def integration_provider() -> DeepSeekProvider:
    """Build a real Provider exclusively from environment-managed credentials."""
    if not _API_KEY:
        raise RuntimeError("DeepSeek integration credentials are unavailable")
    config = ProviderConfig.model_validate(
        {
            "name": "deepseek-integration",
            "protocol": "openai-compatible",
            "api_key": _API_KEY,
            "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            "timeout_seconds": 30,
            "max_tokens": 32,
        }
    )
    return DeepSeekProvider(config)


def integration_messages() -> tuple[Message, ...]:
    return (
        Message(MessageRole.SYSTEM, "Reply briefly and only with text."),
        Message(MessageRole.USER, "Reply with the word OK."),
    )


@pytest.mark.asyncio
async def test_real_deepseek_chat() -> None:
    provider = integration_provider()
    try:
        result = await provider.chat(integration_messages())
        assert result.message.role is MessageRole.ASSISTANT
        assert result.message.content.strip()
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_real_deepseek_stream() -> None:
    provider = integration_provider()
    try:
        events = [event async for event in provider.stream_chat(integration_messages())]
        assert sum(event.done for event in events) == 1
        assert all(event.error is None for event in events)
        assert "".join(event.text for event in events).strip()
    finally:
        await provider.close()
