"""Vendor-neutral language model interfaces for Codewright."""

from codewright.llm.errors import (
    LLMAuthenticationError,
    LLMError,
    LLMModelNotFoundError,
    LLMNetworkError,
    LLMRateLimitError,
    LLMResponseError,
    LLMServiceError,
    LLMTimeoutError,
    PromptTooLongError,
)
from codewright.llm.models import (
    ChatResult,
    Message,
    MessageRole,
    RequestContext,
    StreamEvent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from codewright.llm.provider import Provider, RequestParameters

__all__ = [
    "ChatResult",
    "LLMAuthenticationError",
    "LLMError",
    "LLMModelNotFoundError",
    "LLMNetworkError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMServiceError",
    "LLMTimeoutError",
    "Message",
    "MessageRole",
    "Provider",
    "PromptTooLongError",
    "RequestContext",
    "RequestParameters",
    "StreamEvent",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
]
