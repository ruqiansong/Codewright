"""Anthropic Messages API provider with protocol-neutral tool-call support."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from secrets import token_hex
from time import monotonic
from typing import Any, Protocol, cast

import anthropic
from anthropic import AsyncAnthropic

from codewright.config import ProviderConfig
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
)
from codewright.llm.provider import RequestParameters

_PROTECTED_PARAMETERS = frozenset(
    {"api_key", "base_url", "messages", "model", "system", "tools", "max_tokens"}
)
logger = logging.getLogger(__name__)

_PTL_CODES = frozenset(
    {
        "context_length_exceeded",
        "context_window_exceeded",
        "input_too_long",
        "prompt_too_long",
        "request_too_large",
    }
)
_PTL_TEXT_MARKERS = (
    "maximum context length",
    "context length exceeded",
    "context window exceeded",
    "prompt is too long",
)


class _MessageStream(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...

    async def get_final_message(self) -> object: ...


class _StreamManager(Protocol):
    async def __aenter__(self) -> _MessageStream: ...

    async def __aexit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None: ...


class _MessagesResource(Protocol):
    async def create(self, **parameters: object) -> object: ...

    def stream(self, **parameters: object) -> _StreamManager: ...


class _AsyncClient(Protocol):
    messages: _MessagesResource

    async def close(self) -> None: ...


class AnthropicProvider:
    """Translate the public Provider contract to Anthropic Messages."""

    def __init__(self, config: ProviderConfig, *, client: object | None = None) -> None:
        self._config = config
        if client is None:
            client = AsyncAnthropic(
                api_key=config.api_key.get_secret_value(),
                base_url=str(config.base_url),
                timeout=config.timeout_seconds,
            )
        self._client = cast(_AsyncClient, client)

    @property
    def provider_name(self) -> str:
        return self._config.name

    @property
    def model_name(self) -> str:
        return self._config.model

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        """Return one complete Anthropic response."""
        request_id = token_hex(6)
        started_at = monotonic()
        self._log_started(request_id, mode="chat")
        try:
            response = await self._client.messages.create(
                **self._build_request(
                    messages,
                    parameters=parameters,
                    tools=tools,
                    request_context=request_context,
                )
            )
            result = self._parse_chat_result(response)
        except Exception as error:
            mapped_error = self._map_error(error)
            self._log_failed(request_id, "chat", started_at, mapped_error)
            raise mapped_error from None
        self._log_completed(request_id, "chat", started_at)
        return result

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield text deltas, final tool calls, and one completion event."""
        request_id = token_hex(6)
        started_at = monotonic()
        self._log_started(request_id, mode="stream")
        try:
            manager = self._client.messages.stream(
                **self._build_request(
                    messages,
                    parameters=parameters,
                    tools=tools,
                    request_context=request_context,
                )
            )
            async with manager as stream:
                async for event in stream:
                    delta = getattr(event, "delta", None)
                    delta_type = getattr(delta, "type", None)
                    if delta_type == "text_delta":
                        text = getattr(delta, "text", None)
                        if not isinstance(text, str):
                            raise LLMResponseError()
                        if text:
                            yield StreamEvent.delta(text)
                    # thinking_delta and input_json_delta are deliberately not emitted.
                final_message = await stream.get_final_message()

            calls = self._extract_tool_calls(final_message)
            if calls:
                yield StreamEvent.tool_calls_ready(calls)
            usage = self._parse_usage(getattr(final_message, "usage", None))
            if usage is not None:
                yield StreamEvent.usage_report(usage)
            self._log_completed(request_id, "stream", started_at)
            yield StreamEvent.completed()
        except asyncio.CancelledError:
            logger.info(
                "LLM request cancelled provider=%s model=%s mode=stream request_id=%s "
                "elapsed_seconds=%.3f",
                self.provider_name,
                self.model_name,
                request_id,
                monotonic() - started_at,
            )
            raise
        except Exception as error:
            mapped_error = self._map_error(error)
            self._log_failed(request_id, "stream", started_at, mapped_error)
            yield StreamEvent.failed(mapped_error)

    async def close(self) -> None:
        await self._client.close()

    def _build_request(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None,
        tools: Sequence[ToolDefinition],
        request_context: RequestContext | None,
    ) -> dict[str, object]:
        request_parameters: dict[str, object] = dict(self._config.extra_params)
        if self._config.temperature is not None:
            request_parameters["temperature"] = self._config.temperature
        if parameters is not None:
            request_parameters.update(parameters)
        protected = sorted(_PROTECTED_PARAMETERS.intersection(request_parameters))
        if protected:
            fields = ", ".join(protected)
            raise LLMResponseError(f"Request parameters cannot override protected fields: {fields}")
        if any(message.tool_calls or message.tool_results for message in messages):
            request_parameters.pop("thinking", None)

        system_text, converted_messages = self._build_messages(messages)
        if request_context is not None and request_context.reminder:
            self._append_reminder(converted_messages, request_context.reminder)
        request_parameters.update(
            {
                "model": self.model_name,
                "max_tokens": self._config.max_tokens or 4_096,
                "messages": converted_messages,
            }
        )
        if request_context is None:
            if system_text:
                request_parameters["system"] = system_text
        else:
            system_blocks: list[dict[str, object]] = []
            if system_text:
                system_blocks.append(
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                )
            if request_context.environment:
                system_blocks.append({"type": "text", "text": request_context.environment})
            if system_blocks:
                request_parameters["system"] = system_blocks
        if tools:
            request_parameters["tools"] = [self._build_tool_definition(tool) for tool in tools]
        return request_parameters

    @staticmethod
    def _build_messages(
        messages: Sequence[Message],
    ) -> tuple[str, list[dict[str, object]]]:
        system_parts: list[str] = []
        converted: list[dict[str, object]] = []
        for message in messages:
            if message.role is MessageRole.SYSTEM:
                system_parts.append(message.content)
                continue
            if message.role is MessageRole.TOOL:
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": result.tool_call_id,
                                "content": result.content,
                                "is_error": result.is_error,
                            }
                            for result in message.tool_results
                        ],
                    }
                )
                continue
            if message.tool_calls:
                blocks: list[dict[str, object]] = []
                if message.content:
                    blocks.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.name,
                            "input": _parse_json_object(call.arguments_json),
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
                continue
            converted.append({"role": message.role.value, "content": message.content})
        return "\n\n".join(system_parts), converted

    @staticmethod
    def _append_reminder(messages: list[dict[str, object]], reminder: str) -> None:
        reminder_block = {"type": "text", "text": reminder}
        if not messages or messages[-1].get("role") != "user":
            messages.append({"role": "user", "content": [reminder_block]})
            return

        content = messages[-1].get("content")
        if isinstance(content, str):
            messages[-1]["content"] = [
                {"type": "text", "text": content},
                reminder_block,
            ]
        elif isinstance(content, list):
            messages[-1]["content"] = [*content, reminder_block]
        else:
            raise LLMResponseError()

    @staticmethod
    def _build_tool_definition(tool: ToolDefinition) -> dict[str, object]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": dict(tool.input_schema),
        }

    def _parse_chat_result(self, response: object) -> ChatResult:
        text = self._extract_text(response)
        calls = self._extract_tool_calls(response)
        if not text.strip() and not calls:
            raise LLMResponseError()
        model = getattr(response, "model", self.model_name)
        if not isinstance(model, str):
            model = self.model_name
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason is not None and not isinstance(stop_reason, str):
            stop_reason = str(stop_reason)
        return ChatResult(
            message=Message(MessageRole.ASSISTANT, text, tool_calls=calls),
            model=model,
            usage=self._parse_usage(getattr(response, "usage", None)),
            finish_reason=stop_reason,
        )

    @staticmethod
    def _extract_text(response: object) -> str:
        blocks = getattr(response, "content", None)
        if blocks is None:
            raise LLMResponseError()
        try:
            return "".join(
                block.text for block in cast(Any, blocks) if getattr(block, "type", None) == "text"
            )
        except (AttributeError, TypeError) as error:
            raise LLMResponseError() from error

    @staticmethod
    def _extract_tool_calls(response: object) -> tuple[ToolCall, ...]:
        blocks = getattr(response, "content", None)
        if blocks is None:
            raise LLMResponseError()
        calls: list[ToolCall] = []
        try:
            for block in cast(Any, blocks):
                if getattr(block, "type", None) != "tool_use":
                    continue
                input_value = block.input
                if not isinstance(input_value, Mapping):
                    raise LLMResponseError()
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments_json=json.dumps(
                            dict(input_value),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    )
                )
        except (AttributeError, TypeError, ValueError) as error:
            raise LLMResponseError() from error
        return tuple(calls)

    @staticmethod
    def _parse_usage(usage: object | None) -> TokenUsage | None:
        if usage is None:
            return None
        try:
            input_tokens = cast(Any, usage).input_tokens
            output_tokens = cast(Any, usage).output_tokens
            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise LLMResponseError() from error

    @staticmethod
    def _map_error(error: Exception) -> LLMError:
        if isinstance(error, LLMError):
            return error
        if _is_prompt_too_long(error):
            mapped = PromptTooLongError()
            mapped.__cause__ = error
            mapped.__suppress_context__ = True
            return mapped
        if isinstance(error, anthropic.APITimeoutError):
            return LLMTimeoutError()
        if isinstance(error, anthropic.APIConnectionError):
            return LLMNetworkError()
        if isinstance(error, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
            return LLMAuthenticationError()
        if isinstance(error, anthropic.NotFoundError):
            return LLMModelNotFoundError()
        if isinstance(error, anthropic.RateLimitError):
            return LLMRateLimitError()
        if isinstance(error, anthropic.InternalServerError):
            return LLMServiceError()
        if isinstance(error, anthropic.APIStatusError):
            return AnthropicProvider._map_status_error(error.status_code)
        if isinstance(error, anthropic.APIError):
            return LLMResponseError()
        return LLMResponseError()

    @staticmethod
    def _map_status_error(status_code: int) -> LLMError:
        if status_code in {401, 403}:
            return LLMAuthenticationError()
        if status_code == 404:
            return LLMModelNotFoundError()
        if status_code in {408, 504}:
            return LLMTimeoutError()
        if status_code == 429:
            return LLMRateLimitError()
        if status_code >= 500:
            return LLMServiceError()
        return LLMResponseError()

    def _log_started(self, request_id: str, *, mode: str) -> None:
        logger.debug(
            "LLM request started provider=%s model=%s mode=%s request_id=%s",
            self.provider_name,
            self.model_name,
            mode,
            request_id,
        )

    def _log_completed(self, request_id: str, mode: str, started_at: float) -> None:
        logger.info(
            "LLM request completed provider=%s model=%s mode=%s request_id=%s elapsed_seconds=%.3f",
            self.provider_name,
            self.model_name,
            mode,
            request_id,
            monotonic() - started_at,
        )

    def _log_failed(self, request_id: str, mode: str, started_at: float, error: LLMError) -> None:
        logger.error(
            "LLM request failed provider=%s model=%s mode=%s request_id=%s "
            "error=%s elapsed_seconds=%.3f",
            self.provider_name,
            self.model_name,
            mode,
            request_id,
            type(error).__name__,
            monotonic() - started_at,
        )


def _parse_json_object(arguments_json: str) -> dict[str, object]:
    try:
        value = json.loads(arguments_json)
    except json.JSONDecodeError as error:
        raise LLMResponseError() from error
    if not isinstance(value, dict):
        raise LLMResponseError()
    return cast(dict[str, object], value)


def _is_prompt_too_long(error: Exception) -> bool:
    body = getattr(error, "body", None)
    if _contains_ptl_code(body):
        return True
    rendered = str(error).casefold()
    return any(marker in rendered for marker in _PTL_TEXT_MARKERS)


def _contains_ptl_code(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"code", "type"} and isinstance(nested, str):
                if nested.casefold() in _PTL_CODES:
                    return True
            if isinstance(nested, Mapping) and _contains_ptl_code(nested):
                return True
    return False
