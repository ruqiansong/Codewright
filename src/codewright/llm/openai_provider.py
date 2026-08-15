"""OpenAI Chat Completions compatible provider with tool-call support."""

import asyncio
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from secrets import token_hex
from time import monotonic
from typing import Any, Protocol, cast

import openai
from openai import AsyncOpenAI

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

_PROTECTED_PARAMETERS = frozenset({"api_key", "base_url", "messages", "model", "stream", "tools"})
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


class _CompletionsResource(Protocol):
    async def create(self, **parameters: object) -> Any:
        """Create a chat completion or stream."""
        ...


class _ChatResource(Protocol):
    completions: _CompletionsResource


class _AsyncClient(Protocol):
    chat: _ChatResource

    async def close(self) -> None:
        """Close client resources."""
        ...


class OpenAICompatibleProvider:
    """Translate the public Provider contract to OpenAI Chat Completions."""

    def __init__(self, config: ProviderConfig, *, client: object | None = None) -> None:
        self._config = config
        if client is None:
            client = AsyncOpenAI(
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
        """Return one complete OpenAI-compatible response."""
        request_id = token_hex(6)
        started_at = monotonic()
        self._log_started(request_id, mode="chat")
        try:
            response = await self._client.chat.completions.create(
                **self._build_request(
                    messages,
                    stream=False,
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
        """Yield text, complete tool calls, and one completion event."""
        request_id = token_hex(6)
        started_at = monotonic()
        self._log_started(request_id, mode="stream")
        call_parts: dict[int, dict[str, str]] = {}
        stream_usage: TokenUsage | None = None
        try:
            stream = await self._client.chat.completions.create(
                **self._build_request(
                    messages,
                    stream=True,
                    parameters=parameters,
                    tools=tools,
                    request_context=request_context,
                )
            )
            async for chunk in stream:
                usage = self._parse_usage(getattr(chunk, "usage", None))
                if usage is not None:
                    stream_usage = usage
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if content not in (None, ""):
                    if not isinstance(content, str):
                        raise LLMResponseError()
                    yield StreamEvent.delta(content)
                self._accumulate_tool_calls(getattr(delta, "tool_calls", None), call_parts)

            if call_parts:
                yield StreamEvent.tool_calls_ready(self._complete_tool_calls(call_parts))
            if stream_usage is not None:
                yield StreamEvent.usage_report(stream_usage)
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
        stream: bool,
        parameters: RequestParameters | None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None,
    ) -> dict[str, object]:
        request_parameters: dict[str, object] = dict(self._config.extra_params)
        if self._config.temperature is not None:
            request_parameters["temperature"] = self._config.temperature
        if self._config.max_tokens is not None:
            request_parameters["max_tokens"] = self._config.max_tokens
        if parameters is not None:
            request_parameters.update(parameters)

        protected = sorted(_PROTECTED_PARAMETERS.intersection(request_parameters))
        if protected:
            fields = ", ".join(protected)
            raise LLMResponseError(f"Request parameters cannot override protected fields: {fields}")

        request_parameters.update(
            {
                "model": self.model_name,
                "messages": self._build_messages(messages, request_context=request_context),
                "stream": stream,
                "timeout": self._config.timeout_seconds,
            }
        )
        if stream:
            request_parameters["stream_options"] = {"include_usage": True}
        if tools:
            request_parameters["tools"] = [self._build_tool_definition(tool) for tool in tools]
        return request_parameters

    @staticmethod
    def _build_messages(
        messages: Sequence[Message],
        *,
        request_context: RequestContext | None,
    ) -> list[dict[str, object]]:
        converted: list[dict[str, object]] = []
        system_parts: list[str] = []
        for message in messages:
            if request_context is not None and message.role is MessageRole.SYSTEM:
                system_parts.append(message.content)
                continue
            if message.role is MessageRole.TOOL:
                converted.extend(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content,
                    }
                    for result in message.tool_results
                )
                continue
            converted_message: dict[str, object] = {
                "role": message.role.value,
                "content": message.content,
            }
            if message.tool_calls:
                converted_message["content"] = message.content or None
                converted_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": call.arguments_json,
                        },
                    }
                    for call in message.tool_calls
                ]
            converted.append(converted_message)
        if request_context is not None:
            system_text = "\n\n".join(system_parts)
            if request_context.environment:
                system_text = "\n\n".join(
                    part for part in (system_text, request_context.environment) if part
                )
            if system_text:
                converted.insert(0, {"role": "system", "content": system_text})
            if request_context.reminder:
                converted.append({"role": "user", "content": request_context.reminder})
        return converted

    @staticmethod
    def _build_tool_definition(tool: ToolDefinition) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.input_schema),
            },
        }

    def _parse_chat_result(self, response: object) -> ChatResult:
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMResponseError()
        choice = choices[0]
        raw_message = getattr(choice, "message", None)
        content = getattr(raw_message, "content", None)
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise LLMResponseError()
        calls = self._parse_complete_calls(getattr(raw_message, "tool_calls", None))
        if not content.strip() and not calls:
            raise LLMResponseError()

        response_model = getattr(response, "model", None)
        model = response_model if isinstance(response_model, str) else self.model_name
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)
        return ChatResult(
            message=Message(MessageRole.ASSISTANT, content, tool_calls=calls),
            model=model,
            usage=self._parse_usage(getattr(response, "usage", None)),
            finish_reason=finish_reason,
        )

    @staticmethod
    def _parse_complete_calls(raw_calls: object) -> tuple[ToolCall, ...]:
        if raw_calls is None:
            return ()
        try:
            return tuple(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments_json=call.function.arguments or "{}",
                )
                for call in cast(Any, raw_calls)
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise LLMResponseError() from error

    @staticmethod
    def _accumulate_tool_calls(
        raw_calls: object,
        call_parts: dict[int, dict[str, str]],
    ) -> None:
        if raw_calls is None:
            return
        try:
            for raw_call in cast(Any, raw_calls):
                index = raw_call.index
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    raise LLMResponseError()
                slot = call_parts.setdefault(index, {"id": "", "name": "", "arguments": ""})
                call_id = getattr(raw_call, "id", None)
                function = getattr(raw_call, "function", None)
                name = getattr(function, "name", None)
                arguments = getattr(function, "arguments", None)
                if call_id is not None:
                    if not isinstance(call_id, str):
                        raise LLMResponseError()
                    slot["id"] += call_id
                if name is not None:
                    if not isinstance(name, str):
                        raise LLMResponseError()
                    slot["name"] += name
                if arguments is not None:
                    if not isinstance(arguments, str):
                        raise LLMResponseError()
                    slot["arguments"] += arguments
        except TypeError as error:
            raise LLMResponseError() from error

    @staticmethod
    def _complete_tool_calls(call_parts: dict[int, dict[str, str]]) -> tuple[ToolCall, ...]:
        try:
            return tuple(
                ToolCall(
                    id=parts["id"],
                    name=parts["name"],
                    arguments_json=parts["arguments"] or "{}",
                )
                for _, parts in sorted(call_parts.items())
            )
        except (TypeError, ValueError) as error:
            raise LLMResponseError() from error

    @staticmethod
    def _parse_usage(usage: object | None) -> TokenUsage | None:
        if usage is None:
            return None
        usage_data = cast(Any, usage)
        try:
            details = getattr(usage_data, "prompt_tokens_details", None)
            if isinstance(details, Mapping):
                cache_read_tokens = details.get("cached_tokens", 0) or 0
            else:
                cache_read_tokens = getattr(details, "cached_tokens", 0) or 0
            if not cache_read_tokens:
                cache_read_tokens = getattr(usage_data, "prompt_cache_hit_tokens", 0) or 0
            return TokenUsage(
                input_tokens=usage_data.prompt_tokens,
                output_tokens=usage_data.completion_tokens,
                total_tokens=usage_data.total_tokens,
                cache_read_tokens=cache_read_tokens,
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
        if isinstance(error, openai.APITimeoutError):
            return LLMTimeoutError()
        if isinstance(error, openai.APIConnectionError):
            return LLMNetworkError()
        if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return LLMAuthenticationError()
        if isinstance(error, openai.NotFoundError):
            return LLMModelNotFoundError()
        if isinstance(error, openai.RateLimitError):
            return LLMRateLimitError()
        if isinstance(error, openai.InternalServerError):
            return LLMServiceError()
        if isinstance(error, openai.APIStatusError):
            return OpenAICompatibleProvider._map_status_error(error.status_code)
        if isinstance(error, openai.APIError):
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
