"""Vendor-neutral errors raised by LLM providers."""


class LLMError(Exception):
    """Base error containing only information safe to show to a user."""

    default_safe_message = "The language model request failed."
    default_retryable = False

    def __init__(
        self,
        safe_message: str | None = None,
        *,
        code: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        resolved_message = safe_message or self.default_safe_message
        if not resolved_message.strip():
            raise ValueError("safe_message must not be empty")

        super().__init__(resolved_message)
        self.safe_message = resolved_message
        self.code = code
        self.retryable = self.default_retryable if retryable is None else retryable


class LLMAuthenticationError(LLMError):
    """The provider rejected the configured credentials."""

    default_safe_message = "Authentication failed. Check the configured API key."


class LLMNetworkError(LLMError):
    """The provider could not be reached over the network."""

    default_safe_message = "Unable to reach the language model service. Check the network."
    default_retryable = True


class LLMTimeoutError(LLMError):
    """The provider request exceeded its allowed duration."""

    default_safe_message = "The language model request timed out."
    default_retryable = True


class LLMModelNotFoundError(LLMError):
    """The configured model does not exist or is not accessible."""

    default_safe_message = "The configured model was not found or is not accessible."


class LLMRateLimitError(LLMError):
    """The provider rejected the request because of a rate or quota limit."""

    default_safe_message = "The language model service is rate limited. Try again later."
    default_retryable = True


class LLMServiceError(LLMError):
    """The provider reported an internal service failure."""

    default_safe_message = "The language model service is temporarily unavailable."
    default_retryable = True


class LLMResponseError(LLMError):
    """The provider returned an invalid or incomplete response."""

    default_safe_message = "The language model returned an invalid or incomplete response."


class PromptTooLongError(LLMError):
    """The provider rejected a request that exceeded its context window."""

    default_safe_message = "The conversation exceeds the model context window."
