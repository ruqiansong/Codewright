"""Configuration-driven construction of language model providers."""

from codewright.config import ConfigError, ProviderConfig
from codewright.llm.anthropic_provider import AnthropicProvider
from codewright.llm.deepseek import DeepSeekProvider
from codewright.llm.provider import Provider


def create_provider(config: ProviderConfig) -> Provider:
    """Create the provider selected by the validated protocol configuration."""
    if config.protocol == "openai-compatible":
        return DeepSeekProvider(config)
    if config.protocol == "anthropic":
        return AnthropicProvider(config)
    raise ConfigError(f"Unsupported provider protocol: {config.protocol}")


__all__ = ["create_provider"]
