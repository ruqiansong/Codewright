"""Configuration models and YAML loading for Codewright."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

type ProviderProtocol = Literal["openai-compatible", "anthropic"]
type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

logger = logging.getLogger(__name__)

DEFAULT_ANTHROPIC_CONTEXT_WINDOW = 200_000
DEFAULT_OPENAI_CONTEXT_WINDOW = 128_000


class ConfigError(ValueError):
    """Raised when the Codewright configuration cannot be loaded or validated."""


class ProviderConfig(BaseModel):
    """Validated configuration for one LLM provider."""

    model_config = ConfigDict(frozen=True, extra="allow")

    name: str
    protocol: ProviderProtocol
    api_key: SecretStr
    base_url: HttpUrl
    model: str
    stream: bool = True
    timeout_seconds: float = Field(default=60.0, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    context_window: int = Field(default=0, ge=0)
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "model")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Reject empty identifiers and normalize surrounding whitespace."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        """Reject empty credentials without exposing their value."""
        normalized = value.get_secret_value().strip()
        if not normalized:
            raise ValueError("must not be empty")
        return SecretStr(normalized)


class Config(BaseModel):
    """Validated top-level Codewright configuration."""

    model_config = ConfigDict(frozen=True, extra="allow")

    providers: tuple[ProviderConfig, ...]
    default_provider: str | None = None
    system_prompt: str | None = None
    log_level: LogLevel = "INFO"
    enable_subagent_background: bool = True
    enable_coordinator_mode: bool = False
    enable_fork_teammate: bool = False

    @field_validator(
        "enable_subagent_background",
        "enable_coordinator_mode",
        "enable_fork_teammate",
        mode="before",
    )
    @classmethod
    def validate_enable_subagent_background(cls, value: object) -> object:
        """Reject YAML integers and strings instead of coercing them to booleans."""
        if not isinstance(value, bool):
            raise ValueError("must be a boolean")
        return value

    @field_validator("providers")
    @classmethod
    def validate_providers_not_empty(
        cls, providers: tuple[ProviderConfig, ...]
    ) -> tuple[ProviderConfig, ...]:
        """Require at least one configured provider."""
        if not providers:
            raise ValueError("must contain at least one provider")
        return providers

    @field_validator("default_provider")
    @classmethod
    def normalize_default_provider(cls, value: str | None) -> str | None:
        """Normalize an optional default provider name."""
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @model_validator(mode="after")
    def validate_provider_names(self) -> Self:
        """Ensure provider names are unique and the default exists."""
        names = [provider.name for provider in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")
        if self.default_provider is not None and self.default_provider not in names:
            raise ValueError("default_provider must reference a configured provider")
        return self


def effective_context_window(provider: ProviderConfig) -> int:
    """Return an explicit context window or the provider protocol default."""
    if not isinstance(provider, ProviderConfig):
        raise TypeError("provider must be a ProviderConfig")
    if provider.context_window > 0:
        return provider.context_window
    if provider.protocol == "anthropic":
        return DEFAULT_ANTHROPIC_CONTEXT_WINDOW
    return DEFAULT_OPENAI_CONTEXT_WINDOW


def load(path: str | Path) -> Config:
    """Load and validate a Codewright YAML configuration file."""
    config_path = Path(path)
    logger.debug("Loading configuration path=%s", config_path)

    try:
        with config_path.open(encoding="utf-8") as stream:
            raw_config = yaml.safe_load(stream)
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found: {config_path}") from None
    except OSError:
        raise ConfigError(f"Unable to read configuration file: {config_path}") from None
    except yaml.YAMLError:
        raise ConfigError(f"Invalid YAML in configuration file: {config_path}") from None

    if not isinstance(raw_config, Mapping):
        raise ConfigError("Configuration root must be a YAML mapping.")

    try:
        config = Config.model_validate(raw_config)
    except ValidationError as error:
        details = _format_validation_errors(error)
        raise ConfigError(f"Invalid configuration: {details}") from None

    logger.info("Configuration loaded providers=%d", len(config.providers))
    return config


def select_provider(config: Config, name: str | None = None) -> ProviderConfig:
    """Select a provider explicitly, by default, or when only one exists."""
    selected_name = name or config.default_provider

    if selected_name is not None:
        for provider in config.providers:
            if provider.name == selected_name:
                logger.debug("Provider selected name=%s", provider.name)
                return provider
        raise ConfigError(f"Provider is not configured: {selected_name}")

    if len(config.providers) == 1:
        logger.debug("Only configured provider selected name=%s", config.providers[0].name)
        return config.providers[0]

    raise ConfigError("Multiple providers are configured; choose a provider explicitly.")


def _format_location(location: tuple[int | str, ...]) -> str:
    """Format a Pydantic error location without including rejected input."""
    return ".".join(str(part) for part in location)


def _format_validation_errors(error: ValidationError) -> str:
    """Return actionable validation details without rejected values or context."""
    details = {
        f"{_format_location(item['loc']) or 'configuration'}: {item['msg']}"
        for item in error.errors(include_url=False, include_context=False, include_input=False)
    }
    return "; ".join(sorted(details))
