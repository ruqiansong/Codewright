"""Tests for Codewright configuration loading and validation."""

from collections.abc import Mapping
from pathlib import Path
from traceback import format_exception
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from codewright.config import (
    Config,
    ConfigError,
    ProviderConfig,
    effective_context_window,
    load,
    select_provider,
)

SYNTHETIC_SECRET = "test-key-not-a-real-secret"
PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / ".codewright" / "config.yaml.example"
GITIGNORE_PATH = PROJECT_ROOT / ".gitignore"


def write_config(path: Path, data: Mapping[str, Any]) -> Path:
    """Write YAML test data to a temporary configuration file."""
    path.write_text(yaml.safe_dump(dict(data)), encoding="utf-8")
    return path


def provider_data(**overrides: Any) -> dict[str, Any]:
    """Build valid provider data with optional field overrides."""
    data: dict[str, Any] = {
        "name": "deepseek",
        "protocol": "openai-compatible",
        "api_key": SYNTHETIC_SECRET,
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "context_window": 128000,
    }
    data.update(overrides)
    return data


def test_load_valid_config(valid_config_path: Path) -> None:
    config = load(valid_config_path)

    assert config.default_provider == "deepseek"
    assert config.log_level == "INFO"
    assert len(config.providers) == 1

    provider = config.providers[0]
    assert provider.name == "deepseek"
    assert provider.protocol == "openai-compatible"
    assert provider.api_key.get_secret_value() == SYNTHETIC_SECRET
    assert str(provider.base_url) == "https://api.deepseek.com/"
    assert provider.model == "deepseek-chat"
    assert provider.stream is True
    assert provider.timeout_seconds == 30
    assert provider.temperature == 0.5
    assert provider.max_tokens == 1024


def test_example_config_has_deepseek_and_anthropic_without_credentials() -> None:
    raw_config = yaml.safe_load(EXAMPLE_CONFIG_PATH.read_text(encoding="utf-8"))

    assert isinstance(raw_config, dict)
    assert raw_config["default_provider"] == "deepseek"
    assert len(raw_config["providers"]) == 2

    provider = raw_config["providers"][0]
    assert provider == {
        "name": "deepseek",
        "protocol": "openai-compatible",
        "api_key": "",
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "context_window": 128000,
        "stream": True,
        "timeout_seconds": 60,
        "temperature": 0.5,
        "max_tokens": 4096,
    }
    anthropic_provider = raw_config["providers"][1]
    assert anthropic_provider == {
        "name": "anthropic",
        "protocol": "anthropic",
        "api_key": "",
        "base_url": "https://api.anthropic.com",
        "model": "claude-sonnet-4-5",
        "context_window": 200000,
        "stream": True,
        "timeout_seconds": 60,
        "temperature": 0.5,
        "max_tokens": 4096,
    }


def test_example_config_requires_user_api_key() -> None:
    with pytest.raises(ConfigError, match="providers.0.api_key"):
        load(EXAMPLE_CONFIG_PATH)


def test_anthropic_protocol_is_validated() -> None:
    provider = ProviderConfig.model_validate(
        provider_data(
            name="anthropic",
            protocol="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-5",
        )
    )

    assert provider.protocol == "anthropic"


def test_gitignore_protects_real_config_and_keeps_example() -> None:
    rules = {
        line.strip()
        for line in GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".codewright/config.yaml" in rules
    assert ".codewright/config.*.yaml" in rules
    assert "!.codewright/config.yaml.example" in rules


def test_load_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ConfigError, match="Configuration file not found"):
        load(missing_path)


def test_load_empty_file(tmp_path: Path) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match="root must be a YAML mapping"):
        load(config_path)


def test_load_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("providers: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load(config_path)


@pytest.mark.parametrize("missing_field", ["name", "protocol", "api_key", "base_url", "model"])
def test_load_reports_missing_provider_fields_without_secret(
    tmp_path: Path, missing_field: str
) -> None:
    provider = provider_data()
    del provider[missing_field]
    config_path = write_config(tmp_path / "config.yaml", {"providers": [provider]})

    with pytest.raises(ConfigError) as captured:
        load(config_path)

    message = str(captured.value)
    assert f"providers.0.{missing_field}" in message
    assert SYNTHETIC_SECRET not in message


@pytest.mark.parametrize(
    ("overrides", "expected_field"),
    [
        ({"name": "   "}, "providers.0.name"),
        ({"protocol": "unsupported"}, "providers.0.protocol"),
        ({"api_key": "   "}, "providers.0.api_key"),
        ({"base_url": "not-a-url"}, "providers.0.base_url"),
        ({"model": ""}, "providers.0.model"),
        ({"timeout_seconds": 0}, "providers.0.timeout_seconds"),
        ({"temperature": 3}, "providers.0.temperature"),
        ({"max_tokens": 0}, "providers.0.max_tokens"),
        ({"context_window": -1}, "providers.0.context_window"),
    ],
)
def test_load_rejects_invalid_provider_values(
    tmp_path: Path, overrides: dict[str, Any], expected_field: str
) -> None:
    config_path = write_config(
        tmp_path / "config.yaml", {"providers": [provider_data(**overrides)]}
    )

    with pytest.raises(ConfigError, match=expected_field):
        load(config_path)


def test_load_requires_at_least_one_provider(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "config.yaml", {"providers": []})

    with pytest.raises(ConfigError, match="providers"):
        load(config_path)


def test_load_rejects_duplicate_provider_names(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.yaml",
        {"providers": [provider_data(), provider_data(model="deepseek-reasoner")]},
    )

    with pytest.raises(ConfigError, match="provider names must be unique"):
        load(config_path)


def test_load_rejects_unknown_default_provider(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.yaml",
        {"providers": [provider_data()], "default_provider": "unknown"},
    )

    with pytest.raises(ConfigError, match="default_provider must reference"):
        load(config_path)


def test_provider_models_are_frozen() -> None:
    provider = ProviderConfig.model_validate(provider_data())

    with pytest.raises(ValidationError):
        provider.model = "replacement"  # type: ignore[misc]


def test_secret_is_masked_in_model_output() -> None:
    provider = ProviderConfig.model_validate(provider_data())

    assert SYNTHETIC_SECRET not in repr(provider)
    assert SYNTHETIC_SECRET not in provider.model_dump_json()


def test_validation_error_does_not_expose_secret(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "config.yaml",
        {"providers": [provider_data(base_url="invalid", api_key=SYNTHETIC_SECRET)]},
    )

    with pytest.raises(ConfigError) as captured:
        load(config_path)

    rendered_error = "".join(format_exception(captured.value))
    assert SYNTHETIC_SECRET not in rendered_error


def test_select_provider_uses_explicit_name() -> None:
    config = Config.model_validate(
        {
            "providers": [
                provider_data(),
                provider_data(name="secondary", model="deepseek-reasoner"),
            ]
        }
    )

    assert select_provider(config, "secondary").model == "deepseek-reasoner"


def test_subagent_background_defaults_true_and_accepts_booleans() -> None:
    assert Config.model_validate({"providers": [provider_data()]}).enable_subagent_background
    assert not Config.model_validate(
        {"providers": [provider_data()], "enable_subagent_background": False}
    ).enable_subagent_background


@pytest.mark.parametrize("value", [0, 1, "true", "false", None])
def test_subagent_background_rejects_non_booleans(value: object) -> None:
    with pytest.raises(ValidationError, match="must be a boolean"):
        Config.model_validate({"providers": [provider_data()], "enable_subagent_background": value})


def test_select_provider_uses_default() -> None:
    config = Config.model_validate(
        {
            "providers": [
                provider_data(),
                provider_data(name="secondary", model="deepseek-reasoner"),
            ],
            "default_provider": "secondary",
        }
    )

    assert select_provider(config).name == "secondary"


def test_select_provider_uses_only_provider() -> None:
    config = Config.model_validate({"providers": [provider_data()]})

    assert select_provider(config).name == "deepseek"


def test_select_provider_requires_choice_when_ambiguous() -> None:
    config = Config.model_validate(
        {"providers": [provider_data(), provider_data(name="secondary")]}
    )

    with pytest.raises(ConfigError, match="choose a provider"):
        select_provider(config)


def test_select_provider_rejects_unknown_name() -> None:
    config = Config.model_validate({"providers": [provider_data()]})

    with pytest.raises(ConfigError, match="not configured"):
        select_provider(config, "unknown")


def test_effective_context_window_uses_protocol_defaults() -> None:
    openai_data = provider_data()
    openai_data.pop("context_window", None)
    openai_provider = ProviderConfig.model_validate(openai_data)
    anthropic_data = provider_data(
        name="anthropic",
        protocol="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-5",
    )
    anthropic_data.pop("context_window", None)
    anthropic_provider = ProviderConfig.model_validate(anthropic_data)

    assert openai_provider.context_window == 0
    assert effective_context_window(openai_provider) == 128_000
    assert effective_context_window(anthropic_provider) == 200_000


def test_effective_context_window_treats_zero_as_default() -> None:
    provider = ProviderConfig.model_validate(provider_data(context_window=0))

    assert effective_context_window(provider) == 128_000


def test_effective_context_window_uses_positive_override() -> None:
    provider = ProviderConfig.model_validate(
        provider_data(
            name="anthropic",
            protocol="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-5",
            context_window=80_000,
        )
    )

    assert effective_context_window(provider) == 80_000
