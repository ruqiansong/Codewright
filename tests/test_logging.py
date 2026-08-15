"""Logging safety, redaction, and failure-isolation tests."""

import logging
from io import StringIO
from pathlib import Path

import pytest

from codewright import cli
from codewright.config import ConfigError, load
from codewright.utils.logging import configure_logging, redact_sensitive

SYNTHETIC_SECRET = "logging-test-key-not-a-real-secret"
SENSITIVE_USER_TEXT = "private-user-message-314159"


class BrokenStream(StringIO):
    """A stream that simulates an unavailable log destination."""

    def write(self, value: str) -> int:
        del value
        raise OSError("log destination unavailable")


def test_logging_includes_diagnostics_and_redacts_credentials() -> None:
    stream = StringIO()
    configure_logging("DEBUG", stream=stream, sensitive_values=(SYNTHETIC_SECRET,))
    logger = logging.getLogger("codewright.test")

    logger.debug(
        "request provider=deepseek model=deepseek-chat api_key=%s Authorization: Bearer %s",
        SYNTHETIC_SECRET,
        SYNTHETIC_SECRET,
    )

    output = stream.getvalue()
    assert "DEBUG codewright.test" in output
    assert "provider=deepseek" in output
    assert "model=deepseek-chat" in output
    assert SYNTHETIC_SECRET not in output
    assert "Authorization: Bearer [REDACTED]" in output
    assert "api_key=[REDACTED]" in output


def test_formatted_exception_is_redacted() -> None:
    stream = StringIO()
    configure_logging("ERROR", stream=stream, sensitive_values=(SYNTHETIC_SECRET,))
    logger = logging.getLogger("codewright.test")

    try:
        raise RuntimeError(f"unsafe exception {SYNTHETIC_SECRET}")
    except RuntimeError:
        logger.exception("request failed")

    output = stream.getvalue()
    assert "RuntimeError" in output
    assert SYNTHETIC_SECRET not in output


def test_error_level_suppresses_debug_events() -> None:
    stream = StringIO()
    configure_logging("ERROR", stream=stream)
    logger = logging.getLogger("codewright.test")

    logger.debug("hidden-debug-event")
    logger.error("visible-error-event")

    output = stream.getvalue()
    assert "hidden-debug-event" not in output
    assert "visible-error-event" in output


def test_broken_log_destination_does_not_raise() -> None:
    configure_logging("INFO", stream=BrokenStream())

    logging.getLogger("codewright.test").info("safe diagnostic")


def test_redaction_handles_unregistered_authorization_and_api_key_forms() -> None:
    rendered = redact_sensitive("Authorization=Bearer unregistered-token api_key: unregistered-key")

    assert "unregistered-token" not in rendered
    assert "unregistered-key" not in rendered
    assert rendered.count("[REDACTED]") == 2


def test_invalid_config_does_not_log_or_raise_secret(tmp_path: Path) -> None:
    stream = StringIO()
    configure_logging("DEBUG", stream=stream, sensitive_values=(SYNTHETIC_SECRET,))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "providers:\n"
        "  - name: deepseek\n"
        "    protocol: openai-compatible\n"
        f"    api_key: {SYNTHETIC_SECRET}\n"
        "    base_url: invalid\n"
        "    model: deepseek-chat\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as captured:
        load(config_path)

    assert SYNTHETIC_SECRET not in str(captured.value)
    assert SYNTHETIC_SECRET not in stream.getvalue()


def test_cli_debug_logs_do_not_include_prompt_or_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = StringIO()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "providers:\n"
        "  - name: deepseek\n"
        "    protocol: openai-compatible\n"
        f"    api_key: {SYNTHETIC_SECRET}\n"
        "    base_url: https://api.deepseek.com\n"
        "    model: deepseek-chat\n"
        f"system_prompt: {SENSITIVE_USER_TEXT}\n",
        encoding="utf-8",
    )

    original_configure = configure_logging

    def configure_for_test(
        level: str,
        *,
        sensitive_values: object = (),
    ) -> logging.Logger:
        return original_configure(
            level,
            stream=stream,
            sensitive_values=sensitive_values,  # type: ignore[arg-type]
        )

    class NoOpApp:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def run_async(self) -> None:
            return None

    class NoOpProvider:
        provider_name = "deepseek"
        model_name = "deepseek-chat"

        async def chat(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def stream_chat(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    monkeypatch.setattr(cli, "configure_logging", configure_for_test)
    monkeypatch.setattr(cli, "create_provider", lambda _: NoOpProvider())
    monkeypatch.setattr(cli, "CodewrightApp", NoOpApp)
    monkeypatch.setattr(cli.mcp_client, "load_config", lambda _: cli.mcp_client.Config({}))

    assert cli.main(["--config", str(config_path), "--log-level", "DEBUG"]) == 0

    output = stream.getvalue()
    assert "Starting Codewright" in output
    assert SYNTHETIC_SECRET not in output
    assert SENSITIVE_USER_TEXT not in output
