"""Safe package logging with centralized sensitive-data redaction."""

import logging
import re
import sys
from collections.abc import Iterable
from threading import RLock
from typing import TextIO

_REDACTED = "[REDACTED]"
_SECRETS: set[str] = set()
_SECRETS_LOCK = RLock()
_AUTHORIZATION_PATTERN = re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)([^\s,;]+)")
_API_KEY_PATTERN = re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)(?:[\"']?)([^\"'\s,}]+)(?:[\"']?)")


def register_secrets(values: Iterable[str]) -> None:
    """Register concrete credentials that must be removed from future logs."""
    with _SECRETS_LOCK:
        _SECRETS.update(value for value in values if len(value) >= 4)


def redact_sensitive(value: object) -> str:
    """Return text with registered secrets and common credential forms removed."""
    rendered = str(value)
    with _SECRETS_LOCK:
        secrets = sorted(_SECRETS, key=len, reverse=True)
    for secret in secrets:
        rendered = rendered.replace(secret, _REDACTED)
    rendered = _AUTHORIZATION_PATTERN.sub(rf"\1{_REDACTED}", rendered)
    return _API_KEY_PATTERN.sub(rf"\1{_REDACTED}", rendered)


class SensitiveDataFilter(logging.Filter):
    """Redact a LogRecord message before handlers or propagation use it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_sensitive(record.getMessage())
        record.args = ()
        return True


class RedactingFormatter(logging.Formatter):
    """Apply a final redaction pass, including formatted exception text."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_sensitive(super().format(record))


class SafeStreamHandler(logging.StreamHandler[TextIO]):
    """Prevent a broken logging stream from interrupting application behavior."""

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        del record


def configure_logging(
    level: str = "INFO",
    *,
    stream: TextIO | None = None,
    sensitive_values: Iterable[str] = (),
) -> logging.Logger:
    """Configure the Codewright logger and return its package logger."""
    register_secrets(sensitive_values)
    logger = logging.getLogger("codewright")
    logger.setLevel(level.upper())
    logger.propagate = False

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = SafeStreamHandler(stream or sys.stderr)
    handler.setLevel(level.upper())
    handler.addFilter(SensitiveDataFilter())
    handler.setFormatter(
        RedactingFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger
