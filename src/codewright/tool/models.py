"""Vendor-neutral results returned by Codewright tools."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Result:
    """A bounded tool execution result without a provider tool-call ID."""

    content: str
    is_error: bool = False
    error_code: str | None = None
    truncated: bool = False
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not isinstance(self.is_error, bool):
            raise TypeError("is_error must be a boolean")
        if not isinstance(self.truncated, bool):
            raise TypeError("truncated must be a boolean")
        if self.error_code is not None and not isinstance(self.error_code, str):
            raise TypeError("error_code must be a string or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        normalized_error_code = self.error_code.strip() if self.error_code is not None else None
        if self.is_error and not normalized_error_code:
            raise ValueError("an error result must have an error_code")
        if not self.is_error and normalized_error_code:
            raise ValueError("a successful result cannot have an error_code")

        object.__setattr__(self, "error_code", normalized_error_code)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
