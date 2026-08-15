"""Public subagent definition and discovery API."""

from codewright.subagent.catalog import AGENTS_DIR, Catalog, builtin_definitions, load_catalog
from codewright.subagent.definition import DEFAULT_MAX_TURNS, Definition, Source
from codewright.subagent.parser import (
    AGENT_NAME_REGEX,
    DefinitionParseError,
    parse_definition,
    parse_file,
    parse_frontmatter_and_body,
)

__all__ = [
    "AGENTS_DIR",
    "AGENT_NAME_REGEX",
    "DEFAULT_MAX_TURNS",
    "Catalog",
    "Definition",
    "DefinitionParseError",
    "Source",
    "builtin_definitions",
    "load_catalog",
    "parse_definition",
    "parse_file",
    "parse_frontmatter_and_body",
]
