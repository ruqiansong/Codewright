"""Public MCP client configuration and tool-adaptation primitives."""

from codewright.mcp.config import Config, ServerConfig, load_config
from codewright.mcp.manager import Manager, new_manager
from codewright.mcp.tool import CallerSession, McpTool, adapt_tool

__all__ = [
    "CallerSession",
    "Config",
    "Manager",
    "McpTool",
    "ServerConfig",
    "adapt_tool",
    "load_config",
    "new_manager",
]
