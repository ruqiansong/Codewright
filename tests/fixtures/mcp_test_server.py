"""Minimal MCP server used only by Codewright integration tests."""

import argparse
import json
import os
import sys

import mcp.types as mtypes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def tools() -> list[mtypes.Tool]:
    read_only = mtypes.ToolAnnotations(readOnlyHint=True)
    return [
        mtypes.Tool(
            name="echo",
            description="Return the supplied text.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            annotations=read_only,
        ),
        mtypes.Tool(
            name="process_info",
            description="Return the process id and one environment value.",
            inputSchema={
                "type": "object",
                "properties": {"variable": {"type": "string"}},
                "required": ["variable"],
            },
            annotations=read_only,
        ),
        mtypes.Tool(
            name="request_header",
            description="Return one request header when using HTTP.",
            inputSchema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            annotations=read_only,
        ),
    ]


def run_stdio() -> None:
    """Serve the small JSON-RPC subset needed by client integration tests."""
    for line in sys.stdin:
        request = json.loads(line)
        request_id = request.get("id")
        method = request.get("method")
        if request_id is None:
            continue
        if method == "initialize":
            result: object = {
                "protocolVersion": request.get("params", {}).get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "codewright-test-server", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [tool.model_dump(by_alias=True, exclude_none=True) for tool in tools()]
            }
        elif method == "tools/call":
            params = request.get("params", {})
            arguments = params.get("arguments") or {}
            name = params.get("name")
            if name == "echo":
                text = str(arguments.get("text", ""))
            elif name == "process_info":
                variable = str(arguments.get("variable", ""))
                text = f"{os.getpid()}|{os.environ.get(variable, '')}"
            elif name == "request_header":
                text = ""
            else:
                text = f"unknown tool: {name}"
            result = {"content": [{"type": "text", "text": text}], "isError": False}
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not found"},
            }
            print(json.dumps(response), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)


def run_http(host: str, port: int) -> None:
    from mcp.server.fastmcp import Context, FastMCP

    server = FastMCP(
        "Codewright test server",
        host=host,
        port=port,
        json_response=True,
        stateless_http=False,
        log_level="ERROR",
    )

    @server.tool(annotations=mtypes.ToolAnnotations(readOnlyHint=True))
    def echo(text: str) -> str:
        return text

    @server.tool(annotations=mtypes.ToolAnnotations(readOnlyHint=True))
    def request_header(name: str, ctx: Context) -> str:
        request = ctx.request_context.request
        headers = getattr(request, "headers", {}) if request is not None else {}
        return str(headers.get(name, ""))

    server.run(transport="streamable-http")


def main() -> None:
    args = build_parser().parse_args()
    if args.transport == "stdio":
        run_stdio()
    else:
        run_http(args.host, args.port)


if __name__ == "__main__":
    main()
