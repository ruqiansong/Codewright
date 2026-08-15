"""Run a low-output, real-API prompt-cache smoke check."""

import argparse
import asyncio
from pathlib import Path

from codewright.config import load, select_provider
from codewright.llm import Message, MessageRole, RequestContext, TokenUsage
from codewright.llm.factory import create_provider
from codewright.llm.provider import Provider
from codewright.prompt import build_system_prompt
from codewright.tool import new_default_registry


def parse_args() -> argparse.Namespace:
    """Parse smoke-check arguments without accepting credentials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".codewright/config.yaml"),
        help="configuration file containing the provider credentials",
    )
    parser.add_argument("--provider", help="configured provider name", default=None)
    parser.add_argument(
        "--delay",
        type=float,
        default=3.0,
        help="seconds to wait for best-effort cache persistence",
    )
    return parser.parse_args()


async def request_usage(provider: Provider) -> TokenUsage | None:
    """Run one stable streaming request and retain only its usage event."""
    messages = (
        Message(MessageRole.SYSTEM, build_system_prompt()),
        Message(MessageRole.USER, "Reply with exactly the word OK."),
    )
    context = RequestContext(
        environment="Environment:\nCache verification smoke check.",
    )
    tools = new_default_registry(working_directory=Path.cwd()).definitions()
    usage = None
    async for event in provider.stream_chat(
        messages,
        tools=tools,
        request_context=context,
    ):
        if event.error is not None:
            raise event.error
        if event.usage is not None:
            usage = event.usage
    return usage


def print_usage(request_number: int, usage: TokenUsage | None) -> None:
    """Print bounded accounting only; never print prompts, responses, or keys."""
    if usage is None:
        print(f"request={request_number} usage=unavailable")
        return
    print(
        f"request={request_number} input={usage.input_tokens} "
        f"output={usage.output_tokens} cache_write={usage.cache_write_tokens} "
        f"cache_read={usage.cache_read_tokens}"
    )


async def run() -> None:
    """Execute two identical requests against the selected provider."""
    args = parse_args()
    if args.delay < 0:
        raise ValueError("--delay must be non-negative")
    provider = create_provider(select_provider(load(args.config), args.provider))
    try:
        first_usage = await request_usage(provider)
        print_usage(1, first_usage)
        await asyncio.sleep(args.delay)
        second_usage = await request_usage(provider)
        print_usage(2, second_usage)
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            await close()


if __name__ == "__main__":
    asyncio.run(run())
