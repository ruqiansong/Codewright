"""Run an offline permission smoke check without invoking a real shell."""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from codewright.agent import Agent, Event
from codewright.conversation import Conversation
from codewright.llm import (
    ChatResult,
    Message,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolCall,
    ToolDefinition,
)
from codewright.permission import Engine, Mode, Outcome
from codewright.permission.rule import RuleSet
from codewright.tool import Registry, Result


class FakeProvider:
    """Request one scripted tool and then provide a final answer."""

    provider_name = "offline"
    model_name = "permission-smoke"

    def __init__(self, call: ToolCall) -> None:
        self._call = call
        self._requests = 0

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del messages, parameters, tools, request_context
        raise AssertionError("permission smoke uses streaming")

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del messages, parameters, tools, request_context
        if self._requests == 0:
            yield StreamEvent.tool_calls_ready((self._call,))
        else:
            yield StreamEvent.delta("done")
        self._requests += 1
        yield StreamEvent.completed()


@dataclass(slots=True)
class SpyBashTool:
    """Record execution without creating a subprocess."""

    name: str = "bash"
    description: str = "Offline spy command tool."
    parameters: Mapping[str, object] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    read_only: bool = False
    calls: list[str] = field(default_factory=list)

    async def execute(self, arguments_json: str) -> Result:
        self.calls.append(arguments_json)
        return Result("spy execution complete")


def engine(root: Path) -> Engine:
    return Engine(
        root=root.resolve(),
        user=RuleSet(),
        project=RuleSet(),
        local=RuleSet(),
        local_path=root / ".codewright" / "settings.local.yaml",
        default_mode=Mode.DEFAULT,
    )


async def run_scenario(
    root: Path,
    command: str,
    mode: Mode,
    *,
    approval: Outcome | None = None,
) -> tuple[list[Event], SpyBashTool]:
    call = ToolCall("call-1", "bash", json.dumps({"command": command}))
    provider = FakeProvider(call)
    spy = SpyBashTool()
    registry = Registry()
    registry.register(spy)
    conversation = Conversation("You are Codewright.")
    conversation.add_user("Run the offline permission scenario.")
    events: list[Event] = []
    async for event in Agent(provider, registry, engine(root)).run(
        conversation,
        mode=mode,
    ):
        events.append(event)
        if event.approval is not None:
            if approval is None:
                raise AssertionError("unexpected approval request")
            event.approval.respond.set_result(approval)
    return events, spy


async def main() -> None:
    root = Path.cwd()
    default_events, default_spy = await run_scenario(
        root,
        "echo permission-smoke",
        Mode.DEFAULT,
        approval=Outcome.DENY_ONCE,
    )
    assert any(event.approval is not None for event in default_events)
    assert default_spy.calls == []

    bypass_events, bypass_spy = await run_scenario(
        root,
        "echo permission-smoke",
        Mode.BYPASS,
    )
    assert not any(event.approval is not None for event in bypass_events)
    assert len(bypass_spy.calls) == 1

    dangerous_events, dangerous_spy = await run_scenario(
        root,
        "rm -rf /",
        Mode.BYPASS,
    )
    assert not any(event.approval is not None for event in dangerous_events)
    assert dangerous_spy.calls == []
    print("permission smoke: ok")


if __name__ == "__main__":
    asyncio.run(main())
