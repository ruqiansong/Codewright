from __future__ import annotations

from pathlib import Path

from codewright.agent import CompletionResult
from codewright.conversation import Conversation
from codewright.llm import TokenUsage
from codewright.session import load_session
from codewright.team.runtime import TeammateRuntimeFactory


class Agent:
    async def run_to_completion(self, *args, **kwargs):
        del args, kwargs
        return CompletionResult("done", TokenUsage(0, 0, 0))


def test_runtime_factory_persists_initial_history_once(tmp_path: Path) -> None:
    def builder(**kwargs):
        del kwargs
        conversation = Conversation("system")
        conversation.add_user("initial")
        return Agent(), conversation, None

    runtime = TeammateRuntimeFactory(str(tmp_path), builder).create(
        initial_prompt="initial",
        description="work",
        model="test",
    )
    loaded = load_session(str(runtime.writer.path.parent))

    assert [message.content for message in loaded.messages] == ["initial"]
    runtime.conversation.add_assistant("answer")
    runtime.writer.close()
    loaded = load_session(str(runtime.writer.path.parent))
    assert [message.content for message in loaded.messages] == ["initial", "answer"]
