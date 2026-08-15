"""Tests for secure Markdown memory storage and LLM-backed updates."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest
import yaml

from codewright.llm import (
    ChatResult,
    Message,
    MessageRole,
    RequestContext,
    RequestParameters,
    StreamEvent,
    ToolDefinition,
)
from codewright.memory import Manager, Store, UpdateAction
from codewright.memory.store import MAX_INDEX_BYTES, MAX_INDEX_LINES


@pytest.fixture(autouse=True)
def synchronous_memory_file_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid this CI Python build's default-executor shutdown defect."""

    async def run_immediately(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("codewright.memory.manager.asyncio.to_thread", run_immediately)


class MemoryProvider:
    provider_name = "fake"
    model_name = "memory-model"

    def __init__(self, response: str | Exception) -> None:
        self.response = response
        self.requests: list[tuple[Message, ...]] = []
        self.tools: list[tuple[ToolDefinition, ...]] = []

    async def chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> ChatResult:
        del messages, parameters, tools, request_context
        raise AssertionError("memory tests use streaming")

    async def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        parameters: RequestParameters | None = None,
        tools: Sequence[ToolDefinition] = (),
        request_context: RequestContext | None = None,
    ) -> AsyncIterator[StreamEvent]:
        del parameters, request_context
        self.requests.append(tuple(messages))
        self.tools.append(tuple(tools))
        if isinstance(self.response, Exception):
            raise self.response
        midpoint = len(self.response) // 2
        if midpoint:
            yield StreamEvent.delta(self.response[:midpoint])
        yield StreamEvent.delta(self.response[midpoint:])
        yield StreamEvent.completed()


def frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text()
    boundary = text.find("\n---\n", 4)
    value = yaml.safe_load(text[4:boundary])
    assert isinstance(value, dict)
    return value


def create_action(*, slug: str = "api_style", level: str = "project") -> UpdateAction:
    return UpdateAction(
        action="create",
        level=level,
        type="project_knowledge",
        title="API style",
        slug=slug,
        content="Use resource-oriented endpoint names.",
    )


def test_manager_list_files_is_layered_sorted_and_ignores_non_markdown(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    user = tmp_path / "user"
    manager = Manager(str(project), str(user), None, "model")

    assert manager.list_files() == ([], [])

    project.mkdir()
    user.mkdir()
    (project / "z.md").write_text("z")
    (project / "MEMORY.md").write_text("index")
    (project / "ignored.txt").write_text("ignored")
    (user / "preference_a.md").write_text("a")

    assert manager.list_files() == (
        ["MEMORY.md", "z.md"],
        ["preference_a.md"],
    )


def test_manager_list_files_ignores_symlinks(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret")
    try:
        (project / "linked.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    manager = Manager(str(project), str(tmp_path / "user"), None, "model")

    assert manager.list_files() == ([], [])


def test_store_create_update_and_delete_note(tmp_path: Path) -> None:
    store = Store(str(tmp_path / "memory"))
    store.apply([create_action()])
    note = store.directory / "project_knowledge_api_style.md"

    created = frontmatter(note)
    assert created["type"] == "project_knowledge"
    assert created["title"] == "API style"
    assert "resource-oriented" in note.read_text()
    assert "[project_knowledge] API style" in store.load_index()

    store.apply(
        [
            UpdateAction(
                action="update",
                level="project",
                filename=note.name,
                title="Updated API style",
                content="Prefer plural resource names.",
            )
        ]
    )
    updated = frontmatter(note)
    assert updated["created"] == created["created"]
    assert updated["updated"] >= created["updated"]
    index = store.load_index()
    assert "Updated API style" in index
    assert "[project_knowledge] API style —" not in index

    store.apply([UpdateAction(action="delete", level="project", filename=note.name)])
    assert not note.exists()
    assert store.load_index() == ""


@pytest.mark.parametrize(
    "action",
    [
        create_action(slug="../escape"),
        create_action(slug="UPPER"),
        UpdateAction(action="delete", level="project", filename="../outside.md"),
        UpdateAction(action="delete", level="project", filename="/tmp/outside.md"),
        UpdateAction(action="unknown", level="project"),
        UpdateAction(action="delete", level="invalid", filename="project_knowledge_x.md"),
    ],
)
def test_store_rejects_untrusted_actions_without_escape(
    tmp_path: Path,
    action: UpdateAction,
) -> None:
    root = tmp_path / "memory"
    store = Store(str(root))

    store.apply([action])

    assert list(root.glob("*.md")) == [root / "MEMORY.md"]
    assert not (tmp_path / "outside.md").exists()


def test_store_rejects_symlink_target(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    outside = tmp_path / "outside.md"
    root.mkdir()
    outside.write_text("do not change")
    link = root / "project_knowledge_link.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    Store(str(root)).apply(
        [
            UpdateAction(
                action="update",
                level="project",
                filename=link.name,
                title="unsafe",
                content="changed",
            )
        ]
    )

    assert outside.read_text() == "do not change"


def test_store_hard_limits_index_lines_and_bytes(tmp_path: Path) -> None:
    store = Store(str(tmp_path / "memory"))
    actions = [create_action(slug=f"note_{index}") for index in range(220)]
    store.apply(actions)

    index = store.load_index()

    assert len(index.splitlines()) <= MAX_INDEX_LINES
    assert len(index.encode()) <= MAX_INDEX_BYTES


def test_manager_load_index_is_project_first_and_bounded(tmp_path: Path) -> None:
    manager = Manager(
        str(tmp_path / "project"),
        str(tmp_path / "user"),
        None,
        "",
    )
    manager.project_store.apply([create_action()])
    manager.user_store.apply(
        [
            UpdateAction(
                action="create",
                level="user",
                type="user_preference",
                title="Concise",
                slug="concise",
                content="Keep replies concise.",
            )
        ]
    )

    index = manager.load_index()

    assert index.index("[project memory]") < index.index("[user memory]")
    assert len(index.encode()) <= MAX_INDEX_BYTES


async def test_manager_update_uses_no_tools_and_refreshes_index(tmp_path: Path) -> None:
    response = (
        '[{"action":"create","level":"project","type":"project_knowledge",'
        '"title":"Python version","slug":"python_version",'
        '"content":"The project uses Python 3.12."}]'
    )
    provider = MemoryProvider(response)
    manager = Manager(
        str(tmp_path / "project"),
        str(tmp_path / "user"),
        provider,
        "memory-model",
    )

    await manager.update_async(
        [
            Message(MessageRole.USER, "Remember that this project uses Python 3.12"),
            Message(MessageRole.ASSISTANT, "Remembered."),
        ]
    )

    assert provider.tools == [()]
    assert provider.requests[0][0].role is MessageRole.SYSTEM
    assert "Python version" in manager.load_index()
    assert (tmp_path / "project" / "project_knowledge_python_version.md").is_file()


@pytest.mark.parametrize("response", ["not json", "{}", '[{"action":1}]'])
async def test_manager_update_failures_are_isolated(tmp_path: Path, response: str) -> None:
    manager = Manager(
        str(tmp_path / "project"),
        str(tmp_path / "user"),
        MemoryProvider(response),
        "model",
    )

    await manager.update_async([Message(MessageRole.USER, "remember")])

    assert manager.load_index() == ""


async def test_manager_without_provider_degrades_safely(tmp_path: Path) -> None:
    manager = Manager(str(tmp_path / "project"), str(tmp_path / "user"), None, "")

    await manager.update_async([Message(MessageRole.USER, "remember")])

    assert manager.load_index() == ""
