"""Owned lifecycle and state transitions for background Agent tasks."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum

from codewright.agent import Agent, CompletionResult, Event, Phase
from codewright.conversation import Conversation
from codewright.llm import LLMError, Provider, TokenUsage

DONE_QUEUE_SIZE = 100
PROVIDER_CLOSE_TIMEOUT = 5.0
logger = logging.getLogger(__name__)
type CompletionListener = Callable[[BackgroundTask], object]


class Status(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ManagerError(RuntimeError):
    """Stable task-manager failure for model-facing adapters."""

    def __init__(self, code: str, safe_message: str) -> None:
        self.code = code
        self.safe_message = safe_message
        super().__init__(safe_message)


@dataclass(slots=True)
class BackgroundTask:
    id: str
    name: str
    description: str
    sub_agent: Agent
    conversation: Conversation
    initial_prompt: str
    status: Status = Status.RUNNING
    handle: asyncio.Task[CompletionResult] | None = None
    result: str = ""
    error_type: str = ""
    error_message: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0, 0))
    tool_count: int = 0
    last_activity: str = ""
    owned_provider: Provider | None = None
    notification_generation: int = 0
    _generation: int = field(default=0, repr=False)
    _notified_generation: int = field(default=-1, repr=False)


class Manager:
    """Register, observe, continue, stop, and close background Agent tasks."""

    def __init__(self, *, done_queue_size: int = DONE_QUEUE_SIZE) -> None:
        if (
            not isinstance(done_queue_size, int)
            or isinstance(done_queue_size, bool)
            or done_queue_size <= 0
        ):
            raise ValueError("done_queue_size must be a positive integer")
        self._tasks: dict[str, BackgroundTask] = {}
        self._names: dict[str, str] = {}
        self._observers: dict[str, asyncio.Task[None]] = {}
        self._done: asyncio.Queue[BackgroundTask | None] = asyncio.Queue(maxsize=done_queue_size)
        self._lock = asyncio.Lock()
        self._counter = 0
        self._closed = False
        self._completion_listeners: set[CompletionListener] = set()

    def add_completion_listener(self, listener: CompletionListener) -> None:
        """Add a non-consuming observer called after every terminal generation."""
        if not callable(listener):
            raise TypeError("completion listener must be callable")
        self._completion_listeners.add(listener)

    def remove_completion_listener(self, listener: CompletionListener) -> None:
        """Remove a previously registered completion observer."""
        self._completion_listeners.discard(listener)

    async def launch(
        self,
        sub_agent: Agent,
        conversation: Conversation,
        initial_prompt: str,
        description: str,
        *,
        name: str = "",
        owned_provider: Provider | None = None,
    ) -> BackgroundTask:
        """Register first, then start one new Agent completion generation."""
        self._validate_task_inputs(
            sub_agent, conversation, initial_prompt, description, name, owned_provider
        )
        async with self._lock:
            self._ensure_open()
            task = self._new_task(
                sub_agent,
                conversation,
                initial_prompt,
                description,
                name,
                owned_provider,
            )
            self._register(task)
            sink: asyncio.Queue[Event | None] = asyncio.Queue()
            task.handle = asyncio.create_task(
                sub_agent.run_to_completion(conversation, "", event_sink=sink)
            )
            self._observe(task, sink)
            return task

    async def adopt_running(
        self,
        sub_agent: Agent,
        conversation: Conversation,
        initial_prompt: str,
        description: str,
        handle: asyncio.Task[CompletionResult],
        *,
        name: str = "",
        owned_provider: Provider | None = None,
        event_sink: asyncio.Queue[Event | None] | None = None,
    ) -> BackgroundTask:
        """Observe an existing completion Task without invoking the Agent again."""
        self._validate_task_inputs(
            sub_agent, conversation, initial_prompt, description, name, owned_provider
        )
        if not isinstance(handle, asyncio.Task):
            raise TypeError("handle must be an asyncio Task")
        async with self._lock:
            self._ensure_open()
            task = self._new_task(
                sub_agent,
                conversation,
                initial_prompt,
                description,
                name,
                owned_provider,
            )
            task.handle = handle
            self._register(task)
            self._observe(task, event_sink)
            return task

    async def get(self, task_id: str) -> BackgroundTask | None:
        """Return one retained task by id."""
        async with self._lock:
            return self._tasks.get(task_id)

    async def list(self) -> tuple[BackgroundTask, ...]:
        """Return all retained tasks in launch order."""
        async with self._lock:
            return tuple(self._tasks.values())

    async def stop(self, task_id: str) -> BackgroundTask:
        """Cancel one running generation and wait for terminal state."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ManagerError("unknown_task", f"Unknown task: {task_id}")
            if task.status is not Status.RUNNING or task.handle is None:
                raise ManagerError("task_not_running", f"Task is not running: {task_id}")
            handle = task.handle
            observer = self._observers.get(task_id)
            handle.cancel()
        await asyncio.gather(handle, return_exceptions=True)
        if observer is not None:
            await observer
        return task

    async def send_message(self, name: str, message: str) -> BackgroundTask:
        """Continue the latest named completed task as a new generation."""
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ManagerError("invalid_name", "name must be a non-empty trimmed string")
        if not isinstance(message, str) or not message.strip() or message != message.strip():
            raise ManagerError("invalid_message", "message must be a non-empty trimmed string")
        async with self._lock:
            self._ensure_open()
            task_id = self._names.get(name)
            task = self._tasks.get(task_id) if task_id is not None else None
            if task is None:
                raise ManagerError("unknown_task_name", f"Unknown task name: {name}")
            if task.status is not Status.COMPLETED:
                raise ManagerError("task_not_completed", f"Task is not completed: {task.id}")
            task.status = Status.RUNNING
            task.result = ""
            task.error_type = ""
            task.error_message = ""
            task.started_at = time.time()
            task.ended_at = 0.0
            task.tool_count = 0
            task.last_activity = ""
            task.usage = TokenUsage(0, 0, 0)
            task._generation += 1
            sink: asyncio.Queue[Event | None] = asyncio.Queue()
            task.handle = asyncio.create_task(
                task.sub_agent.run_to_completion(
                    task.conversation,
                    message,
                    event_sink=sink,
                )
            )
            self._observe(task, sink)
            return task

    def subscribe_done(self) -> asyncio.Queue[BackgroundTask | None]:
        """Return the single stable task-completion queue."""
        return self._done

    async def aclose(self) -> None:
        """Idempotently stop all work, close owned Providers, and end subscribers."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            running = tuple(
                task.handle
                for task in self._tasks.values()
                if task.status is Status.RUNNING and task.handle is not None
            )
            observers = tuple(self._observers.values())
            providers = tuple(
                {
                    id(task.owned_provider): task.owned_provider for task in self._tasks.values()
                }.values()
            )
            for handle in running:
                handle.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        if observers:
            await asyncio.gather(*observers, return_exceptions=True)
        for provider in providers:
            if provider is not None:
                await _close_provider(provider)
        _offer_shutdown(self._done)

    def _new_task(
        self,
        sub_agent: Agent,
        conversation: Conversation,
        initial_prompt: str,
        description: str,
        name: str,
        owned_provider: Provider | None,
    ) -> BackgroundTask:
        self._counter += 1
        return BackgroundTask(
            id=f"task-{self._counter}",
            name=name,
            description=description,
            sub_agent=sub_agent,
            conversation=conversation,
            initial_prompt=initial_prompt,
            started_at=time.time(),
            owned_provider=owned_provider,
        )

    def _register(self, task: BackgroundTask) -> None:
        self._tasks[task.id] = task
        if task.name:
            self._names[task.name] = task.id

    def _observe(
        self,
        task: BackgroundTask,
        event_sink: asyncio.Queue[Event | None] | None,
    ) -> None:
        observer = asyncio.create_task(self._observe_generation(task, event_sink))
        self._observers[task.id] = observer

    async def _observe_generation(
        self,
        task: BackgroundTask,
        event_sink: asyncio.Queue[Event | None] | None,
    ) -> None:
        generation = task._generation
        terminal_status = Status.FAILED
        collector = (
            asyncio.create_task(self._collect_events(task, event_sink))
            if event_sink is not None
            else None
        )
        try:
            if task.handle is None:
                raise RuntimeError("task handle was not initialized")
            result = await task.handle
            task.result = result.text
            task.usage = result.usage
            terminal_status = Status.COMPLETED
        except asyncio.CancelledError:
            terminal_status = Status.CANCELLED
            task.error_type = "CancelledError"
            task.error_message = "Task was cancelled."
        except Exception as error:
            terminal_status = Status.FAILED
            task.error_type = type(error).__name__
            task.error_message = _safe_error_message(error)
        finally:
            if collector is not None:
                await asyncio.sleep(0)
                if not collector.done():
                    collector.cancel()
                with suppress(asyncio.CancelledError):
                    await collector
            task.ended_at = time.time()
            task.status = terminal_status
            if task._notified_generation < generation:
                task._notified_generation = generation
                task.notification_generation = generation + 1
                _offer_done(self._done, task)
                await self._notify_completion_listeners(task)

    async def _notify_completion_listeners(self, task: BackgroundTask) -> None:
        for listener in tuple(self._completion_listeners):
            try:
                result = listener(task)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning(
                    "Task completion listener failed listener=%r error=%s",
                    listener,
                    type(error).__name__,
                )

    async def _collect_events(
        self,
        task: BackgroundTask,
        event_sink: asyncio.Queue[Event | None],
    ) -> None:
        while True:
            event = await event_sink.get()
            if event is None:
                return
            if event.tool is not None and event.tool.phase is Phase.END:
                task.tool_count += 1
                task.last_activity = f"tool:{event.tool.name}"
            elif event.text:
                task.last_activity = event.text[-200:]
            elif event.notice:
                task.last_activity = event.notice[-200:]

    def _ensure_open(self) -> None:
        if self._closed:
            raise ManagerError("manager_closed", "Task manager is closed")

    @staticmethod
    def _validate_task_inputs(
        sub_agent: Agent,
        conversation: Conversation,
        initial_prompt: str,
        description: str,
        name: str,
        owned_provider: Provider | None,
    ) -> None:
        if not isinstance(sub_agent, Agent):
            raise TypeError("sub_agent must be an Agent")
        if not isinstance(conversation, Conversation):
            raise TypeError("conversation must be a Conversation")
        for field_name, value in (
            ("initial_prompt", initial_prompt),
            ("description", description),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(name, str) or name != name.strip():
            raise ValueError("name must be a trimmed string")
        if owned_provider is not None and not isinstance(owned_provider, Provider):
            raise TypeError("owned_provider must satisfy Provider or be None")


def _safe_error_message(error: Exception) -> str:
    if isinstance(error, LLMError):
        return error.safe_message
    safe_message = getattr(error, "safe_message", None)
    if isinstance(safe_message, str) and safe_message.strip():
        return safe_message.strip()
    return "Background task failed."


async def _close_provider(provider: Provider) -> None:
    close = getattr(provider, "close", None)
    if close is None:
        return
    try:
        result = close()
        if inspect.isawaitable(result):
            await asyncio.wait_for(result, timeout=PROVIDER_CLOSE_TIMEOUT)
    except asyncio.CancelledError:
        raise
    except Exception:
        pass


def _offer_done(queue: asyncio.Queue[BackgroundTask | None], task: BackgroundTask) -> None:
    try:
        queue.put_nowait(task)
    except asyncio.QueueFull:
        pass


def _offer_shutdown(queue: asyncio.Queue[BackgroundTask | None]) -> None:
    try:
        queue.put_nowait(None)
    except asyncio.QueueFull:
        with suppress(asyncio.QueueEmpty):
            queue.get_nowait()
        with suppress(asyncio.QueueFull):
            queue.put_nowait(None)


__all__ = [
    "BackgroundTask",
    "DONE_QUEUE_SIZE",
    "Manager",
    "ManagerError",
    "Status",
]
