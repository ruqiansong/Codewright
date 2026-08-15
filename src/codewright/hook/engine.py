"""Ordered Hook dispatch with failure isolation and background task ownership."""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Protocol

from codewright.hook.event import Event, is_blocking
from codewright.hook.executor import Executor
from codewright.hook.matcher import eval_condition
from codewright.hook.rule import Payload, Rule


class HookSessionState(Protocol):
    """Minimal session-scoped state required by the Hook Engine."""

    def claim_hook_once(self, name: str) -> bool:
        """Return True only for the first claim of a Hook name."""
        ...


@dataclass(slots=True)
class DispatchResult:
    blocked: bool = False
    reason: str = ""
    blocking_hook_name: str = ""
    injected_prompts: list[str] = field(default_factory=list)


class Engine:
    """Evaluate Hook rules in declaration order and coordinate action execution."""

    def __init__(
        self,
        rules: list[Rule] | tuple[Rule, ...],
        sources: list[str] | tuple[str, ...],
        *,
        executor: Executor | None = None,
    ) -> None:
        self._rules = tuple(rules)
        self._sources = tuple(sources)
        self._executor = executor or Executor()
        self._background: set[asyncio.Task[None]] = set()

    async def dispatch(
        self,
        event: Event,
        payload: Payload,
        session: HookSessionState,
    ) -> DispatchResult:
        """Run every matching rule until one blocks a blocking event."""
        result = DispatchResult()
        for rule in self._rules:
            if rule.event is not event or not eval_condition(rule.condition, payload):
                continue
            if rule.only_once and not session.claim_hook_once(rule.name):
                continue
            if rule.asyncio_mode:
                task = asyncio.create_task(self._run_background(rule, dict(payload)))
                self._background.add(task)
                task.add_done_callback(self._background.discard)
                continue

            outcome = await self._executor.run(rule, payload, blocking=is_blocking(event))
            if outcome.err is not None:
                self._log_failure(rule, outcome.err)
                continue
            if outcome.prompt:
                result.injected_prompts.append(outcome.prompt)
            if outcome.blocked and is_blocking(event):
                result.blocked = True
                result.reason = outcome.reason
                result.blocking_hook_name = rule.name
                break
        return result

    async def _run_background(self, rule: Rule, payload: Payload) -> None:
        try:
            outcome = await self._executor.run(rule, payload, blocking=False)
            if outcome.err is not None:
                self._log_failure(rule, outcome.err)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._log_failure(rule, error)

    @staticmethod
    def _log_failure(rule: Rule, error: Exception) -> None:
        print(f"[hook {rule.name}] {rule.event.value} failed: {error}", file=sys.stderr)

    async def aclose(self) -> None:
        """Drain background work, cancelling it only if shutdown is cancelled."""
        try:
            await asyncio.gather(*tuple(self._background), return_exceptions=True)
        except asyncio.CancelledError:
            for task in tuple(self._background):
                task.cancel()
            await asyncio.gather(*tuple(self._background), return_exceptions=True)
            raise
        finally:
            self._background.clear()
            await self._executor.aclose()

    @property
    def sources(self) -> list[str]:
        return list(self._sources)

    @property
    def rules(self) -> list[Rule]:
        return list(self._rules)


__all__ = ["DispatchResult", "Engine", "HookSessionState"]
