from __future__ import annotations

from codewright.team.backend.tmux import TmuxBackend
from codewright.team.types import RuntimeHandle, SpawnRequest


class Runner:
    def __init__(self) -> None:
        self.calls = []

    async def __call__(self, argv):
        self.calls.append(tuple(argv))
        if "list-panes" in argv:
            return 0, "%1\n%7", ""
        return 0, "%7", ""


async def test_tmux_inside_uses_argv_without_shell_quoting() -> None:
    runner = Runner()
    backend = TmuxBackend(runner=runner, environ={"TMUX": "/tmp/tmux socket"})
    result = await backend.spawn(SpawnRequest("demo team", "alice's name", "secret"))

    argv = runner.calls[0]
    assert argv[:2] == ("tmux", "split-window")
    assert "demo team" in argv
    assert "alice's name" in argv
    assert "secret" not in argv
    assert result.pane_id == "%7"


async def test_tmux_lifecycle_targets_exact_pane() -> None:
    runner = Runner()
    backend = TmuxBackend(runner=runner, environ={})
    handle = RuntimeHandle("demo", "alice", pane_id="%7")
    assert await backend.is_alive(handle)
    await backend.wake(handle)
    await backend.kill(handle)
    assert runner.calls[-1] == ("tmux", "kill-pane", "-t", "%7")
