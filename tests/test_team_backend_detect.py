from __future__ import annotations

from codewright.team.backend.detect import detect_backend
from codewright.team.types import BackendType


class Controller:
    async def probe(self) -> bool:
        return True


async def test_detect_prefers_capable_iterm_then_inprocess(monkeypatch) -> None:
    monkeypatch.setattr("codewright.team.backend.detect.shutil.which", lambda name: None)
    selected = await detect_backend(
        environ={"TERM_PROGRAM": "iTerm.app"},
        iterm_controller=Controller(),  # type: ignore[arg-type]
    )
    assert selected is BackendType.ITERM2
    assert await detect_backend(environ={}) is BackendType.IN_PROCESS


async def test_detect_prefers_current_tmux(monkeypatch) -> None:
    monkeypatch.setattr("codewright.team.backend.detect.shutil.which", lambda name: "/bin/tmux")
    assert await detect_backend(environ={"TMUX": "yes"}) is BackendType.TMUX
