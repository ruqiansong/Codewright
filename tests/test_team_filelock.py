from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from codewright.team.persistence import FileLock, LockTimeoutError


async def test_filelock_times_out_while_live_owner_holds_lock(tmp_path: Path) -> None:
    path = tmp_path / "resource.lock"
    async with FileLock(path):
        with pytest.raises(LockTimeoutError):
            async with FileLock(path, timeout=0.01):
                raise AssertionError("unreachable")


async def test_filelock_clears_only_dead_stale_owner(tmp_path: Path) -> None:
    path = tmp_path / "resource.lock"
    path.write_text(json.dumps({"token": "old", "pid": 999_999_999, "created_at": 0}))
    old = time.time() - 60
    os.utime(path, (old, old))
    async with FileLock(path, timeout=0.2, stale_after=0.01) as lock:
        owner = json.loads(path.read_text())
        assert owner["token"] == lock.owner_token


async def test_release_does_not_remove_replaced_owner(tmp_path: Path) -> None:
    path = tmp_path / "resource.lock"
    lock = FileLock(path)
    await lock.__aenter__()
    path.write_text(json.dumps({"token": "replacement", "pid": os.getpid()}))
    await lock.__aexit__(None, None, None)
    assert path.exists()
