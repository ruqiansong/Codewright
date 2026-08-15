"""Tests for the non-configurable dangerous-command denylist."""

import pytest

from codewright.permission.blacklist import hits_blacklist


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -fr ~",
        "sudo rm --recursive --force /",
        ":(){ :|:& };:",
        "dd if=/dev/zero of=/dev/sda",
        "mkfs.ext4 /dev/sda1",
        "echo data > /dev/nvme0n1",
        "chmod -R 777 /",
    ],
)
def test_dangerous_commands_hit_blacklist(command: str) -> None:
    assert hits_blacklist(command)


@pytest.mark.parametrize(
    "command",
    ["rm -rf ./build", "git status", "ls -la", "chmod -R 755 ./scripts"],
)
def test_routine_commands_do_not_hit_blacklist(command: str) -> None:
    assert not hits_blacklist(command)


def test_blacklist_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="string"):
        hits_blacklist(123)  # type: ignore[arg-type]
