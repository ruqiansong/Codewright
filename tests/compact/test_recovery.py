"""Tests for deterministic post-summary recovery attachments."""

from datetime import UTC, datetime, timedelta

from codewright.compact.const import ESTIMATE_CHARS_PER_TOKEN, RECOVERY_TOKENS_PER_FILE
from codewright.compact.recovery import (
    BOUNDARY_NOTICE,
    build_recovery_attachment,
    render_file_block,
    render_tools_block,
)
from codewright.compact.state import FileReadRecord
from codewright.llm import ToolDefinition


def definition(name: str) -> ToolDefinition:
    return ToolDefinition(
        name,
        f"Use {name}",
        {"type": "object", "properties": {"path": {"type": "string"}}},
    )


def test_render_file_block_keeps_head_and_marks_truncation() -> None:
    limit = int(RECOVERY_TOKENS_PER_FILE * ESTIMATE_CHARS_PER_TOKEN)
    record = FileReadRecord("/tmp/file.py", "a" * (limit + 10), datetime.now(UTC))

    rendered = render_file_block(record)

    assert "a" * limit in rendered
    assert rendered.endswith("(content truncated)")
    assert "a" * (limit + 1) not in rendered


def test_render_tools_block_includes_stable_complete_schema() -> None:
    definitions = (definition("read_file"), definition("grep"))

    rendered = render_tools_block(definitions)

    assert rendered == render_tools_block(definitions)
    assert "- read_file: Use read_file" in rendered
    assert 'schema: {"properties":{"path":{"type":"string"}},"type":"object"}' in rendered


def test_build_recovery_attachment_limits_files_and_is_stable() -> None:
    now = datetime.now(UTC)
    records = [
        FileReadRecord(f"/tmp/{index}.py", str(index), now - timedelta(seconds=index))
        for index in range(7)
    ]
    definitions = (definition("read_file"),)

    attachment = build_recovery_attachment(records, definitions)

    assert attachment == build_recovery_attachment(records, definitions)
    for index in range(5):
        assert f"/tmp/{index}.py" in attachment
    assert "/tmp/5.py" not in attachment
    assert "/tmp/6.py" not in attachment
    assert "## 最近读过的文件" in attachment
    assert "## 当前可用工具" in attachment
    assert f"## 边界提示\n{BOUNDARY_NOTICE}" in attachment


def test_build_recovery_attachment_marks_empty_sections() -> None:
    attachment = build_recovery_attachment((), ())

    assert attachment.count("(无)") == 2
