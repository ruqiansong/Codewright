"""Stable recovery attachments added after a conversation summary."""

from __future__ import annotations

import json
from collections.abc import Sequence

from codewright.compact.const import (
    ESTIMATE_CHARS_PER_TOKEN,
    RECOVERY_FILE_LIMIT,
    RECOVERY_TOKENS_PER_FILE,
)
from codewright.compact.state import FileReadRecord
from codewright.llm import ToolDefinition

BOUNDARY_NOTICE = """\
需要文件原文、错误原文或用户原话时，请使用文件读取工具重新读取对应路径。
不要依据摘要内容猜测未保留的原文或细节。"""


def render_file_block(record: FileReadRecord) -> str:
    """Render one recent file snapshot with a deterministic size bound."""
    if not isinstance(record, FileReadRecord):
        raise TypeError("record must be a FileReadRecord")
    character_limit = int(RECOVERY_TOKENS_PER_FILE * ESTIMATE_CHARS_PER_TOKEN)
    content = record.content
    if len(content) > character_limit:
        content = f"{content[:character_limit]}\n(content truncated)"
    return f"### {record.path}\n[read at] {record.timestamp.isoformat()}\n{content}"


def render_tools_block(definitions: Sequence[ToolDefinition]) -> str:
    """Render model-facing tools and their complete input schemas."""
    if not definitions:
        return "(无)"
    lines: list[str] = []
    for definition in definitions:
        if not isinstance(definition, ToolDefinition):
            raise TypeError("definitions must contain only ToolDefinition values")
        schema = json.dumps(
            dict(definition.input_schema),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        lines.extend((f"- {definition.name}: {definition.description}", f"  schema: {schema}"))
    return "\n".join(lines)


def build_recovery_attachment(
    snapshot: Sequence[FileReadRecord],
    tool_definitions: Sequence[ToolDefinition],
) -> str:
    """Build recent-files, available-tools, and boundary-notice sections."""
    file_blocks = [render_file_block(record) for record in snapshot[:RECOVERY_FILE_LIMIT]]
    recent_files = "\n\n".join(file_blocks) if file_blocks else "(无)"
    tools = render_tools_block(tool_definitions)
    return "\n\n".join(
        (
            f"## 最近读过的文件\n{recent_files}",
            f"## 当前可用工具\n{tools}",
            f"## 边界提示\n{BOUNDARY_NOTICE}",
        )
    )
