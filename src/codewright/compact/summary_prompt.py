"""Stable prompt construction and parsing for conversation summaries."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from codewright.llm import Message, MessageRole

logger = logging.getLogger(__name__)

SUMMARY_INSTRUCTION = """\
你正在压缩 Codewright 编程助手的一段历史会话。

请分两个阶段输出：
1. 在 <analysis>...</analysis> 中整理分析草稿。
2. 在 <summary>...</summary> 中给出正式摘要。

正式摘要必须严格包含以下九个小节，保持标题和顺序不变：
## 1 主要请求和意图
## 2 关键技术概念
## 3 文件和代码段
## 4 错误和修复
## 5 问题解决过程
## 6 摘要请求实际收到的用户消息原文
## 7 待办任务
## 8 当前工作（最详细）
## 9 可能的下一步

第 6 节按时间顺序逐条保留本次摘要请求实际收到的用户消息原文。
如果输入中注明旧消息组已被丢弃，必须记录丢弃数量。
不要调用任何工具，输出纯文本。"""


def serialize_conversation(messages: Sequence[Message]) -> str:
    """Serialize provider-neutral conversation messages deterministically."""
    lines: list[str] = []
    for message in messages:
        if not isinstance(message, Message):
            raise TypeError("messages must contain only Message values")
        if message.role is not MessageRole.TOOL:
            lines.append(f"{message.role.value}: {message.content}")
        for call in message.tool_calls:
            lines.append(f"[call {call.name} id={call.id} args={call.arguments_json}]")
        for result in message.tool_results:
            lines.append(
                f"[result id={result.tool_call_id} name={result.tool_name} "
                f"is_error={str(result.is_error).lower()}] {result.content}"
            )
    return "\n".join(lines)


def build_summary_prompt(messages: Sequence[Message]) -> list[Message]:
    """Build one user message containing instructions and serialized history."""
    serialized = serialize_conversation(messages)
    content = f"{SUMMARY_INSTRUCTION}\n\n[conversation]\n{serialized}"
    return [Message(MessageRole.USER, content)]


def extract_summary(raw: str) -> str:
    """Return the final tagged summary, or the unmodified response as fallback."""
    if not isinstance(raw, str):
        raise TypeError("raw must be a string")
    start = raw.rfind("<summary>")
    if start >= 0:
        content_start = start + len("<summary>")
        end = raw.find("</summary>", content_start)
        if end >= 0:
            return raw[content_start:end].strip()
    logger.warning("Summary tags not found in compact response")
    return raw
