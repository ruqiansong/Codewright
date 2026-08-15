"""Backward-compatible DeepSeek provider import."""

from codewright.llm.openai_provider import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider retained under the V0.1 public name."""
