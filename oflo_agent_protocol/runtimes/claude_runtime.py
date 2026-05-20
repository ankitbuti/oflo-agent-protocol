"""Anthropic Claude runtime with prompt caching and full tool support."""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import anthropic

from oflo_agent_protocol.core.message import CanonicalMessage, ToolCall
from oflo_agent_protocol.core.types import MessageRole, TokenUsage
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)

# Anthropic models that support prompt caching
_CACHE_CAPABLE = {
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5-20251001",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
}


class ClaudeRuntime(BaseRuntime):
    """
    Anthropic Claude runtime.

    Prompt-caching strategy
    ───────────────────────
    When `use_cache=True` (default), the system prompt is wrapped with
    a `cache_control` breakpoint.  This reduces cost by ~90 % on repeated
    calls with the same system prompt (cache TTL = 5 min).
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        use_cache: bool = True,
    ) -> None:
        self.model_id = model_id
        self.use_cache = use_cache and model_id in _CACHE_CAPABLE
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", "")
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: List[CanonicalMessage],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[CanonicalMessage, TokenUsage]:
        anthropic_messages = self._to_anthropic_messages(messages)
        system_param = self._build_system(system)
        anthropic_tools = self._convert_tools(tools or [])

        params: Dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=anthropic_messages,
        )
        if system_param:
            params["system"] = system_param
        if anthropic_tools:
            params["tools"] = anthropic_tools

        try:
            response = await self._client.messages.create(**params)
        except anthropic.APIStatusError as e:
            logger.error("Anthropic API error: %s", e)
            raise

        msg = CanonicalMessage.from_anthropic_response(response)
        usage = self._parse_usage(response.usage)
        return msg, usage

    async def stream(
        self,
        messages: List[CanonicalMessage],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        anthropic_messages = self._to_anthropic_messages(messages)
        system_param = self._build_system(system)

        params: Dict[str, Any] = dict(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=anthropic_messages,
        )
        if system_param:
            params["system"] = system_param

        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    async def health_check(self) -> bool:
        try:
            await self._client.messages.create(
                model=self.model_id,
                max_tokens=5,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system(self, system: Optional[str]) -> Any:
        if not system:
            return None
        if self.use_cache:
            return [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return system

    def _to_anthropic_messages(self, messages: List[CanonicalMessage]) -> List[Dict[str, Any]]:
        out = []
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                continue  # system passed separately
            out.append(m.to_anthropic())
        return out

    @staticmethod
    def _convert_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style tool schemas to Anthropic format."""
        out = []
        for t in tools:
            if t.get("type") == "function":
                fn = t["function"]
                out.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                    }
                )
            else:
                out.append(t)
        return out

    @staticmethod
    def _parse_usage(usage: Any) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=getattr(usage, "input_tokens", 0),
            completion_tokens=getattr(usage, "output_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
