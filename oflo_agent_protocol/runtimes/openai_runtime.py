"""OpenAI runtime — Chat Completions API + Agent SDK support."""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import openai

from oflo_agent_protocol.core.message import CanonicalMessage, ToolCall
from oflo_agent_protocol.core.types import MessageRole, TokenUsage
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)


class OpenAIRuntime(BaseRuntime):
    """OpenAI Chat Completions runtime with tool-call support."""

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    async def complete(
        self,
        messages: List[CanonicalMessage],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[CanonicalMessage, TokenUsage]:
        oai_messages = self._build_messages(messages, system)

        params: Dict[str, Any] = dict(
            model=self.model_id,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        try:
            response = await self._client.chat.completions.create(**params)
        except openai.APIStatusError as e:
            logger.error("OpenAI API error: %s", e)
            raise

        choice = response.choices[0].message
        tool_calls: List[ToolCall] = []
        for tc in choice.tool_calls or []:
            import json
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                )
            )

        msg = CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=choice.content or "",
            tool_calls=tool_calls,
        )
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
        )
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
        oai_messages = self._build_messages(messages, system)
        params: Dict[str, Any] = dict(
            model=self.model_id,
            messages=oai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async with self._client.chat.completions.stream(**params) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

    async def health_check(self) -> bool:
        try:
            await self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=3,
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        messages: List[CanonicalMessage], system: Optional[str]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if system:
            out.append({"role": "system", "content": system})
        for m in messages:
            if m.role == MessageRole.SYSTEM:
                continue
            out.append(m.to_openai())
        return out


class GroqRuntime(OpenAIRuntime):
    """Groq runtime — OpenAI-compatible API at high speed."""

    def __init__(self, model_id: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None) -> None:
        super().__init__(
            model_id=model_id,
            api_key=api_key or os.getenv("GROQ_API_KEY", ""),
            base_url="https://api.groq.com/openai/v1",
        )

    @property
    def provider_name(self) -> str:
        return "groq"


class OllamaRuntime(OpenAIRuntime):
    """Local Ollama runtime — OpenAI-compatible endpoint."""

    def __init__(
        self,
        model_id: str = "llama3.2",
        host: Optional[str] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            api_key="ollama",  # Ollama doesn't require a real key
            base_url=f"{host or os.getenv('OLLAMA_HOST', 'http://localhost:11434')}/v1",
        )

    @property
    def provider_name(self) -> str:
        return "ollama"
