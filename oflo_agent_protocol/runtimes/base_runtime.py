"""Abstract runtime interface — every LLM backend implements this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from oflo_agent_protocol.core.message import CanonicalMessage
from oflo_agent_protocol.core.types import TokenUsage


class BaseRuntime(ABC):
    """
    A runtime wraps a single LLM provider SDK and converts between
    CanonicalMessage ↔ provider format.

    Token optimisation contract
    ───────────────────────────
    Runtimes MUST populate TokenUsage including cache_read/write_tokens
    when the provider supports it (Anthropic prompt caching).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...

    @abstractmethod
    async def complete(
        self,
        messages: List[CanonicalMessage],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[CanonicalMessage, TokenUsage]:
        """Single-turn completion. Returns (reply_message, token_usage)."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: List[CanonicalMessage],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Streaming completion — yield text chunks."""
        ...

    async def health_check(self) -> bool:
        """Returns True if the provider is reachable."""
        return True
