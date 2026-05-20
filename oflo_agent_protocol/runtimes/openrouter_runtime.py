"""OpenRouter runtime — access 300+ models through a single OpenAI-compatible API.

Best practices applied
──────────────────────
1. Multi-model fallback: pass a `models` list; OpenRouter tries each in order.
2. `route="fallback"` — silently fall through to next model on error.
3. Cost guard: `max_price` caps spend per request.
4. Provider preferences: `provider.order` for latency vs. cost tradeoffs.
5. `HTTP-Referer` + `X-Title` headers are sent on every request (required by
   OpenRouter for leaderboard rankings and rate-limit tiers).
6. Prompt caching: transparently passed through to providers that support it
   (Anthropic, OpenAI) — no extra code needed.
7. Streaming: fully supported, same as OpenAI.
8. Usage tracking: OpenRouter returns native token usage + cost in `usage`.

Docs: https://openrouter.ai/docs/quickstart
"""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import openai

from oflo_agent_protocol.core.message import CanonicalMessage, ToolCall
from oflo_agent_protocol.core.types import MessageRole, TokenUsage
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Popular OpenRouter model slugs — use these in RoutingRequest.preferred_model
MODELS = {
    # Anthropic via OpenRouter
    "claude-sonnet": "anthropic/claude-sonnet-4-6",
    "claude-opus": "anthropic/claude-opus-4-7",
    "claude-haiku": "anthropic/claude-haiku-4-5-20251001",
    # OpenAI via OpenRouter
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "o3": "openai/o3",
    # Google via OpenRouter
    "gemini-flash": "google/gemini-2.0-flash-001",
    "gemini-pro": "google/gemini-2.5-pro-preview",
    # Meta / open-source
    "llama-70b": "meta-llama/llama-3.3-70b-instruct",
    "llama-8b": "meta-llama/llama-3.1-8b-instruct",
    "mistral": "mistralai/mistral-7b-instruct",
    # Cost-optimized auto-router
    "auto": "openrouter/auto",
}


class OpenRouterRuntime(BaseRuntime):
    """
    Runtime backed by OpenRouter's unified multi-LLM API.

    Usage::

        runtime = OpenRouterRuntime(
            model_id="anthropic/claude-sonnet-4-6",
            fallback_models=["openai/gpt-4o", "meta-llama/llama-3.3-70b-instruct"],
        )
        agent = BaseAgentV2("Analyst", runtime=runtime)

    Multi-model fallback::

        runtime = OpenRouterRuntime.with_fallbacks(
            primary="anthropic/claude-sonnet-4-6",
            fallbacks=["openai/gpt-4o", MODELS["llama-70b"]],
            strategy="cost",        # "cost" | "speed" | "quality"
            max_price_per_m=5.0,    # USD per million output tokens
        )
    """

    def __init__(
        self,
        model_id: str = "anthropic/claude-sonnet-4-6",
        api_key: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        provider_order: Optional[List[str]] = None,
        max_price_input: Optional[float] = None,
        max_price_output: Optional[float] = None,
        site_url: str = "https://oflo.ai",
        site_name: str = "Oflo Agent Protocol",
        route: str = "fallback",
    ) -> None:
        self._model_id = model_id
        self._fallback_models = fallback_models or []
        self._provider_order = provider_order
        self._max_price_input = max_price_input
        self._max_price_output = max_price_output
        self._route = route
        self._site_url = site_url
        self._site_name = site_name

        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("OPENROUTER_API_KEY", ""),
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
        )

    @classmethod
    def with_fallbacks(
        cls,
        primary: str,
        fallbacks: Optional[List[str]] = None,
        strategy: str = "balanced",
        max_price_per_m: Optional[float] = None,
        **kwargs: Any,
    ) -> "OpenRouterRuntime":
        """
        Factory for multi-model routing with a clear intent.

        strategy options:
          "cost"    → sort by cheapest first
          "speed"   → route to fastest responding provider
          "quality" → prefer most capable model
          "balanced" → OpenRouter's default auto-balancing
        """
        provider_order: Optional[List[str]] = None
        if strategy == "speed":
            provider_order = ["Together", "Groq", "Fireworks"]
        elif strategy == "cost":
            provider_order = ["DeepInfra", "Together", "Groq"]
        elif strategy == "quality":
            provider_order = ["Anthropic", "OpenAI", "Google"]

        return cls(
            model_id=primary,
            fallback_models=fallbacks or [],
            provider_order=provider_order,
            max_price_output=max_price_per_m,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # BaseRuntime interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "openrouter"

    @property
    def model_id(self) -> str:
        return self._model_id

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
        params = self._build_params(oai_messages, tools, max_tokens, temperature, **kwargs)

        try:
            response = await self._client.chat.completions.create(**params)
        except openai.APIStatusError as exc:
            logger.error("OpenRouter error: %s", exc)
            raise

        choice = response.choices[0].message
        tool_calls: List[ToolCall] = []
        import json as _json
        for tc in choice.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=_json.loads(tc.function.arguments or "{}"),
                )
            )

        msg = CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=choice.content or "",
            tool_calls=tool_calls,
        )

        # OpenRouter extends usage with cost data
        usage_obj = response.usage
        usage = TokenUsage(
            prompt_tokens=usage_obj.prompt_tokens if usage_obj else 0,
            completion_tokens=usage_obj.completion_tokens if usage_obj else 0,
        )

        # Log OpenRouter cost if returned
        if usage_obj and hasattr(usage_obj, "cost"):
            logger.debug("OpenRouter call cost: $%.6f", usage_obj.cost)

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
        params = self._build_params(oai_messages, tools, max_tokens, temperature, stream=True, **kwargs)

        async with self._client.chat.completions.stream(**params) as stream:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

    async def health_check(self) -> bool:
        try:
            await self._client.chat.completions.create(
                model=self._model_id,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=3,
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_params(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        temperature: float,
        stream: bool = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Multi-model fallback: if we have fallbacks, use `models` array instead of `model`
        if self._fallback_models:
            params["models"] = [self._model_id] + self._fallback_models
            params["route"] = self._route
        else:
            params["model"] = self._model_id

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        if stream:
            params["stream"] = True

        # OpenRouter-specific extensions (passed via `extra_body` in openai SDK)
        or_extras: Dict[str, Any] = {}
        if self._provider_order:
            or_extras["provider"] = {"order": self._provider_order, "allow_fallbacks": True}
        if self._max_price_input or self._max_price_output:
            or_extras["max_price"] = {}
            if self._max_price_input:
                or_extras["max_price"]["input"] = str(self._max_price_input)
            if self._max_price_output:
                or_extras["max_price"]["output"] = str(self._max_price_output)

        if or_extras:
            params["extra_body"] = or_extras

        params.update(extra)
        return params

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


async def list_openrouter_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch live model list from OpenRouter (prices, context windows, etc.)."""
    import aiohttp
    headers = {"Authorization": f"Bearer {api_key or os.getenv('OPENROUTER_API_KEY', '')}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://openrouter.ai/api/v1/models", headers=headers
        ) as resp:
            data = await resp.json()
            return data.get("data", [])
