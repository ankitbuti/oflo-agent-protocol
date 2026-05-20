"""Smart LLM router — selects the optimal model per request.

Strategy hierarchy:
  1. Explicit override on AgentManager / agent
  2. Task-level routing hints  (need_vision, need_long_ctx, etc.)
  3. Strategy: CHEAPEST / FASTEST / SMARTEST / BALANCED / CAPABILITY_MATCH
  4. Fallback chain if the primary provider is unavailable
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from oflo_agent_protocol.core.types import (
    ModelCapabilities,
    ModelConfig,
    ModelProvider,
    RoutingStrategy,
)
from oflo_agent_protocol.routing.providers import PROVIDER_REGISTRY

logger = logging.getLogger(__name__)


# Providers we can actually reach (driven by env-var presence)
def _available_providers() -> Set[ModelProvider]:
    available: Set[ModelProvider] = set()
    if os.getenv("ANTHROPIC_API_KEY"):
        available.add(ModelProvider.ANTHROPIC)
    if os.getenv("OPENAI_API_KEY"):
        available.add(ModelProvider.OPENAI)
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        available.add(ModelProvider.GOOGLE)
    if os.getenv("GROQ_API_KEY"):
        available.add(ModelProvider.GROQ)
    if os.getenv("OLLAMA_HOST"):
        available.add(ModelProvider.OLLAMA)
    if os.getenv("OPENROUTER_API_KEY"):
        available.add(ModelProvider.OPENROUTER)
    return available or {ModelProvider.ANTHROPIC}  # at least try something


class RoutingRequest:
    """Describes what we need from the model."""

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        need_vision: bool = False,
        need_long_context: bool = False,
        need_function_calling: bool = True,
        need_json_mode: bool = False,
        max_cost_per_m: Optional[float] = None,
        preferred_provider: Optional[ModelProvider] = None,
        preferred_model: Optional[str] = None,
        excluded_providers: Optional[List[ModelProvider]] = None,
        max_tokens: int = 4096,
        task_complexity: float = 0.5,  # 0.0=trivial, 1.0=expert
    ) -> None:
        self.strategy = strategy
        self.need_vision = need_vision
        self.need_long_context = need_long_context
        self.need_function_calling = need_function_calling
        self.need_json_mode = need_json_mode
        self.max_cost_per_m = max_cost_per_m
        self.preferred_provider = preferred_provider
        self.preferred_model = preferred_model
        self.excluded_providers = set(excluded_providers or [])
        self.max_tokens = max_tokens
        self.task_complexity = task_complexity

    def required_capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            vision=self.need_vision,
            long_context=self.need_long_context,
            function_calling=self.need_function_calling,
            json_mode=self.need_json_mode,
        )


class RouterDecision:
    def __init__(self, model: ModelConfig, fallback_chain: List[ModelConfig]) -> None:
        self.model = model
        self.fallback_chain = fallback_chain

    @property
    def provider(self) -> ModelProvider:
        return self.model.provider

    @property
    def model_id(self) -> str:
        return self.model.model_id

    def __repr__(self) -> str:
        return f"RouterDecision({self.provider.value}/{self.model_id})"


class SmartRouter:
    """
    Stateless router — call `route(request)` to get the best model.

    Token-optimization notes
    ─────────────────────────
    • Anthropic models support prompt caching (`cache_control` blocks).
      The runtimes handle cache headers; the router just picks Anthropic
      models when the task benefits from caching (long system prompt).
    • BALANCED strategy biases toward Anthropic Haiku / OpenAI gpt-4o-mini
      for simple tasks and Claude Sonnet / GPT-4o for complex ones.
    """

    def __init__(self, registry=PROVIDER_REGISTRY) -> None:
        self._registry = registry
        self._available = _available_providers()

    def refresh_availability(self) -> None:
        self._available = _available_providers()

    def route(self, request: RoutingRequest) -> RouterDecision:
        caps = request.required_capabilities()
        candidates = self._eligible_candidates(request, caps)

        if not candidates:
            # Desperate fallback — ignore exclusions
            candidates = self._registry.list_all()

        primary = self._score_and_pick(candidates, request)
        fallbacks = [m for m in self._fallback_order(candidates, request) if m != primary][:3]

        logger.debug(
            "Routing decision: %s/%s (strategy=%s, fallbacks=%d)",
            primary.provider.value,
            primary.model_id,
            request.strategy.value,
            len(fallbacks),
        )
        return RouterDecision(primary, fallbacks)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eligible_candidates(
        self, req: RoutingRequest, caps: ModelCapabilities
    ) -> List[ModelConfig]:
        all_models = self._registry._filter(caps)

        # Hard preference
        if req.preferred_provider and req.preferred_model:
            m = self._registry.get_model(req.preferred_provider, req.preferred_model)
            if m:
                return [m]

        out = []
        for m in all_models:
            if m.provider in req.excluded_providers:
                continue
            if m.provider not in self._available:
                continue
            if req.max_cost_per_m and m.cost_score > req.max_cost_per_m:
                continue
            out.append(m)

        # Respect preferred provider if specified
        if req.preferred_provider:
            preferred = [m for m in out if m.provider == req.preferred_provider]
            if preferred:
                return preferred

        return out

    def _score_and_pick(self, candidates: List[ModelConfig], req: RoutingRequest) -> ModelConfig:
        s = req.strategy
        if s == RoutingStrategy.CHEAPEST:
            return min(candidates, key=lambda m: m.cost_score)
        if s == RoutingStrategy.FASTEST:
            return min(candidates, key=lambda m: m.avg_latency_ms)
        if s == RoutingStrategy.SMARTEST:
            return max(candidates, key=lambda m: m.cost_score)
        if s == RoutingStrategy.CAPABILITY_MATCH:
            return self._capability_match(candidates, req)
        # BALANCED — composite score
        return min(candidates, key=lambda m: self._balanced_score(m, req))

    def _balanced_score(self, m: ModelConfig, req: RoutingRequest) -> float:
        cost_norm = m.cost_score / 100.0
        latency_norm = m.avg_latency_ms / 10_000.0
        complexity_adj = req.task_complexity  # 0-1; high complexity → prefer capable models
        # Invert: lower score = better pick
        return cost_norm * (1 - complexity_adj * 0.5) + latency_norm * 0.3

    def _capability_match(self, candidates: List[ModelConfig], req: RoutingRequest) -> ModelConfig:
        # Score by how many required caps the model satisfies, then by balanced score
        def _cap_score(m: ModelConfig) -> Tuple[int, float]:
            caps = m.capabilities
            hits = sum([
                caps.vision if req.need_vision else 1,
                caps.long_context if req.need_long_context else 1,
                caps.function_calling if req.need_function_calling else 1,
                caps.json_mode if req.need_json_mode else 1,
            ])
            return (-hits, self._balanced_score(m, req))

        return min(candidates, key=_cap_score)

    def _fallback_order(self, candidates: List[ModelConfig], req: RoutingRequest) -> List[ModelConfig]:
        return sorted(candidates, key=lambda m: self._balanced_score(m, req))

    def describe(self) -> Dict[str, Any]:
        return {
            "available_providers": [p.value for p in self._available],
            "total_models": len(self._registry.list_all()),
        }


# Module-level singleton — import and use directly
_router = SmartRouter()


def get_router() -> SmartRouter:
    return _router


def route(
    strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    *,
    need_vision: bool = False,
    need_long_context: bool = False,
    need_function_calling: bool = True,
    preferred_provider: Optional[ModelProvider] = None,
    preferred_model: Optional[str] = None,
    task_complexity: float = 0.5,
) -> RouterDecision:
    """Convenience function — route a request with common params."""
    return _router.route(
        RoutingRequest(
            strategy=strategy,
            need_vision=need_vision,
            need_long_context=need_long_context,
            need_function_calling=need_function_calling,
            preferred_provider=preferred_provider,
            preferred_model=preferred_model,
            task_complexity=task_complexity,
        )
    )
