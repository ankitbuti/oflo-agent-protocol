"""Provider registry — model configs, pricing, and capability flags."""
from __future__ import annotations

from typing import Dict, List, Optional

from oflo_agent_protocol.core.types import (
    ModelCapabilities,
    ModelConfig,
    ModelProvider,
)

# ---------------------------------------------------------------------------
# Canonical model catalogue  (prices as of May 2025 — update as needed)
# ---------------------------------------------------------------------------

_MODELS: List[ModelConfig] = [
    # ── Anthropic ──────────────────────────────────────────────────────────
    ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-opus-4-7",
        display_name="Claude Opus 4.7",
        input_cost_per_m=15.0,
        output_cost_per_m=75.0,
        cache_read_cost_per_m=1.50,
        cache_write_cost_per_m=18.75,
        avg_latency_ms=2500,
        priority=2,
        capabilities=ModelCapabilities(
            vision=True, long_context=True, function_calling=True,
            streaming=True, json_mode=True, context_window=200_000
        ),
    ),
    ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        input_cost_per_m=3.0,
        output_cost_per_m=15.0,
        cache_read_cost_per_m=0.30,
        cache_write_cost_per_m=3.75,
        avg_latency_ms=1200,
        priority=1,
        capabilities=ModelCapabilities(
            vision=True, long_context=True, function_calling=True,
            streaming=True, json_mode=True, context_window=200_000
        ),
    ),
    ModelConfig(
        provider=ModelProvider.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
        display_name="Claude Haiku 4.5",
        input_cost_per_m=0.25,
        output_cost_per_m=1.25,
        cache_read_cost_per_m=0.03,
        cache_write_cost_per_m=0.30,
        avg_latency_ms=400,
        priority=1,
        capabilities=ModelCapabilities(
            vision=True, function_calling=True, streaming=True,
            json_mode=True, context_window=200_000
        ),
    ),
    # ── OpenAI ─────────────────────────────────────────────────────────────
    ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o",
        display_name="GPT-4o",
        input_cost_per_m=2.50,
        output_cost_per_m=10.0,
        avg_latency_ms=1500,
        priority=2,
        capabilities=ModelCapabilities(
            vision=True, long_context=False, function_calling=True,
            streaming=True, json_mode=True, context_window=128_000
        ),
    ),
    ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        input_cost_per_m=0.15,
        output_cost_per_m=0.60,
        avg_latency_ms=500,
        priority=1,
        capabilities=ModelCapabilities(
            vision=True, function_calling=True, streaming=True,
            json_mode=True, context_window=128_000
        ),
    ),
    ModelConfig(
        provider=ModelProvider.OPENAI,
        model_id="o3",
        display_name="OpenAI o3",
        input_cost_per_m=10.0,
        output_cost_per_m=40.0,
        avg_latency_ms=8000,
        priority=3,
        capabilities=ModelCapabilities(
            function_calling=True, streaming=False,
            json_mode=True, context_window=200_000
        ),
    ),
    # ── Google ─────────────────────────────────────────────────────────────
    ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        input_cost_per_m=0.075,
        output_cost_per_m=0.30,
        avg_latency_ms=350,
        priority=1,
        capabilities=ModelCapabilities(
            vision=True, long_context=True, function_calling=True,
            streaming=True, json_mode=True, context_window=1_000_000
        ),
    ),
    ModelConfig(
        provider=ModelProvider.GOOGLE,
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        input_cost_per_m=1.25,
        output_cost_per_m=5.0,
        avg_latency_ms=2000,
        priority=2,
        capabilities=ModelCapabilities(
            vision=True, long_context=True, function_calling=True,
            streaming=True, json_mode=True, context_window=1_000_000
        ),
    ),
    # ── Groq ───────────────────────────────────────────────────────────────
    ModelConfig(
        provider=ModelProvider.GROQ,
        model_id="llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B (Groq)",
        input_cost_per_m=0.59,
        output_cost_per_m=0.79,
        avg_latency_ms=200,
        priority=1,
        capabilities=ModelCapabilities(
            function_calling=True, streaming=True,
            json_mode=True, context_window=128_000
        ),
    ),
]


class ProviderRegistry:
    """Immutable model catalogue with fast lookup helpers."""

    def __init__(self, models: List[ModelConfig]) -> None:
        self._by_provider: Dict[ModelProvider, Dict[str, ModelConfig]] = {}
        for m in models:
            self._by_provider.setdefault(m.provider, {})[m.model_id] = m

    def get_model(self, provider: ModelProvider, model_id: str) -> Optional[ModelConfig]:
        return self._by_provider.get(provider, {}).get(model_id)

    def list_provider(self, provider: ModelProvider) -> List[ModelConfig]:
        return list(self._by_provider.get(provider, {}).values())

    def list_all(self) -> List[ModelConfig]:
        return [m for p in self._by_provider.values() for m in p.values()]

    def cheapest(self, cap: Optional[ModelCapabilities] = None) -> ModelConfig:
        candidates = self._filter(cap)
        return min(candidates, key=lambda m: m.cost_score)

    def fastest(self, cap: Optional[ModelCapabilities] = None) -> ModelConfig:
        candidates = self._filter(cap)
        return min(candidates, key=lambda m: m.avg_latency_ms)

    def smartest(self, cap: Optional[ModelCapabilities] = None) -> ModelConfig:
        candidates = self._filter(cap)
        return max(candidates, key=lambda m: m.cost_score)  # proxy: higher cost ≈ more capable

    def _filter(self, cap: Optional[ModelCapabilities]) -> List[ModelConfig]:
        all_models = self.list_all()
        if cap is None:
            return all_models
        out = []
        for m in all_models:
            mc = m.capabilities
            if cap.vision and not mc.vision:
                continue
            if cap.long_context and not mc.long_context:
                continue
            if cap.function_calling and not mc.function_calling:
                continue
            out.append(m)
        return out or all_models


PROVIDER_REGISTRY = ProviderRegistry(_MODELS)
