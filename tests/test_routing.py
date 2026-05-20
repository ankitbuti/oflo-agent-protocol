"""Tests for SmartRouter, ProviderRegistry, and routing strategies."""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from oflo_agent_protocol.core.types import ModelCapabilities, ModelProvider, RoutingStrategy
from oflo_agent_protocol.routing.llm_router import (
    RouterDecision,
    RoutingRequest,
    SmartRouter,
    _available_providers,
)
from oflo_agent_protocol.routing.providers import PROVIDER_REGISTRY, ProviderRegistry


# ── ProviderRegistry ──────────────────────────────────────────────────────────

class TestProviderRegistry:
    def test_all_models_present(self):
        all_models = PROVIDER_REGISTRY.list_all()
        assert len(all_models) >= 10

    def test_providers_covered(self):
        providers = {m.provider for m in PROVIDER_REGISTRY.list_all()}
        assert ModelProvider.ANTHROPIC in providers
        assert ModelProvider.OPENAI in providers
        assert ModelProvider.GOOGLE in providers
        assert ModelProvider.GROQ in providers
        assert ModelProvider.OPENROUTER in providers

    def test_get_model_by_id(self):
        m = PROVIDER_REGISTRY.get_model(ModelProvider.ANTHROPIC, "claude-sonnet-4-6")
        assert m is not None
        assert m.input_cost_per_m == 3.0

    def test_get_model_missing(self):
        m = PROVIDER_REGISTRY.get_model(ModelProvider.ANTHROPIC, "nonexistent-model")
        assert m is None

    def test_cheapest(self):
        cheap = PROVIDER_REGISTRY.cheapest()
        # Haiku / Gemini Flash / GPT-4o-mini should be among cheapest
        assert cheap.cost_score < 5.0

    def test_fastest(self):
        fast = PROVIDER_REGISTRY.fastest()
        # Should be a sub-500ms model (Groq / Gemini Flash)
        assert fast.avg_latency_ms <= 500

    def test_smartest_has_higher_cost(self):
        smart = PROVIDER_REGISTRY.smartest()
        cheap = PROVIDER_REGISTRY.cheapest()
        assert smart.cost_score >= cheap.cost_score

    def test_filter_by_vision(self):
        cap = ModelCapabilities(vision=True)
        vision_models = PROVIDER_REGISTRY._filter(cap)
        assert all(m.capabilities.vision for m in vision_models)

    def test_filter_by_long_context(self):
        cap = ModelCapabilities(long_context=True)
        models = PROVIDER_REGISTRY._filter(cap)
        assert all(m.capabilities.long_context for m in models)

    def test_list_provider_anthropic(self):
        anthropic_models = PROVIDER_REGISTRY.list_provider(ModelProvider.ANTHROPIC)
        assert len(anthropic_models) >= 3

    def test_cost_score_property(self):
        m = PROVIDER_REGISTRY.get_model(ModelProvider.ANTHROPIC, "claude-sonnet-4-6")
        assert m.cost_score == m.input_cost_per_m + m.output_cost_per_m


# ── _available_providers ──────────────────────────────────────────────────────

class TestAvailableProviders:
    def test_anthropic_detected_via_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            available = _available_providers()
        assert ModelProvider.ANTHROPIC in available

    def test_openai_detected_via_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            available = _available_providers()
        assert ModelProvider.OPENAI in available

    def test_groq_detected_via_env(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk-test"}):
            available = _available_providers()
        assert ModelProvider.GROQ in available

    def test_openrouter_detected_via_env(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "or-test"}):
            available = _available_providers()
        assert ModelProvider.OPENROUTER in available

    def test_fallback_to_anthropic_if_no_keys(self):
        env_copy = {k: v for k, v in os.environ.items()
                    if k not in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                                 "GROQ_API_KEY", "OPENROUTER_API_KEY",
                                 "OLLAMA_HOST", "GOOGLE_API_KEY",
                                 "GOOGLE_APPLICATION_CREDENTIALS")}
        with patch.dict(os.environ, env_copy, clear=True):
            available = _available_providers()
        # Falls back to at least Anthropic
        assert len(available) >= 1


# ── SmartRouter ───────────────────────────────────────────────────────────────

class TestSmartRouter:
    @pytest.fixture
    def anthropic_router(self):
        """Router that only sees Anthropic as available."""
        router = SmartRouter()
        router._available = {ModelProvider.ANTHROPIC}
        return router

    @pytest.fixture
    def all_router(self):
        """Router that sees all providers."""
        router = SmartRouter()
        router._available = {
            ModelProvider.ANTHROPIC, ModelProvider.OPENAI,
            ModelProvider.GROQ, ModelProvider.OPENROUTER,
            ModelProvider.GOOGLE,
        }
        return router

    def test_route_returns_decision(self, anthropic_router):
        decision = anthropic_router.route(RoutingRequest())
        assert isinstance(decision, RouterDecision)
        assert decision.provider == ModelProvider.ANTHROPIC

    def test_cheapest_strategy(self, all_router):
        req = RoutingRequest(strategy=RoutingStrategy.CHEAPEST)
        decision = all_router.route(req)
        all_models = PROVIDER_REGISTRY._filter(req.required_capabilities())
        cheapest_cost = min(m.cost_score for m in all_models
                            if m.provider in all_router._available)
        assert decision.model.cost_score == cheapest_cost

    def test_fastest_strategy(self, all_router):
        req = RoutingRequest(strategy=RoutingStrategy.FASTEST)
        decision = all_router.route(req)
        # Should be a fast model (Groq or Gemini Flash)
        assert decision.model.avg_latency_ms <= 500

    def test_smartest_strategy_high_cost(self, all_router):
        cheap = all_router.route(RoutingRequest(strategy=RoutingStrategy.CHEAPEST))
        smart = all_router.route(RoutingRequest(strategy=RoutingStrategy.SMARTEST))
        assert smart.model.cost_score >= cheap.model.cost_score

    def test_balanced_strategy_returns_valid_model(self, all_router):
        decision = all_router.route(RoutingRequest(strategy=RoutingStrategy.BALANCED))
        assert decision.model is not None

    def test_preferred_provider_respected(self, all_router):
        req = RoutingRequest(preferred_provider=ModelProvider.OPENAI)
        decision = all_router.route(req)
        assert decision.provider == ModelProvider.OPENAI

    def test_excluded_provider_not_chosen(self, all_router):
        req = RoutingRequest(excluded_providers=[ModelProvider.ANTHROPIC])
        decision = all_router.route(req)
        assert decision.provider != ModelProvider.ANTHROPIC

    def test_vision_requirement_filters(self, all_router):
        req = RoutingRequest(need_vision=True)
        decision = all_router.route(req)
        assert decision.model.capabilities.vision is True

    def test_long_context_requirement_filters(self, all_router):
        req = RoutingRequest(need_long_context=True)
        decision = all_router.route(req)
        assert decision.model.capabilities.long_context is True

    def test_fallback_chain_populated(self, all_router):
        decision = all_router.route(RoutingRequest())
        assert isinstance(decision.fallback_chain, list)

    def test_describe(self, anthropic_router):
        d = anthropic_router.describe()
        assert "available_providers" in d
        assert "total_models" in d
        assert d["total_models"] >= 10

    def test_capability_match_strategy(self, all_router):
        req = RoutingRequest(
            strategy=RoutingStrategy.CAPABILITY_MATCH,
            need_vision=True,
            need_long_context=True,
        )
        decision = all_router.route(req)
        assert decision.model.capabilities.vision is True

    def test_max_cost_filter(self, all_router):
        req = RoutingRequest(max_cost_per_m=2.0)
        decision = all_router.route(req)
        assert decision.model.cost_score <= 2.0
