"""Core types, enums, and dataclasses shared across the entire protocol."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStatus(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    WORKING = "working"
    IDLE = "idle"
    PAUSED = "paused"
    INACTIVE = "inactive"
    ERROR = "error"
    TERMINATED = "terminated"


class ModelProvider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    GROQ = "groq"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    COHERE = "cohere"


class TaskStatus(Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class RoutingStrategy(Enum):
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    SMARTEST = "smartest"
    BALANCED = "balanced"
    CAPABILITY_MATCH = "capability_match"


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self, provider: ModelProvider, model: str) -> float:
        from oflo_agent_protocol.routing.providers import PROVIDER_REGISTRY
        cfg = PROVIDER_REGISTRY.get_model(provider, model)
        if cfg is None:
            return 0.0
        return round(
            (self.prompt_tokens / 1_000_000) * cfg.input_cost_per_m
            + (self.completion_tokens / 1_000_000) * cfg.output_cost_per_m
            + (self.cache_read_tokens / 1_000_000) * cfg.cache_read_cost_per_m
            + (self.cache_write_tokens / 1_000_000) * cfg.cache_write_cost_per_m,
            8,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class AuditRecord:
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    agent_id: str = ""
    agent_name: str = ""
    project_id: str = ""
    provider: str = ""
    model: str = ""
    routing_strategy: str = ""
    prompt_hash: str = ""
    token_usage: Optional[TokenUsage] = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    success: bool = True
    error: Optional[str] = None
    guardrail_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "project_id": self.project_id,
            "provider": self.provider,
            "model": self.model,
            "routing_strategy": self.routing_strategy,
            "prompt_hash": self.prompt_hash,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "success": self.success,
            "error": self.error,
            "guardrail_flags": self.guardrail_flags,
            "metadata": self.metadata,
        }
        if self.token_usage:
            d["token_usage"] = self.token_usage.to_dict()
        return d


@dataclass
class ModelCapabilities:
    vision: bool = False
    long_context: bool = False
    function_calling: bool = True
    streaming: bool = True
    json_mode: bool = False
    batch: bool = False
    context_window: int = 8192


@dataclass
class ModelConfig:
    provider: ModelProvider
    model_id: str
    display_name: str
    input_cost_per_m: float
    output_cost_per_m: float
    cache_read_cost_per_m: float = 0.0
    cache_write_cost_per_m: float = 0.0
    avg_latency_ms: float = 1000.0
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    priority: int = 5  # 1=highest

    @property
    def cost_score(self) -> float:
        return self.input_cost_per_m + self.output_cost_per_m

    @property
    def speed_score(self) -> float:
        return 1.0 / max(self.avg_latency_ms, 1.0)
