"""
Oflo Agent Protocol v2
======================
A modular, auditable, multi-LLM agent protocol with MCP, Google A2A,
LangGraph, OpenAI Agent SDK, and smart routing.

Quick start
───────────
    from oflo_agent_protocol import Project, ClaudeRuntime

    marketing = Project("marketing", cost_budget_usd=5.0)
    analyst = marketing.add_agent("Analyst", system_prompt="You are a data analyst.")

    @analyst.tool(description="Look up campaign performance")
    async def get_campaign_metrics(campaign_id: str) -> dict:
        return {"clicks": 1200, "conversions": 48, "ctr": 0.04}

    reply = await marketing.ask("Analyst", "How did campaign A1 perform?")
    print(reply)

Multi-LLM routing
─────────────────
    from oflo_agent_protocol import route, RoutingStrategy
    decision = route(strategy=RoutingStrategy.CHEAPEST, need_function_calling=True)
    print(decision.provider, decision.model_id)

A2A cross-project call
──────────────────────
    result = await marketing.call_project("http://sales-agent:9000", "Top leads today?")

MCP server
──────────
    mcp = marketing.mcp_server(port=8080)
    await mcp.start()
"""

__version__ = "2.0.0"
__author__ = "Ankit Buti"
__email__ = "ankit@oflo.ai"

# ── Core ─────────────────────────────────────────────────────────────────────
from oflo_agent_protocol.core.agent import BaseAgentV2
from oflo_agent_protocol.core.message import CanonicalMessage
from oflo_agent_protocol.core.registry import AgentRegistry
from oflo_agent_protocol.core.types import (
    AgentStatus,
    AuditRecord,
    MessageRole,
    ModelCapabilities,
    ModelConfig,
    ModelProvider,
    RoutingStrategy,
    TaskStatus,
    TokenUsage,
)

# ── Projects & Managers ───────────────────────────────────────────────────────
from oflo_agent_protocol.projects.base_project import Project
from oflo_agent_protocol.managers.agent_manager import AgentManager

# ── Routing ───────────────────────────────────────────────────────────────────
from oflo_agent_protocol.routing.llm_router import (
    RouterDecision,
    RoutingRequest,
    SmartRouter,
    get_router,
    route,
)
from oflo_agent_protocol.routing.providers import PROVIDER_REGISTRY

# ── Runtimes (lazy — imported only when used to avoid missing SDK errors) ──────
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

try:
    from oflo_agent_protocol.runtimes.claude_runtime import ClaudeRuntime
except ImportError:
    ClaudeRuntime = None  # type: ignore

try:
    from oflo_agent_protocol.runtimes.openai_runtime import (
        GroqRuntime,
        OllamaRuntime,
        OpenAIRuntime,
    )
except ImportError:
    OpenAIRuntime = GroqRuntime = OllamaRuntime = None  # type: ignore

try:
    from oflo_agent_protocol.runtimes.openrouter_runtime import OpenRouterRuntime
except ImportError:
    OpenRouterRuntime = None  # type: ignore

try:
    from oflo_agent_protocol.runtimes.daytona_runtime import (
        DaytonaSandbox,
        DaytonaSandboxRuntime,
        daytona_session,
    )
except ImportError:
    DaytonaSandbox = DaytonaSandboxRuntime = daytona_session = None  # type: ignore

try:
    from oflo_agent_protocol.runtimes.langgraph_runtime import (
        LangGraphOrchestrator,
        LangGraphRuntime,
    )
except ImportError:
    LangGraphRuntime = LangGraphOrchestrator = None  # type: ignore

# ── Protocols ─────────────────────────────────────────────────────────────────
try:
    from oflo_agent_protocol.protocols.mcp.server import MCPServer
    from oflo_agent_protocol.protocols.mcp.client import MCPClient
    from oflo_agent_protocol.protocols.a2a.server import A2AServer
    from oflo_agent_protocol.protocols.a2a.client import A2AClient
except ImportError:
    MCPServer = MCPClient = A2AServer = A2AClient = None  # type: ignore

from oflo_agent_protocol.protocols.a2a.types import (
    A2ATask,
    AgentCard,
    AgentSkill,
    Artifact,
)

# ── Audit & Safety ────────────────────────────────────────────────────────────
from oflo_agent_protocol.audit.audit_logger import AuditLogger
from oflo_agent_protocol.audit.telemetry import Telemetry
from oflo_agent_protocol.audit.guardrails import GuardrailConfig, Guardrails

# ── Memory ────────────────────────────────────────────────────────────────────
from oflo_agent_protocol.memory.memory_manager import MemoryManager

try:
    from oflo_agent_protocol.memory.redis_memory import (
        RedisMemoryManager,
        SharedSessionMemory,
    )
except ImportError:
    RedisMemoryManager = SharedSessionMemory = None  # type: ignore

# ── Connectors (Composio + future integrations) ───────────────────────────────
try:
    from oflo_agent_protocol.connectors.composio_connector import (
        ComposioConnector,
        ComposioToolKit,
    )
except ImportError:
    ComposioConnector = ComposioToolKit = None  # type: ignore

# ── Voice ─────────────────────────────────────────────────────────────────────
try:
    from oflo_agent_protocol.voice.voice_agent import VoiceAgent, VoiceSessionStats
    from oflo_agent_protocol.voice.audio_interface import (
        BufferedAudioInterface,
        FileAudioInterface,
        NullAudioInterface,
        WebSocketAudioInterface,
    )
except ImportError:
    VoiceAgent = VoiceSessionStats = None  # type: ignore
    BufferedAudioInterface = FileAudioInterface = None  # type: ignore
    NullAudioInterface = WebSocketAudioInterface = None  # type: ignore

# ── Backward compat with v1 ───────────────────────────────────────────────────
# Legacy imports still work: from oflo_agent_protocol import BaseAgent, Message
try:
    from oflo_agent_protocol.agent import BaseAgent, Message, FunctionDefinition  # type: ignore
except ImportError:
    pass

__all__ = [
    # v2 core
    "BaseAgentV2",
    "CanonicalMessage",
    "AgentRegistry",
    "Project",
    "AgentManager",
    # types
    "AgentStatus",
    "AuditRecord",
    "MessageRole",
    "ModelCapabilities",
    "ModelConfig",
    "ModelProvider",
    "RoutingStrategy",
    "TaskStatus",
    "TokenUsage",
    # routing
    "RouterDecision",
    "RoutingRequest",
    "SmartRouter",
    "PROVIDER_REGISTRY",
    "get_router",
    "route",
    # runtimes
    "BaseRuntime",
    "ClaudeRuntime",
    "OpenAIRuntime",
    "GroqRuntime",
    "OllamaRuntime",
    "OpenRouterRuntime",
    "DaytonaSandbox",
    "DaytonaSandboxRuntime",
    "daytona_session",
    "LangGraphRuntime",
    "LangGraphOrchestrator",
    # protocols
    "MCPServer",
    "MCPClient",
    "A2AServer",
    "A2AClient",
    "A2ATask",
    "AgentCard",
    "AgentSkill",
    "Artifact",
    # audit
    "AuditLogger",
    "Telemetry",
    "GuardrailConfig",
    "Guardrails",
    # memory
    "MemoryManager",
    "RedisMemoryManager",
    "SharedSessionMemory",
    # connectors
    "ComposioConnector",
    "ComposioToolKit",
    # voice
    "VoiceAgent",
    "VoiceSessionStats",
    "BufferedAudioInterface",
    "FileAudioInterface",
    "NullAudioInterface",
    "WebSocketAudioInterface",
]
