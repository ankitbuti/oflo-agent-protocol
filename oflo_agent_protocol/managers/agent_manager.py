"""AgentManager — per-project orchestrator of agent lifecycle, routing, and comms.

Each project (marketing, sales, trading, etc.) gets its own AgentManager instance.
The manager:
  - Creates and registers agents
  - Routes messages to the right agent (by capability, name, or ID)
  - Delegates tasks between agents
  - Exposes the project as both an MCP server and an A2A server
  - Maintains shared project-level telemetry and audit logging
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Type

from oflo_agent_protocol.audit.audit_logger import AuditLogger
from oflo_agent_protocol.audit.guardrails import GuardrailConfig
from oflo_agent_protocol.audit.telemetry import Telemetry
from oflo_agent_protocol.core.agent import BaseAgentV2
from oflo_agent_protocol.core.message import CanonicalMessage
from oflo_agent_protocol.core.registry import AgentRegistry
from oflo_agent_protocol.core.types import AgentStatus, ModelProvider, RoutingStrategy
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

try:
    from oflo_agent_protocol.memory.redis_memory import RedisMemoryManager
    _HAS_REDIS = True
except ImportError:
    RedisMemoryManager = None  # type: ignore
    _HAS_REDIS = False

try:
    from oflo_agent_protocol.connectors.composio_connector import ComposioConnector
    _HAS_COMPOSIO = True
except ImportError:
    ComposioConnector = None  # type: ignore
    _HAS_COMPOSIO = False

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Per-project agent orchestrator.

    Typical usage::

        mgr = AgentManager(project_id="marketing", cost_budget_usd=5.0)

        analyst = mgr.create_agent("Analyst", system_prompt="You are a data analyst.")
        writer = mgr.create_agent("Writer", system_prompt="You write marketing copy.")

        @analyst.tool(description="Look up campaign metrics")
        async def get_metrics(campaign_id: str) -> dict:
            return {"clicks": 1200, "conversions": 48}

        reply = await mgr.route_message("Analyst", "Summarise last week's campaigns")

        # Delegate: Analyst → Writer
        await mgr.delegate("Analyst", "Writer", "Turn the analysis into a blog post")
    """

    def __init__(
        self,
        project_id: str,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        cost_budget_usd: Optional[float] = None,
        token_budget: Optional[int] = None,
        audit_dir: Optional[str] = None,
        guardrail_config: Optional[GuardrailConfig] = None,
        redis_memory: Optional[Any] = None,
        redis_base_url: Optional[str] = None,
        session_id: Optional[str] = None,
        composio_connector: Optional[Any] = None,
        composio_api_key: Optional[str] = None,
        composio_user_id: Optional[str] = None,
    ) -> None:
        self.project_id = project_id
        self._strategy = strategy
        self._registry = AgentRegistry(project_id)
        self._audit = AuditLogger(project_id, log_dir=audit_dir)
        self._telemetry = Telemetry(
            cost_budget_usd=cost_budget_usd,
            token_budget=token_budget,
        )
        self._guardrail_config = guardrail_config or GuardrailConfig()
        self._logger = logging.getLogger(f"manager.{project_id}")

        # Optional shared Redis memory — one session per manager instance
        self._redis: Optional[Any] = redis_memory
        if self._redis is None and redis_base_url and _HAS_REDIS:
            self._redis = RedisMemoryManager(
                session_id=session_id or project_id,
                project_id=project_id,
                base_url=redis_base_url,
            )
        self._session_id: str = session_id or project_id

        # Optional Composio connector — wires 300+ app tools into agents
        self._composio: Optional[Any] = composio_connector
        if self._composio is None and (composio_api_key or os.getenv("COMPOSIO_API_KEY")) and _HAS_COMPOSIO:
            import os as _os
            self._composio = ComposioConnector(
                api_key=composio_api_key or _os.getenv("COMPOSIO_API_KEY"),
                user_id=composio_user_id or project_id,
            )

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def create_agent(
        self,
        name: str,
        system_prompt: str = "You are a helpful assistant.",
        runtime: Optional[BaseRuntime] = None,
        strategy: Optional[RoutingStrategy] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        composio_toolkits: Optional[List[str]] = None,
        composio_actions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseAgentV2:
        """Create, register, and return a new agent.

        If *composio_toolkits* or *composio_actions* are provided (and a
        ComposioConnector is configured on this manager), the corresponding
        Composio tools are injected into the agent automatically.
        """
        agent = BaseAgentV2(
            name=name,
            system_prompt=system_prompt,
            runtime=runtime,
            strategy=strategy or self._strategy,
            project_id=self.project_id,
            audit_logger=self._audit,
            telemetry=self._telemetry,
            guardrail_config=self._guardrail_config,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        asyncio.get_event_loop().run_until_complete(self._registry.register(agent))
        agent._status = AgentStatus.ACTIVE
        self._logger.info("Created agent '%s' (%s)", name, agent.id[:8])

        if (composio_toolkits or composio_actions) and self._composio:
            try:
                n = asyncio.get_event_loop().run_until_complete(
                    self._composio.inject_into_agent(
                        agent,
                        toolkits=composio_toolkits,
                        actions=composio_actions,
                    )
                )
                self._logger.info(
                    "Injected %d Composio tool(s) into agent '%s'", n, name
                )
            except Exception as exc:
                self._logger.warning("Composio tool injection failed: %s", exc)

        return agent

    async def register_agent(self, agent: BaseAgentV2) -> None:
        agent._project_id = self.project_id
        agent._audit = self._audit
        agent._telemetry = self._telemetry
        await self._registry.register(agent)

    async def remove_agent(self, agent_id: str) -> bool:
        return await self._registry.unregister(agent_id)

    # ------------------------------------------------------------------
    # Routing & delegation
    # ------------------------------------------------------------------

    async def route_message(self, agent_name: str, message: str) -> str:
        """Send a message to a named agent, return text reply.

        If a Redis memory manager is configured, enriches the agent's working
        memory with relevant context before routing and persists the exchange.
        """
        agent = self._registry.get_by_name(agent_name)
        if agent is None:
            raise ValueError(f"No agent named '{agent_name}' in project '{self.project_id}'")

        if self._redis:
            try:
                enriched = await self._redis.enrich_system_prompt(agent.system_prompt, message)
                if enriched != agent.system_prompt:
                    agent._system_prompt = enriched
                await self._redis.append_to_working_memory("user", message)
            except Exception as exc:
                self._logger.warning("Redis memory unavailable: %s", exc)

        reply = await agent.chat(message)

        if self._redis:
            try:
                await self._redis.append_to_working_memory("assistant", reply)
                await self._redis.add_long_term(
                    text=f"Q: {message}\nA: {reply}",
                    agent_id=agent.id,
                    memory_type="conversation",
                )
            except Exception as exc:
                self._logger.warning("Redis memory write failed: %s", exc)

        return reply

    async def route_to_capable(self, message: str, capability_hint: str = "") -> str:
        """Route to the first active agent (optionally filtered by capability)."""
        agents = self._registry.active_agents()
        if not agents:
            raise RuntimeError(f"No active agents in project '{self.project_id}'")
        target = agents[0]  # Round-robin / first-match; extend as needed
        return await target.chat(message)

    async def delegate(self, from_agent: str, to_agent: str, message: str) -> str:
        """Agent-to-agent delegation within the project."""
        source = self._registry.get_by_name(from_agent)
        target = self._registry.get_by_name(to_agent)
        if source is None:
            raise ValueError(f"Source agent '{from_agent}' not found")
        if target is None:
            raise ValueError(f"Target agent '{to_agent}' not found")

        # Get source agent's perspective first
        context = await source.chat(
            f"Prepare a handoff summary for this task: {message}"
        )
        # Deliver to target with context
        result = await target.chat(
            f"You are receiving a delegated task.\n\nContext from {from_agent}:\n{context}\n\nTask: {message}"
        )
        self._logger.info("Delegation %s → %s complete", from_agent, to_agent)
        return result

    async def broadcast(self, message: str) -> Dict[str, str]:
        """Broadcast a message to all active agents and collect replies."""
        agents = self._registry.active_agents()
        tasks = [agent.chat(message) for agent in agents]
        replies = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            agent.name: (reply if isinstance(reply, str) else f"ERROR: {reply}")
            for agent, reply in zip(agents, replies)
        }

    async def chain(self, messages: List[Dict[str, str]]) -> List[str]:
        """
        Execute a pipeline: [{"agent": "A", "message": "..."}, {"agent": "B", "use_previous": True}].
        If `use_previous` is True, prepends the previous agent's reply to the message.
        """
        replies: List[str] = []
        prev = ""
        for step in messages:
            agent_name = step["agent"]
            msg = step.get("message", "")
            if step.get("use_previous") and prev:
                msg = f"Previous output:\n{prev}\n\nYour task: {msg}"
            reply = await self.route_message(agent_name, msg)
            replies.append(reply)
            prev = reply
        return replies

    # ------------------------------------------------------------------
    # Protocol exposure
    # ------------------------------------------------------------------

    def expose_as_mcp(self, port: int = 8080) -> Any:
        """Return a configured MCPServer exposing all project agents."""
        from oflo_agent_protocol.protocols.mcp.server import MCPServer
        srv = MCPServer(name=self.project_id, port=port)
        for agent in self._registry._agents.values():
            srv.register_agent(agent)
        return srv

    def expose_as_a2a(self, port: int = 9000, base_url: str = "") -> Any:
        """Return a configured A2AServer for the project's primary agent."""
        from oflo_agent_protocol.protocols.a2a.server import A2AServer
        from oflo_agent_protocol.protocols.a2a.types import AgentCard
        agents = self._registry.active_agents()
        if not agents:
            raise RuntimeError("No active agents to expose via A2A")
        primary = agents[0]
        url = base_url or f"http://localhost:{port}"
        card = AgentCard(
            name=self.project_id,
            description=f"Oflo project: {self.project_id}",
            url=url,
        )
        return A2AServer(card=card, agent=primary, port=port)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def telemetry_summary(self) -> Dict[str, Any]:
        return self._telemetry.summary()

    async def audit_report(self, limit: int = 50) -> List[Dict[str, Any]]:
        return await self._audit.query(limit=limit)

    def agents(self) -> List[Dict[str, Any]]:
        return self._registry.list_agents()

    def on_cost_alert(self, callback: Any) -> None:
        self._telemetry.on_alert(callback)

    # ------------------------------------------------------------------
    # Shared memory helpers
    # ------------------------------------------------------------------

    async def memory_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search long-term Redis memory for relevant past exchanges."""
        if not self._redis:
            return []
        try:
            return await self._redis.search(query, limit=limit)
        except Exception as exc:
            self._logger.warning("Memory search failed: %s", exc)
            return []

    async def clear_session_memory(self) -> None:
        """Clear working memory for the current session."""
        if self._redis:
            try:
                await self._redis.clear_working_memory()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Composio helpers
    # ------------------------------------------------------------------

    async def inject_composio_tools(
        self,
        agent_name: str,
        toolkits: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        search: Optional[str] = None,
    ) -> int:
        """Inject Composio tools into an already-created agent by name."""
        if not self._composio:
            raise RuntimeError(
                "No ComposioConnector configured. Pass composio_api_key= to AgentManager."
            )
        agent = self._registry.get_by_name(agent_name)
        if agent is None:
            raise ValueError(f"Agent '{agent_name}' not found")
        return await self._composio.inject_into_agent(
            agent, toolkits=toolkits, actions=actions, search=search
        )

    async def connect_composio_app(
        self,
        app_name: str,
        callback_url: Optional[str] = None,
    ) -> str:
        """Initiate an OAuth / API-key flow for a Composio app."""
        if not self._composio:
            raise RuntimeError("No ComposioConnector configured.")
        return await self._composio.connect_app(app_name, callback_url=callback_url)

    async def list_composio_apps(self) -> List[Dict[str, Any]]:
        """List all connected Composio apps for this project's user."""
        if not self._composio:
            return []
        return await self._composio.list_connected_apps()

    async def execute_composio_action(
        self,
        action_slug: str,
        params: Dict[str, Any],
    ) -> Any:
        """Execute a Composio action directly (bypasses the LLM)."""
        if not self._composio:
            raise RuntimeError("No ComposioConnector configured.")
        return await self._composio.execute_action(action_slug, params)

    def __repr__(self) -> str:
        tags = []
        if self._redis:
            tags.append("+redis")
        if self._composio:
            tags.append("+composio")
        suffix = " " + " ".join(tags) if tags else ""
        return f"AgentManager(project={self.project_id!r}, agents={len(self._registry)}{suffix})"
