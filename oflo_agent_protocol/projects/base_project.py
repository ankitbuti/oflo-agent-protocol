"""Project container — top-level isolation unit for a multi-agent system.

A Project:
  - Owns an AgentManager (agents, routing, telemetry, audit)
  - Can expose itself via MCP and/or A2A
  - Connects to other projects via A2AClient
  - Provides a simple dev-friendly interface

Usage::

    from oflo_agent_protocol.projects.base_project import Project

    marketing = Project("marketing", cost_budget_usd=10.0)

    @marketing.agent("Copywriter", "You write compelling ad copy.")
    async def _copywriter(): pass

    @marketing.agents["Copywriter"].tool(description="Get brand guidelines")
    async def brand_guidelines() -> str:
        return "Our brand voice is friendly and direct."

    reply = await marketing.ask("Copywriter", "Write a tagline for our new product")

# Cross-project call
    sales = Project("sales")
    result = await marketing.call_project("http://sales-agent:9000", "What are top leads today?")
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from oflo_agent_protocol.audit.guardrails import GuardrailConfig
from oflo_agent_protocol.core.types import RoutingStrategy
from oflo_agent_protocol.managers.agent_manager import AgentManager
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)


class Project:
    """
    Top-level project container.

    Each project maps to a real-world business domain (marketing, sales, trading…)
    and has isolated agents, budgets, and audit logs.
    """

    _registry: Dict[str, "Project"] = {}

    def __init__(
        self,
        project_id: str,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        cost_budget_usd: Optional[float] = None,
        token_budget: Optional[int] = None,
        audit_dir: Optional[str] = None,
        guardrail_config: Optional[GuardrailConfig] = None,
    ) -> None:
        self.project_id = project_id
        self._manager = AgentManager(
            project_id=project_id,
            strategy=strategy,
            cost_budget_usd=cost_budget_usd,
            token_budget=token_budget,
            audit_dir=audit_dir,
            guardrail_config=guardrail_config,
        )
        self.agents: Dict[str, Any] = {}  # name → BaseAgentV2
        Project._registry[project_id] = self

    # ------------------------------------------------------------------
    # Agent definition
    # ------------------------------------------------------------------

    def agent(
        self,
        name: str,
        system_prompt: str = "You are a helpful assistant.",
        runtime: Optional[BaseRuntime] = None,
        strategy: Optional[RoutingStrategy] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Callable:
        """Decorator — create and register an agent by decorating a placeholder fn."""

        def decorator(fn: Callable) -> Callable:
            a = self._manager.create_agent(
                name=name,
                system_prompt=system_prompt,
                runtime=runtime,
                strategy=strategy,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self.agents[name] = a
            return fn

        return decorator

    def add_agent(
        self,
        name: str,
        system_prompt: str = "You are a helpful assistant.",
        runtime: Optional[BaseRuntime] = None,
        **kwargs: Any,
    ) -> Any:
        """Imperatively create and register an agent, returning the instance."""
        a = self._manager.create_agent(
            name=name,
            system_prompt=system_prompt,
            runtime=runtime,
            **kwargs,
        )
        self.agents[name] = a
        return a

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    async def ask(self, agent_name: str, message: str) -> str:
        return await self._manager.route_message(agent_name, message)

    async def delegate(self, from_agent: str, to_agent: str, message: str) -> str:
        return await self._manager.delegate(from_agent, to_agent, message)

    async def pipeline(self, steps: List[Dict[str, str]]) -> List[str]:
        return await self._manager.chain(steps)

    async def broadcast(self, message: str) -> Dict[str, str]:
        return await self._manager.broadcast(message)

    # ------------------------------------------------------------------
    # Cross-project communication (A2A)
    # ------------------------------------------------------------------

    async def call_project(
        self,
        remote_url: str,
        message: str,
        api_key: Optional[str] = None,
    ) -> str:
        """Send a message to a remote project's A2A endpoint."""
        from oflo_agent_protocol.protocols.a2a.client import A2AClient
        async with A2AClient(remote_url, api_key=api_key) as client:
            task = await client.send_and_wait(message)
            if task.artifacts:
                for part in task.artifacts[0].parts:
                    if hasattr(part, "text"):
                        return part.text
            return task.status.message.text if task.status.message else ""

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def mcp_server(self, port: int = 8080) -> Any:
        return self._manager.expose_as_mcp(port=port)

    def a2a_server(self, port: int = 9000, base_url: str = "") -> Any:
        return self._manager.expose_as_a2a(port=port, base_url=base_url)

    async def serve(self, mcp_port: int = 8080, a2a_port: int = 9000) -> None:
        """Start both MCP and A2A servers."""
        mcp = self.mcp_server(port=mcp_port)
        a2a = self.a2a_server(port=a2a_port)
        await asyncio.gather(mcp.start(), a2a.start())
        logger.info(
            "Project '%s' serving: MCP :%d  A2A :%d",
            self.project_id,
            mcp_port,
            a2a_port,
        )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "agents": self._manager.agents(),
            "telemetry": self._manager.telemetry_summary(),
        }

    # ------------------------------------------------------------------
    # Global registry
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, project_id: str) -> Optional["Project"]:
        return cls._registry.get(project_id)

    @classmethod
    def all_projects(cls) -> List[str]:
        return list(cls._registry.keys())

    def __repr__(self) -> str:
        return f"Project({self.project_id!r}, agents={list(self.agents.keys())})"
