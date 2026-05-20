"""OpenAI Agent SDK adapter — wrap Oflo agents as OpenAI Agents SDK agents.

Requires: pip install openai-agents

OpenAI Agents SDK (formerly Swarm) uses:
  - Agent(name, instructions, tools, model)
  - Runner.run(agent, messages)
  - @function_tool decorator

This adapter bridges Oflo's BaseAgentV2 ↔ OpenAI Agents SDK.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def _import_sdk() -> Any:
    try:
        import agents
        return agents
    except ImportError as e:
        raise ImportError(
            "openai-agents package is required. Install with: pip install openai-agents"
        ) from e


class OpenAIAgentsAdapter:
    """
    Converts Oflo BaseAgentV2 instances to OpenAI Agents SDK Agent objects
    and runs them via the SDK's Runner.

    Usage::

        adapter = OpenAIAgentsAdapter()
        sdk_agent = adapter.to_sdk_agent(my_oflo_agent)
        result = await adapter.run(sdk_agent, "What is the weather?")
    """

    def to_sdk_agent(self, oflo_agent: Any, model: str = "gpt-4o-mini") -> Any:
        """Convert an Oflo agent to an OpenAI Agents SDK Agent."""
        sdk = _import_sdk()

        # Convert Oflo tools to SDK function tools
        sdk_tools = []
        for td in oflo_agent._tools.values():
            sdk_tools.append(self._make_function_tool(sdk, td))

        agent = sdk.Agent(
            name=oflo_agent.name,
            instructions=oflo_agent.system_prompt,
            tools=sdk_tools,
            model=model,
        )
        # Attach reference to Oflo agent for audit passthrough
        agent._oflo_agent = oflo_agent
        return agent

    @staticmethod
    def _make_function_tool(sdk: Any, tool_def: Any) -> Any:
        """Wrap an Oflo ToolDefinition as an OpenAI Agents SDK function tool."""
        import inspect

        handler = tool_def.handler
        # The SDK expects a regular or async function
        # We just return the handler directly since SDK handles async

        # Build a wrapper with correct signature metadata
        async def _tool_wrapper(**kwargs: Any) -> Any:
            import asyncio
            if asyncio.iscoroutinefunction(handler):
                return await handler(**kwargs)
            return handler(**kwargs)

        _tool_wrapper.__name__ = tool_def.name
        _tool_wrapper.__doc__ = tool_def.description

        # Annotate with schema
        return sdk.function_tool(
            _tool_wrapper,
            name_override=tool_def.name,
            description_override=tool_def.description,
        )

    async def run(self, sdk_agent: Any, message: str) -> str:
        """Run an SDK agent with a single user message."""
        sdk = _import_sdk()
        result = await sdk.Runner.run(sdk_agent, message)
        return result.final_output if hasattr(result, "final_output") else str(result)

    async def run_thread(self, sdk_agent: Any, messages: List[Dict[str, str]]) -> str:
        """Run an SDK agent with a message history."""
        sdk = _import_sdk()
        input_list = [sdk.MessageInputItem(role=m["role"], content=m["content"]) for m in messages]
        result = await sdk.Runner.run(sdk_agent, input_list)
        return result.final_output if hasattr(result, "final_output") else str(result)

    def create_handoff(self, agents: List[Any]) -> Any:
        """Create an SDK handoff between multiple SDK agents."""
        sdk = _import_sdk()
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents for a handoff")
        # Wire handoffs: each agent can hand off to the next
        for i, agent in enumerate(agents[:-1]):
            next_agent = agents[i + 1]
            handoff = sdk.handoff(next_agent)
            agent.handoffs = agent.handoffs or []
            agent.handoffs.append(handoff)
        return agents[0]


class MultiAgentOrchestrator:
    """
    Runs a pipeline of OpenAI SDK agents, passing output between them.

    Each step can be an Oflo agent or a raw SDK Agent.
    """

    def __init__(self, adapter: Optional[OpenAIAgentsAdapter] = None) -> None:
        self._adapter = adapter or OpenAIAgentsAdapter()
        self._steps: List[Any] = []

    def add_step(self, agent: Any, model: str = "gpt-4o-mini") -> "MultiAgentOrchestrator":
        """Add an Oflo agent or SDK agent as a pipeline step."""
        from oflo_agent_protocol.core.agent import BaseAgentV2
        if isinstance(agent, BaseAgentV2):
            sdk_agent = self._adapter.to_sdk_agent(agent, model=model)
        else:
            sdk_agent = agent
        self._steps.append(sdk_agent)
        return self

    async def run(self, initial_message: str) -> List[str]:
        """Execute the pipeline sequentially."""
        outputs: List[str] = []
        message = initial_message
        for step in self._steps:
            reply = await self._adapter.run(step, message)
            outputs.append(reply)
            message = reply  # chain output as next input
        return outputs
