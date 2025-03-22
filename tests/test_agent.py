"""Tests for the Agent class."""
import pytest
from oflo_agent_protocol import Agent, Tool

@pytest.mark.asyncio
async def test_agent_tool_registration():
    """Test that tools can be registered with an agent."""
    
    async def dummy_tool(msg: str) -> str:
        return msg.upper()

    agent = Agent("test-agent")
    tool = Tool("dummy", dummy_tool, "A dummy tool")
    
    await agent.register_tool(tool)
    assert "dummy" in agent._tools 