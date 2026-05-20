"""Tests for core agent, message, and tool primitives."""
from __future__ import annotations

import pytest

from oflo_agent_protocol.core.agent import BaseAgentV2, ToolDefinition, _python_type_to_json
from oflo_agent_protocol.core.message import CanonicalMessage, ToolCall, ToolResult
from oflo_agent_protocol.core.registry import AgentRegistry
from oflo_agent_protocol.core.types import AgentStatus, MessageRole, TokenUsage

from tests.conftest import StubRuntime


# ── CanonicalMessage ──────────────────────────────────────────────────────────

class TestCanonicalMessage:
    def test_user_factory(self):
        m = CanonicalMessage.user("hello")
        assert m.role == MessageRole.USER
        assert m.content == "hello"

    def test_assistant_factory(self):
        m = CanonicalMessage.assistant("hi there")
        assert m.role == MessageRole.ASSISTANT
        assert m.content == "hi there"

    def test_system_factory(self):
        m = CanonicalMessage.system("You are helpful.")
        assert m.role == MessageRole.SYSTEM

    def test_to_openai_user(self):
        m = CanonicalMessage.user("ping")
        d = m.to_openai()
        assert d["role"] == "user"
        assert d["content"] == "ping"

    def test_to_anthropic_user(self):
        m = CanonicalMessage.user("ping")
        d = m.to_anthropic()
        assert d["role"] == "user"

    def test_to_openai_with_tool_calls(self):
        tc = ToolCall(id="tc1", name="get_price", arguments={"ticker": "AAPL"})
        m = CanonicalMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[tc])
        d = m.to_openai()
        assert d["role"] == "assistant"
        assert d["tool_calls"][0]["function"]["name"] == "get_price"

    def test_from_openai_message_dict(self):
        raw = {"role": "assistant", "content": "pong"}
        m = CanonicalMessage.from_openai(raw)
        assert m.content == "pong"
        assert m.role == MessageRole.ASSISTANT


# ── TokenUsage ────────────────────────────────────────────────────────────────

class TestTokenUsage:
    def test_total_tokens(self):
        u = TokenUsage(prompt_tokens=100, completion_tokens=50)
        assert u.total_tokens == 150

    def test_cost_unknown_model(self):
        u = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        # Unknown model returns 0 without crashing
        assert u.cost_usd(None, "nonexistent-model-xyz") == 0.0  # type: ignore


# ── ToolDefinition ────────────────────────────────────────────────────────────

class TestToolDefinition:
    def test_openai_schema(self):
        td = ToolDefinition(
            name="get_price",
            description="Fetch a stock price",
            parameters={"ticker": {"type": "string", "description": "Ticker symbol"}},
            handler=lambda ticker: {},
            required=["ticker"],
        )
        schema = td.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_price"
        assert "ticker" in schema["function"]["parameters"]["required"]

    def test_anthropic_schema(self):
        td = ToolDefinition(
            name="lookup",
            description="Look something up",
            parameters={"query": {"type": "string"}},
            handler=lambda query: {},
            required=["query"],
        )
        schema = td.to_anthropic_schema()
        assert schema["name"] == "lookup"
        assert "input_schema" in schema


# ── Type helpers ──────────────────────────────────────────────────────────────

class TestTypeHelpers:
    def test_python_type_to_json(self):
        assert _python_type_to_json(str) == "string"
        assert _python_type_to_json(int) == "integer"
        assert _python_type_to_json(float) == "number"
        assert _python_type_to_json(bool) == "boolean"
        assert _python_type_to_json(dict) == "object"
        assert _python_type_to_json(list) == "array"
        assert _python_type_to_json(object) == "string"  # default


# ── BaseAgentV2 ───────────────────────────────────────────────────────────────

class TestBaseAgentV2:
    def test_properties(self, agent):
        assert agent.name == "TestAgent"
        assert agent.status == AgentStatus.INITIALIZING
        assert agent.project_id == "default"
        assert agent.system_prompt == "You are a test assistant."

    def test_tool_decorator(self, agent):
        @agent.tool(description="Return a greeting")
        async def greet(name: str) -> str:
            return f"Hello, {name}"

        assert "greet" in agent._tools
        td = agent._tools["greet"]
        assert td.description == "Return a greeting"
        assert "name" in td.parameters

    def test_register_tool(self, agent):
        async def handler(x: int) -> int:
            return x * 2

        agent.register_tool(
            name="double",
            description="Double a number",
            parameters={"x": {"type": "integer"}},
            handler=handler,
            required=["x"],
        )
        assert "double" in agent._tools

    def test_repr(self, agent):
        r = repr(agent)
        assert "TestAgent" in r
        assert agent.id[:8] in r

    def test_to_dict(self, agent):
        d = agent.to_dict()
        assert d["name"] == "TestAgent"
        assert "tools" in d
        assert "history_length" in d

    def test_clear_history(self, agent):
        agent._history.append(CanonicalMessage.user("test"))
        agent.clear_history()
        assert len(agent._history) == 0

    @pytest.mark.asyncio
    async def test_chat_basic(self, agent):
        reply = await agent.chat("Hello")
        assert reply == "stub reply"
        assert agent.status == AgentStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_chat_appends_to_history(self, agent):
        await agent.chat("Hello")
        # user message + assistant reply
        assert any(m.role == MessageRole.USER for m in agent._history)
        assert any(m.role == MessageRole.ASSISTANT for m in agent._history)

    @pytest.mark.asyncio
    async def test_tool_execution_in_loop(self):
        """Agent loop: LLM returns a tool call, tool executes, LLM returns final text."""
        tool_called_with = {}
        call_count = 0

        async def get_price(ticker: str) -> dict:
            tool_called_with["ticker"] = ticker
            return {"price": 182.5}

        tc = ToolCall(id="tc1", name="get_price", arguments={"ticker": "AAPL"})
        # First runtime call returns tool_calls; subsequent calls return final text
        runtime = StubRuntime(reply="AAPL is $182.50", tool_calls=[tc])
        local_agent = BaseAgentV2(name="TradingAgent", runtime=runtime)
        local_agent.register_tool(
            name="get_price",
            description="Get stock price",
            parameters={"ticker": {"type": "string"}},
            handler=get_price,
            required=["ticker"],
        )

        reply = await local_agent.chat("What is AAPL trading at?")
        assert tool_called_with.get("ticker") == "AAPL"
        assert "182" in reply

    @pytest.mark.asyncio
    async def test_history_trimming(self):
        runtime = StubRuntime()
        agent = BaseAgentV2(name="Trimmer", runtime=runtime, max_history=5)
        for i in range(10):
            agent._history.append(CanonicalMessage.user(f"msg {i}"))
        agent._trim_history()
        assert len(agent._history) <= 5


# ── AgentRegistry ─────────────────────────────────────────────────────────────

class TestAgentRegistry:
    @pytest.mark.asyncio
    async def test_register_and_get(self):
        registry = AgentRegistry("test-project")
        agent = BaseAgentV2(name="Reg1", runtime=StubRuntime())
        await registry.register(agent)

        found = registry.get(agent.id)
        assert found is agent

        by_name = registry.get_by_name("Reg1")
        assert by_name is agent

    @pytest.mark.asyncio
    async def test_unregister(self):
        registry = AgentRegistry("test-project")
        agent = BaseAgentV2(name="Reg2", runtime=StubRuntime())
        await registry.register(agent)
        result = await registry.unregister(agent.id)
        assert result is True
        assert registry.get(agent.id) is None

    @pytest.mark.asyncio
    async def test_active_agents(self):
        registry = AgentRegistry("test-project")
        a1 = BaseAgentV2(name="A1", runtime=StubRuntime())
        a2 = BaseAgentV2(name="A2", runtime=StubRuntime())
        a1._status = AgentStatus.ACTIVE
        a2._status = AgentStatus.INACTIVE
        await registry.register(a1)
        await registry.register(a2)
        active = registry.active_agents()
        assert a1 in active
        assert a2 not in active
