"""Pytest configuration and shared fixtures for oflo-agent-protocol v2."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from oflo_agent_protocol.core.agent import BaseAgentV2, ToolDefinition
from oflo_agent_protocol.core.message import CanonicalMessage, ToolCall, ToolResult
from oflo_agent_protocol.core.types import MessageRole, TokenUsage
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime


# ── Stub runtime ──────────────────────────────────────────────────────────────

class StubRuntime(BaseRuntime):
    """Deterministic in-process runtime — no real LLM calls."""

    def __init__(self, reply: str = "stub reply", tool_calls: Optional[List[ToolCall]] = None) -> None:
        self._reply = reply
        self._tool_calls = tool_calls or []
        self.calls: List[Dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def model_id(self) -> str:
        return "stub-model"

    async def complete(
        self,
        messages,
        system=None,
        tools=None,
        max_tokens=4096,
        temperature=0.7,
        **kwargs,
    ):
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        msg = CanonicalMessage(
            role=MessageRole.ASSISTANT,
            content=self._reply,
            tool_calls=self._tool_calls if self._tool_calls else None,
        )
        # After first call with tool_calls, clear them so the loop terminates
        self._tool_calls = []
        return msg, TokenUsage(prompt_tokens=10, completion_tokens=5)

    async def stream(self, messages, system=None, tools=None, **kwargs):
        for chunk in self._reply.split():
            yield chunk + " "

    async def health_check(self) -> bool:
        return True


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def stub_runtime():
    return StubRuntime()


@pytest.fixture
def agent(stub_runtime):
    return BaseAgentV2(
        name="TestAgent",
        system_prompt="You are a test assistant.",
        runtime=stub_runtime,
    )


@pytest.fixture
def event_loop():
    """Single event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
