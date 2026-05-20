"""BaseAgentV2 — the core agent class for the v2 protocol.

Key design properties:
- Runtime-injected: the LLM backend (ClaudeRuntime, OpenAIRuntime, etc.) is
  injected at construction time and swappable without subclassing.
- Router-aware: if no runtime is supplied the SmartRouter selects one.
- Fully auditable: every call emits an AuditRecord.
- Tool-first: tools declared as plain async callables via `@agent.tool`.
- Multi-turn memory: conversation history kept in-process (pluggable memory store).
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from oflo_agent_protocol.audit.audit_logger import AuditLogger
from oflo_agent_protocol.audit.guardrails import GuardrailConfig, Guardrails, GuardrailResult
from oflo_agent_protocol.audit.telemetry import Telemetry, timed_call
from oflo_agent_protocol.core.message import CanonicalMessage, ToolCall, ToolResult
from oflo_agent_protocol.core.types import (
    AgentStatus,
    AuditRecord,
    MessageRole,
    ModelProvider,
    RoutingStrategy,
    TokenUsage,
)
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)


class ToolDefinition:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
        required: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler
        self.required = required or []

    def to_openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }

    def to_anthropic_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }


class BaseAgentV2:
    """
    Core Oflo v2 agent.

    Quick start::

        agent = BaseAgentV2(
            name="Analyst",
            system_prompt="You are a financial analyst.",
            runtime=ClaudeRuntime(),
        )

        @agent.tool(description="Fetch stock price")
        async def get_price(ticker: str) -> dict:
            return {"ticker": ticker, "price": 150.0}

        reply = await agent.chat("What is AAPL trading at?")
    """

    def __init__(
        self,
        name: str,
        system_prompt: str = "You are a helpful assistant.",
        runtime: Optional[BaseRuntime] = None,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        project_id: str = "default",
        audit_logger: Optional[AuditLogger] = None,
        telemetry: Optional[Telemetry] = None,
        guardrail_config: Optional[GuardrailConfig] = None,
        max_history: int = 50,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> None:
        self._id = str(uuid.uuid4())
        self._name = name
        self._system_prompt = system_prompt
        self._runtime: Optional[BaseRuntime] = runtime
        self._strategy = strategy
        self._project_id = project_id
        self._audit = audit_logger
        self._telemetry = telemetry
        self._guardrails = Guardrails()
        self._guardrail_config = guardrail_config or GuardrailConfig()
        self._max_history = max_history
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._status = AgentStatus.INITIALIZING
        self._history: List[CanonicalMessage] = []
        self._tools: Dict[str, ToolDefinition] = {}
        self._logger = logging.getLogger(f"agent.{name}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    def project_id(self) -> str:
        return self._project_id

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def history(self) -> List[CanonicalMessage]:
        return list(self._history)

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def tool(
        self,
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        required: Optional[List[str]] = None,
    ) -> Callable:
        """Decorator to register an async function as a tool."""

        def decorator(fn: Callable) -> Callable:
            # Infer parameter schema from type hints if not provided
            import inspect
            hints = {
                k: v for k, v in (fn.__annotations__ or {}).items()
                if k != "return"
            }
            props = parameters or {
                k: {"type": _python_type_to_json(v), "description": k}
                for k, v in hints.items()
            }
            td = ToolDefinition(
                name=fn.__name__,
                description=description or fn.__doc__ or fn.__name__,
                parameters=props,
                handler=fn,
                required=required or list(props.keys()),
            )
            self._tools[fn.__name__] = td
            return fn

        return decorator

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable,
        required: Optional[List[str]] = None,
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            required=required,
        )

    def _tool_schemas(self) -> List[Dict[str, Any]]:
        return [td.to_openai_schema() for td in self._tools.values()]

    # ------------------------------------------------------------------
    # Core chat interface
    # ------------------------------------------------------------------

    async def chat(self, user_message: str, **meta: Any) -> str:
        """Single-turn convenience method. Returns assistant text."""
        reply = await self.process(CanonicalMessage.user(user_message, **meta))
        return reply.content

    async def process(self, message: CanonicalMessage) -> CanonicalMessage:
        """
        Full processing pipeline:
        1. Append to history
        2. Route to runtime
        3. Execute tool calls if any (agentic loop, max 5 iterations)
        4. Run guardrails
        5. Emit audit record
        """
        self._status = AgentStatus.WORKING
        self._history.append(message)
        self._trim_history()

        runtime = await self._get_runtime()
        tools = self._tool_schemas() if self._tools else None

        start = time.monotonic()
        token_usage = TokenUsage()
        error_msg: Optional[str] = None
        reply: Optional[CanonicalMessage] = None

        try:
            # Agentic loop — execute tools if requested
            for _iteration in range(6):
                raw_reply, usage = await runtime.complete(
                    messages=[m for m in self._history if m.role != MessageRole.SYSTEM],
                    system=self._system_prompt,
                    tools=tools,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                token_usage.prompt_tokens += usage.prompt_tokens
                token_usage.completion_tokens += usage.completion_tokens
                token_usage.cache_read_tokens += usage.cache_read_tokens
                token_usage.cache_write_tokens += usage.cache_write_tokens

                if not raw_reply.tool_calls:
                    reply = raw_reply
                    break

                # Execute tools and loop
                self._history.append(raw_reply)
                tool_results = await self._execute_tools(raw_reply.tool_calls)
                tool_msg = CanonicalMessage(
                    role=MessageRole.TOOL,
                    content="",
                    tool_results=tool_results,
                )
                self._history.append(tool_msg)

            reply = reply or raw_reply

        except Exception as exc:
            self._logger.exception("Runtime error: %s", exc)
            error_msg = str(exc)
            reply = CanonicalMessage.assistant(
                "I encountered an error processing your request. Please try again."
            )
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            self._status = AgentStatus.ACTIVE

        # Guardrails
        gr: GuardrailResult = self._guardrails.check(reply, self._guardrail_config)
        if gr.scrubbed_content:
            reply = CanonicalMessage.assistant(gr.scrubbed_content)
        if gr.blocked:
            reply = CanonicalMessage.assistant("[Response blocked by content policy.]")

        # Append to history
        self._history.append(reply)

        # Audit
        provider_name = getattr(runtime, "provider_name", "unknown")
        model_id = getattr(runtime, "model_id", "unknown")
        record = AuditRecord(
            agent_id=self._id,
            agent_name=self._name,
            project_id=self._project_id,
            provider=provider_name,
            model=model_id,
            routing_strategy=self._strategy.value,
            prompt_hash=AuditLogger.hash_prompt(message.content),
            token_usage=token_usage,
            latency_ms=latency_ms,
            cost_usd=token_usage.cost_usd(
                ModelProvider(provider_name.split("/")[-1]) if "/" not in provider_name else ModelProvider.ANTHROPIC,
                model_id,
            ),
            success=error_msg is None,
            error=error_msg,
            guardrail_flags=gr.flags,
        )

        if self._audit:
            await self._audit.log(record)
        if self._telemetry:
            await self._telemetry.record(record)

        return reply

    async def _execute_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        results: List[ToolResult] = []
        for tc in tool_calls:
            td = self._tools.get(tc.name)
            if td is None:
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content={"error": f"Tool '{tc.name}' not found"},
                        is_error=True,
                    )
                )
                continue
            try:
                import asyncio
                if asyncio.iscoroutinefunction(td.handler):
                    result = await td.handler(**tc.arguments)
                else:
                    result = td.handler(**tc.arguments)
                results.append(ToolResult(tool_call_id=tc.id, name=tc.name, content=result))
            except Exception as exc:
                self._logger.error("Tool %s failed: %s", tc.name, exc)
                results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        name=tc.name,
                        content={"error": str(exc)},
                        is_error=True,
                    )
                )
        return results

    async def _get_runtime(self) -> BaseRuntime:
        if self._runtime:
            return self._runtime
        # Auto-select via SmartRouter
        from oflo_agent_protocol.routing.llm_router import RoutingRequest, SmartRouter
        from oflo_agent_protocol.runtimes.claude_runtime import ClaudeRuntime
        from oflo_agent_protocol.runtimes.openai_runtime import OpenAIRuntime

        router = SmartRouter()
        decision = router.route(RoutingRequest(strategy=self._strategy))
        provider = decision.provider
        model_id = decision.model_id

        if provider == ModelProvider.ANTHROPIC:
            self._runtime = ClaudeRuntime(model_id=model_id)
        elif provider in (ModelProvider.OPENAI, ModelProvider.GROQ):
            from oflo_agent_protocol.runtimes.openai_runtime import GroqRuntime
            self._runtime = GroqRuntime(model_id=model_id) if provider == ModelProvider.GROQ else OpenAIRuntime(model_id=model_id)
        else:
            self._runtime = ClaudeRuntime()  # final fallback

        self._logger.info("Auto-selected runtime: %s/%s", provider.value, model_id)
        return self._runtime

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history:
            # Keep system messages + recent messages
            system = [m for m in self._history if m.role == MessageRole.SYSTEM]
            non_system = [m for m in self._history if m.role != MessageRole.SYSTEM]
            self._history = system + non_system[-(self._max_history - len(system)):]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear_history(self) -> None:
        self._history = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "name": self._name,
            "project_id": self._project_id,
            "status": self._status.value,
            "tools": list(self._tools.keys()),
            "history_length": len(self._history),
            "provider": getattr(self._runtime, "provider_name", "auto"),
        }

    def __repr__(self) -> str:
        return f"BaseAgentV2(name={self._name!r}, id={self._id[:8]}, status={self._status.value})"


def _python_type_to_json(hint: Any) -> str:
    import typing
    mapping = {str: "string", int: "integer", float: "number", bool: "boolean", dict: "object", list: "array"}
    return mapping.get(hint, "string")
