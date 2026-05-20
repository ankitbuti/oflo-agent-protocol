"""LangGraph runtime — builds and executes StateGraph agent workflows."""
from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from oflo_agent_protocol.core.message import CanonicalMessage
from oflo_agent_protocol.core.types import MessageRole, ModelProvider, TokenUsage
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    messages: List[BaseMessage]
    next: str


def _to_lc(msg: CanonicalMessage) -> BaseMessage:
    if msg.role == MessageRole.USER:
        return HumanMessage(content=msg.content)
    if msg.role == MessageRole.SYSTEM:
        return SystemMessage(content=msg.content)
    return AIMessage(content=msg.content)


def _from_lc(msg: BaseMessage) -> CanonicalMessage:
    if isinstance(msg, HumanMessage):
        return CanonicalMessage.user(msg.content if isinstance(msg.content, str) else str(msg.content))
    if isinstance(msg, SystemMessage):
        return CanonicalMessage.system(msg.content if isinstance(msg.content, str) else str(msg.content))
    return CanonicalMessage.assistant(msg.content if isinstance(msg.content, str) else str(msg.content))


def _make_llm(provider: ModelProvider, model_id: str) -> Any:
    """Build a LangChain chat model from provider + model_id."""
    if provider == ModelProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_id,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        )
    if provider == ModelProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_id,
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    if provider == ModelProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_id,
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
        )
    if provider == ModelProvider.GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_id,
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
        )
    raise ValueError(f"LangGraph runtime does not support provider {provider}")


class LangGraphRuntime(BaseRuntime):
    """
    Wraps a single LangGraph StateGraph as a runtime.

    For simple completions, builds a two-node graph: system_inject → model.
    For complex multi-agent graphs, use `LangGraphOrchestrator` instead.
    """

    def __init__(
        self,
        provider: ModelProvider = ModelProvider.ANTHROPIC,
        model_id: str = "claude-sonnet-4-6",
    ) -> None:
        self._provider = provider
        self._model_id = model_id
        self._llm = _make_llm(provider, model_id)
        self._graph = self._build_simple_graph()

    @property
    def provider_name(self) -> str:
        return f"langgraph/{self._provider.value}"

    def _build_simple_graph(self):
        llm = self._llm

        def model_node(state: GraphState) -> dict:
            response = llm.invoke(state["messages"])
            return {"messages": state["messages"] + [response]}

        def route(state: GraphState) -> str:
            return END

        builder: StateGraph = StateGraph(GraphState)
        builder.add_node("model", model_node)
        builder.set_entry_point("model")
        builder.add_conditional_edges("model", route, {END: END})
        return builder.compile()

    async def complete(
        self,
        messages: List[CanonicalMessage],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[CanonicalMessage, TokenUsage]:
        lc_messages: List[BaseMessage] = []
        if system:
            lc_messages.append(SystemMessage(content=system))
        lc_messages.extend(_to_lc(m) for m in messages if m.role != MessageRole.SYSTEM)

        result = await self._graph.ainvoke({"messages": lc_messages, "next": ""})
        last_msg = result["messages"][-1]
        reply = _from_lc(last_msg)
        # LangGraph doesn't surface token counts by default
        usage = TokenUsage()
        return reply, usage

    async def stream(
        self,
        messages: List[CanonicalMessage],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        lc_messages: List[BaseMessage] = []
        if system:
            lc_messages.append(SystemMessage(content=system))
        lc_messages.extend(_to_lc(m) for m in messages if m.role != MessageRole.SYSTEM)

        async for event in self._graph.astream({"messages": lc_messages, "next": ""}):
            for node_output in event.values():
                msgs = node_output.get("messages", [])
                if msgs:
                    last = msgs[-1]
                    content = last.content if isinstance(last.content, str) else str(last.content)
                    yield content


class LangGraphOrchestrator:
    """
    Multi-agent orchestrator using LangGraph for complex pipelines.

    Usage:
        orch = LangGraphOrchestrator()
        orch.add_node("researcher", researcher_fn)
        orch.add_node("writer", writer_fn)
        orch.add_edge("researcher", "writer")
        result = await orch.run("Write about AI agents")
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Any] = {}
        self._edges: List[tuple[str, str]] = []
        self._entry: Optional[str] = None

    def add_node(self, name: str, fn: Any) -> "LangGraphOrchestrator":
        self._nodes[name] = fn
        if self._entry is None:
            self._entry = name
        return self

    def add_edge(self, source: str, target: str) -> "LangGraphOrchestrator":
        self._edges.append((source, target))
        return self

    def set_entry(self, name: str) -> "LangGraphOrchestrator":
        self._entry = name
        return self

    def compile(self):
        builder: StateGraph = StateGraph(GraphState)
        for name, fn in self._nodes.items():
            builder.add_node(name, fn)
        if self._entry:
            builder.set_entry_point(self._entry)
        for src, tgt in self._edges:
            builder.add_edge(src, tgt if tgt != "END" else END)
        # Last node → END
        last_node = self._edges[-1][1] if self._edges else self._entry
        if last_node and last_node != END:
            builder.add_edge(last_node, END)
        return builder.compile()

    async def run(self, initial_message: str, shared_memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        graph = self.compile()
        state: GraphState = {
            "messages": [HumanMessage(content=initial_message)],
            "next": self._entry or "",
        }
        return await graph.ainvoke(state)
