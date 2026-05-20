"""Redis Agent Memory Server integration — shared short-term + persistent memory.

Architecture
────────────
Working Memory (per runtime session)
  • Scoped to a session ID (one session = one AgentManager runtime)
  • Shared across ALL agents in the same project/runtime
  • Auto-summarized when conversation grows long
  • Optional TTL (default: 24 hours)
  • Backed by Redis Streams / Hash

Long-Term Memory (per agent, persistent)
  • Semantic + keyword + hybrid search via vector embeddings
  • Persists across sessions
  • Queryable by user_id, topic, entity, time range
  • Used for: learned preferences, facts, episodic history

Memory Prompt Hydration
  • `hydrate_prompt(session_id, query)` → enriched system prompt
  • Pulls relevant working + long-term memories and injects them

Docs: https://redis.github.io/agent-memory-server/

Usage::

    # Start Redis memory server:
    #   docker run -p 6379:6379 redis:latest
    #   pip install agent-memory-server && agent-memory-server start

    from oflo_agent_protocol.memory.redis_memory import RedisMemoryManager

    mem = RedisMemoryManager(session_id="runtime-abc", project_id="marketing")

    # Store working memory (shared across agents in session)
    await mem.set_working_memory([
        {"role": "user", "content": "I prefer concise answers"},
        {"role": "assistant", "content": "Got it, I'll be brief."},
    ])

    # Store long-term fact
    await mem.add_long_term("User is VP of Sales, prefers data-driven insights",
                             agent_id="analyst", topics=["preferences"])

    # Semantic search
    results = await mem.search("sales preferences", agent_id="analyst")

    # Hydrate a prompt with memory context
    system = await mem.hydrate_prompt("How should I respond to this user?")
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

_DEFAULT_URL = os.getenv("REDIS_MEMORY_URL", "http://localhost:8000")
_DEFAULT_TTL = int(os.getenv("REDIS_MEMORY_TTL", "86400"))  # 24 hours


@dataclass
class MemoryRecord:
    text: str
    memory_id: Optional[str] = None
    memory_type: str = "semantic"
    topics: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    namespace: Optional[str] = None
    score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "memory_type": self.memory_type,
            "topics": self.topics,
            "entities": self.entities,
            **({"id": self.memory_id} if self.memory_id else {}),
            **({"user_id": self.agent_id} if self.agent_id else {}),
            **({"session_id": self.session_id} if self.session_id else {}),
            **({"namespace": self.namespace} if self.namespace else {}),
        }


class RedisMemoryManager:
    """
    Unified Redis-backed memory for all agents in a project runtime.

    Two-tier memory model:
      Working memory  → fast, session-scoped, shared, TTL-limited
      Long-term memory → persistent, semantic search, per-agent
    """

    def __init__(
        self,
        session_id: str,
        project_id: str = "default",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        window_token_limit: int = 4000,
        ttl_seconds: int = _DEFAULT_TTL,
    ) -> None:
        self.session_id = session_id
        self.project_id = project_id
        self._base = (base_url or _DEFAULT_URL).rstrip("/")
        self._api_key = api_key or os.getenv("REDIS_MEMORY_API_KEY", "")
        self._window_token_limit = window_token_limit
        self._ttl = ttl_seconds
        self._session: Optional[aiohttp.ClientSession] = None
        self._available: Optional[bool] = None  # lazily checked

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            async with self._get_session().get(
                f"{self._base}/v1/health", timeout=aiohttp.ClientTimeout(total=2)
            ) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        if not self._available:
            logger.warning(
                "Redis memory server not reachable at %s — falling back to in-memory", self._base
            )
        return self._available

    # ------------------------------------------------------------------
    # Working Memory (session-scoped, shared across all agents)
    # ------------------------------------------------------------------

    async def get_working_memory(self) -> Dict[str, Any]:
        """Retrieve the current session's working memory."""
        if not await self._is_available():
            return {}
        try:
            async with self._get_session().get(
                f"{self._base}/v1/working-memory/{self.session_id}"
            ) as resp:
                if resp.status == 404:
                    return {}
                resp.raise_for_status()
                return await resp.json()
        except Exception as exc:
            logger.warning("get_working_memory error: %s", exc)
            return {}

    async def set_working_memory(
        self,
        messages: List[Dict[str, str]],
        structured: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
    ) -> bool:
        """
        Update the session's working memory with new messages.
        Messages are auto-summarized when they exceed the token window.
        """
        if not await self._is_available():
            return False
        body: Dict[str, Any] = {
            "messages": messages,
            "context": context or f"Project: {self.project_id}",
            "window_token_limit": self._window_token_limit,
            "token_ttl_seconds": self._ttl,
        }
        if structured:
            body["structured"] = structured

        try:
            async with self._get_session().put(
                f"{self._base}/v1/working-memory/{self.session_id}",
                json=body,
            ) as resp:
                resp.raise_for_status()
                return True
        except Exception as exc:
            logger.warning("set_working_memory error: %s", exc)
            return False

    async def append_to_working_memory(
        self, role: str, content: str
    ) -> bool:
        """Append a single turn to working memory (reads then writes)."""
        current = await self.get_working_memory()
        messages = current.get("messages", [])
        messages.append({"role": role, "content": content})
        return await self.set_working_memory(messages)

    async def clear_working_memory(self) -> bool:
        """Delete the entire session working memory."""
        if not await self._is_available():
            return False
        try:
            async with self._get_session().delete(
                f"{self._base}/v1/working-memory/{self.session_id}"
            ) as resp:
                return resp.status in (200, 204, 404)
        except Exception as exc:
            logger.warning("clear_working_memory error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Long-Term Memory (persistent, per-agent)
    # ------------------------------------------------------------------

    async def add_long_term(
        self,
        text: str,
        agent_id: Optional[str] = None,
        memory_type: str = "semantic",
        topics: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Store a persistent memory record.
        Returns the memory ID on success.
        """
        if not await self._is_available():
            return None
        record = MemoryRecord(
            text=text,
            memory_type=memory_type,
            topics=topics or [],
            entities=entities or [],
            agent_id=agent_id,
            session_id=self.session_id,
            namespace=self.project_id,
        )
        try:
            async with self._get_session().post(
                f"{self._base}/v1/long-term-memory/",
                json={"memories": [record.to_dict()]},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                ids = data.get("ids", [])
                return ids[0] if ids else None
        except Exception as exc:
            logger.warning("add_long_term error: %s", exc)
            return None

    async def add_long_term_batch(self, records: List[MemoryRecord]) -> List[str]:
        """Batch insert long-term memory records."""
        if not await self._is_available() or not records:
            return []
        try:
            async with self._get_session().post(
                f"{self._base}/v1/long-term-memory/",
                json={"memories": [r.to_dict() for r in records]},
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("ids", [])
        except Exception as exc:
            logger.warning("add_long_term_batch error: %s", exc)
            return []

    async def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
        search_type: str = "semantic",  # "semantic" | "keyword" | "hybrid"
        topics: Optional[List[str]] = None,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> List[MemoryRecord]:
        """
        Search long-term memory with semantic, keyword, or hybrid search.

        search_type:
          "semantic"  → vector similarity (meaning-based)
          "keyword"   → full-text match (exact words)
          "hybrid"    → combines both
        """
        if not await self._is_available():
            return []

        body: Dict[str, Any] = {
            "text": query,
            "limit": limit,
            "search_type": search_type,
        }
        filters: List[Dict[str, Any]] = []
        if agent_id:
            filters.append({"field": "user_id", "op": "eq", "value": agent_id})
        if self.project_id:
            filters.append({"field": "namespace", "op": "eq", "value": self.project_id})
        if topics:
            filters.append({"field": "topics", "op": "any", "value": topics})
        if min_date:
            filters.append({"field": "created_at", "op": "gt", "value": min_date})
        if max_date:
            filters.append({"field": "created_at", "op": "lt", "value": max_date})
        if filters:
            body["filters"] = {"conditions": filters}

        try:
            async with self._get_session().post(
                f"{self._base}/v1/long-term-memory/search", json=body
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                results = []
                for item in data.get("memories", []):
                    results.append(
                        MemoryRecord(
                            text=item.get("text", ""),
                            memory_id=item.get("id"),
                            memory_type=item.get("memory_type", "semantic"),
                            topics=item.get("topics", []),
                            entities=item.get("entities", []),
                            score=item.get("score", 1.0),
                        )
                    )
                return results
        except Exception as exc:
            logger.warning("search error: %s", exc)
            return []

    async def delete_long_term(self, memory_ids: List[str]) -> bool:
        """Delete long-term memory records by ID."""
        if not await self._is_available():
            return False
        try:
            async with self._get_session().delete(
                f"{self._base}/v1/long-term-memory",
                json={"memory_ids": memory_ids},
            ) as resp:
                return resp.status in (200, 204)
        except Exception as exc:
            logger.warning("delete_long_term error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Memory Prompt Hydration
    # ------------------------------------------------------------------

    async def hydrate_prompt(
        self,
        query: str,
        topics: Optional[List[str]] = None,
        long_term_search: bool = True,
        working_memory: bool = True,
    ) -> str:
        """
        Retrieve a memory-enriched prompt string ready to inject as
        additional system context before an LLM call.

        POST /v1/memory/prompt enriches the query with relevant memories.
        Falls back to constructing manually if server unavailable.
        """
        if not await self._is_available():
            return await self._manual_hydrate(query)

        body: Dict[str, Any] = {
            "query": query,
            "session_id": self.session_id,
            "include_working_memory": working_memory,
            "include_long_term": long_term_search,
            "namespace": self.project_id,
        }
        if topics:
            body["topics"] = topics

        try:
            async with self._get_session().post(
                f"{self._base}/v1/memory/prompt", json=body
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("prompt", "")
        except Exception as exc:
            logger.warning("hydrate_prompt error (%s), using manual fallback", exc)
            return await self._manual_hydrate(query)

    async def _manual_hydrate(self, query: str) -> str:
        """Construct a memory context string without the server."""
        parts: List[str] = []
        wm = await self.get_working_memory()
        msgs = wm.get("messages", [])
        if msgs:
            summary = wm.get("context", "")
            if summary:
                parts.append(f"Session context: {summary}")
            recent = msgs[-6:]  # last 3 turns
            turns = [f"{m['role'].capitalize()}: {m['content']}" for m in recent]
            parts.append("Recent conversation:\n" + "\n".join(turns))
        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Convenience: memory-aware process for BaseAgentV2
    # ------------------------------------------------------------------

    async def enrich_system_prompt(self, base_prompt: str, query: str) -> str:
        """Prepend relevant memory context to a system prompt."""
        context = await self.hydrate_prompt(query)
        if not context:
            return base_prompt
        return f"{base_prompt}\n\n--- Memory Context ---\n{context}\n--- End Context ---"

    def __repr__(self) -> str:
        return (
            f"RedisMemoryManager(session={self.session_id!r}, "
            f"project={self.project_id!r}, url={self._base!r})"
        )


class SharedSessionMemory:
    """
    Context manager that wraps RedisMemoryManager for a runtime session.

    All agents created within the context share the same working memory.
    On exit, the session is optionally cleared.

    Usage::

        async with SharedSessionMemory("session-abc", project_id="marketing") as mem:
            await mem.set_working_memory([{"role": "user", "content": "..."}])
            # agents use mem.session_id to share context
    """

    def __init__(
        self,
        session_id: str,
        project_id: str = "default",
        clear_on_exit: bool = False,
        **kwargs: Any,
    ) -> None:
        self._clear_on_exit = clear_on_exit
        self._mem = RedisMemoryManager(
            session_id=session_id, project_id=project_id, **kwargs
        )

    async def __aenter__(self) -> RedisMemoryManager:
        logger.info("Opened memory session: %s", self._mem.session_id)
        return self._mem

    async def __aexit__(self, *_: Any) -> None:
        if self._clear_on_exit:
            await self._mem.clear_working_memory()
            logger.info("Cleared working memory for session: %s", self._mem.session_id)
        await self._mem.close()
