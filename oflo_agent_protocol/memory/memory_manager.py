"""Unified memory manager — in-process (fast) + optional Weaviate (persistent).

Agents use this for semantic search over past conversations and shared knowledge.
The in-memory store is the default; Weaviate is opt-in via environment variables.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    key: str
    content: str
    agent_id: str
    memory_type: str = "episodic"  # episodic | semantic | procedural
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "content": self.content,
            "agent_id": self.agent_id,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class InMemoryStore:
    """Fast in-process memory backed by a list (no external dependencies)."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._store: List[MemoryEntry] = []
        self._max = max_entries

    def add(self, entry: MemoryEntry) -> None:
        if len(self._store) >= self._max:
            self._store.pop(0)
        self._store.append(entry)

    def search(self, query: str, agent_id: Optional[str] = None, limit: int = 10) -> List[MemoryEntry]:
        tokens = set(query.lower().split())
        scored: List[tuple[int, MemoryEntry]] = []
        for entry in self._store:
            if agent_id and entry.agent_id != agent_id:
                continue
            content_lower = entry.content.lower()
            hits = sum(1 for t in tokens if t in content_lower)
            if hits > 0:
                scored.append((hits, entry))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:limit]]

    def get_recent(self, agent_id: str, limit: int = 20) -> List[MemoryEntry]:
        return [e for e in reversed(self._store) if e.agent_id == agent_id][:limit]

    def clear_agent(self, agent_id: str) -> int:
        before = len(self._store)
        self._store = [e for e in self._store if e.agent_id != agent_id]
        return before - len(self._store)


class MemoryManager:
    """
    Unified memory interface — abstracts in-memory and Weaviate storage.

    Usage::

        mem = MemoryManager(project_id="marketing")
        await mem.store(agent_id="abc", content="User prefers concise answers", memory_type="semantic")
        results = await mem.search("concise", agent_id="abc")
    """

    def __init__(self, project_id: str, use_weaviate: bool = False) -> None:
        self.project_id = project_id
        self._local = InMemoryStore()
        self._weaviate: Optional[Any] = None
        if use_weaviate:
            self._try_init_weaviate()

    def _try_init_weaviate(self) -> None:
        import os
        url = os.getenv("WEAVIATE_URL")
        api_key = os.getenv("WEAVIATE_API_KEY")
        if not url:
            logger.warning("WEAVIATE_URL not set, using in-memory only")
            return
        try:
            import weaviate
            auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
            self._weaviate = weaviate.WeaviateClient(
                connection_params=weaviate.connect.ConnectionParams.from_url(url, 50051),
                auth_client_secret=auth,
            )
            self._weaviate.connect()
            self._ensure_schema()
            logger.info("Weaviate connected for project '%s'", self.project_id)
        except Exception as exc:
            logger.warning("Weaviate unavailable (%s) — using in-memory only", exc)
            self._weaviate = None

    def _ensure_schema(self) -> None:
        if self._weaviate is None:
            return
        try:
            col = self._weaviate.collections
            if not col.exists("OfloMemory"):
                import weaviate.classes.config as wc
                col.create(
                    name="OfloMemory",
                    vectorizer_config=wc.Configure.Vectorizer.text2vec_openai(),
                    properties=[
                        wc.Property(name="key", data_type=wc.DataType.TEXT),
                        wc.Property(name="content", data_type=wc.DataType.TEXT),
                        wc.Property(name="agent_id", data_type=wc.DataType.TEXT),
                        wc.Property(name="project_id", data_type=wc.DataType.TEXT),
                        wc.Property(name="memory_type", data_type=wc.DataType.TEXT),
                        wc.Property(name="timestamp", data_type=wc.DataType.DATE),
                    ],
                )
        except Exception as exc:
            logger.warning("Schema init failed: %s", exc)

    async def store(
        self,
        agent_id: str,
        content: str,
        memory_type: str = "episodic",
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        entry_key = key or hashlib.sha256(f"{agent_id}:{content}".encode()).hexdigest()[:16]
        entry = MemoryEntry(
            key=entry_key,
            content=content,
            agent_id=agent_id,
            memory_type=memory_type,
            metadata=metadata or {},
        )
        self._local.add(entry)

        if self._weaviate:
            try:
                col = self._weaviate.collections.get("OfloMemory")
                col.data.insert({
                    "key": entry_key,
                    "content": content,
                    "agent_id": agent_id,
                    "project_id": self.project_id,
                    "memory_type": memory_type,
                    "timestamp": entry.timestamp,
                })
            except Exception as exc:
                logger.warning("Weaviate store failed: %s", exc)

        return entry_key

    async def search(
        self,
        query: str,
        agent_id: Optional[str] = None,
        limit: int = 10,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        if self._weaviate:
            try:
                col = self._weaviate.collections.get("OfloMemory")
                filters = []
                if agent_id:
                    import weaviate.classes.query as wq
                    filters.append(wq.Filter.by_property("agent_id").equal(agent_id))
                result = col.query.near_text(
                    query=query,
                    limit=limit,
                    filters=filters[0] if filters else None,
                )
                entries = []
                for obj in result.objects:
                    p = obj.properties
                    entries.append(MemoryEntry(
                        key=p.get("key", ""),
                        content=p.get("content", ""),
                        agent_id=p.get("agent_id", ""),
                        memory_type=p.get("memory_type", "episodic"),
                        timestamp=str(p.get("timestamp", "")),
                    ))
                return entries
            except Exception as exc:
                logger.warning("Weaviate search failed, falling back local: %s", exc)

        return self._local.search(query, agent_id=agent_id, limit=limit)

    async def get_recent(self, agent_id: str, limit: int = 20) -> List[MemoryEntry]:
        return self._local.get_recent(agent_id, limit=limit)

    async def clear_agent(self, agent_id: str) -> int:
        return self._local.clear_agent(agent_id)
