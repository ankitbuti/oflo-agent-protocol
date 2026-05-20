"""Google A2A client — call remote A2A-compliant agents."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Dict, Optional

import aiohttp

from oflo_agent_protocol.protocols.a2a.types import (
    A2AMessage,
    A2ATask,
    AgentCard,
    jsonrpc_request,
)

logger = logging.getLogger(__name__)


class A2AClient:
    """
    Async client for calling a remote Google A2A-compliant agent.

    Usage::

        async with A2AClient("http://remote-agent:9000") as client:
            card = await client.discover()
            task = await client.send("Analyse Q4 revenue trends")
            print(task.artifacts[0].parts[0].text)
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "A2AClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def discover(self) -> AgentCard:
        """Fetch the remote agent's AgentCard."""
        data = await self._get("/.well-known/agent.json")
        return AgentCard(
            name=data.get("name", ""),
            description=data.get("description", ""),
            url=data.get("url", self._base),
            version=data.get("version", "1.0.0"),
        )

    async def send(
        self,
        message: str,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> A2ATask:
        """Send a task and wait for completion."""
        task_id = task_id or str(uuid.uuid4())
        payload = jsonrpc_request(
            "tasks/send",
            params={
                "id": task_id,
                "sessionId": session_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": message}],
                },
                "metadata": metadata or {},
            },
        )
        result = await self._post("/", payload)
        return self._parse_task(result.get("result", {}))

    async def get_task(self, task_id: str) -> A2ATask:
        """Get the current state of a task."""
        payload = jsonrpc_request("tasks/get", params={"id": task_id})
        result = await self._post("/", payload)
        return self._parse_task(result.get("result", {}))

    async def cancel_task(self, task_id: str) -> A2ATask:
        """Cancel a running task."""
        payload = jsonrpc_request("tasks/cancel", params={"id": task_id})
        result = await self._post("/", payload)
        return self._parse_task(result.get("result", {}))

    async def stream(self, task_id: str) -> AsyncIterator[Dict[str, Any]]:
        """Subscribe to SSE updates for a task."""
        session = self._get_session()
        url = f"{self._base}/tasks/{task_id}/stream"
        async with session.get(url, headers=self._headers()) as resp:
            async for line in resp.content:
                line = line.decode().strip()
                if line.startswith("data:"):
                    try:
                        yield json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        pass

    async def send_and_wait(
        self,
        message: str,
        poll_interval: float = 0.5,
        timeout: float = 120.0,
    ) -> A2ATask:
        """Send a task and poll until it completes."""
        task = await self.send(message)
        elapsed = 0.0
        while task.status.state not in ("completed", "failed", "canceled"):
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            if elapsed >= timeout:
                raise TimeoutError(f"A2A task {task.task_id} timed out after {timeout}s")
            task = await self.get_task(task.task_id)
        return task

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    async def _get(self, path: str) -> Dict[str, Any]:
        session = self._get_session()
        async with session.get(f"{self._base}{path}", headers=self._headers()) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        session = self._get_session()
        async with session.post(
            f"{self._base}{path}", json=payload, headers=self._headers()
        ) as resp:
            resp.raise_for_status()
            return await resp.json()

    @staticmethod
    def _parse_task(data: Dict[str, Any]) -> A2ATask:
        from oflo_agent_protocol.protocols.a2a.types import (
            A2AMessage, Artifact, TaskStatusUpdate, TextPart
        )
        status_data = data.get("status", {})
        status_msg_data = status_data.get("message")
        status_msg = None
        if status_msg_data:
            status_msg = A2AMessage.text_message(
                role=status_msg_data.get("role", "agent"),
                text=" ".join(
                    p.get("text", "") for p in status_msg_data.get("parts", [])
                    if isinstance(p, dict)
                ),
            )
        status = TaskStatusUpdate(
            state=status_data.get("state", "submitted"),
            message=status_msg,
            timestamp=status_data.get("timestamp", ""),
        )

        artifacts = []
        for a in data.get("artifacts", []):
            parts = [TextPart(text=p.get("text", "")) for p in a.get("parts", []) if isinstance(p, dict)]
            artifacts.append(
                Artifact(name=a.get("name", ""), parts=parts, artifact_id=a.get("artifactId", str(uuid.uuid4())))
            )

        return A2ATask(
            task_id=data.get("id", str(uuid.uuid4())),
            session_id=data.get("sessionId"),
            status=status,
            artifacts=artifacts,
        )
