"""MCP client — call MCP-compatible servers (local or remote)."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Async client for interacting with any MCP-compatible server.

    Works with Oflo MCPServer, Claude Desktop MCP servers, and any
    server that implements the MCP 2024-11 protocol.

    Usage::

        async with MCPClient("http://localhost:8080") as client:
            tools = await client.list_tools()
            result = await client.call_tool("agentname.search", {"query": "AI trends"})
            reply = await client.message("agentname", "Summarise AI trends")
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "MCPClient":
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def initialize(self) -> Dict[str, Any]:
        return await self._rpc("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "oflo-client", "version": "2.0.0"},
        })

    async def list_tools(self, agent: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if agent:
            params["agent"] = agent
        result = await self._rpc("tools/list", params)
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        result = await self._rpc("tools/call", {"name": tool_name, "arguments": arguments})
        content = result.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")
        return result

    async def message(self, agent: str, user_text: str) -> str:
        result = await self._rpc("messages/create", {
            "agent": agent,
            "messages": [{"role": "user", "content": user_text}],
        })
        content = result.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")
        return str(result)

    async def list_agents(self) -> List[Dict[str, Any]]:
        session = self._get_session()
        async with session.get(f"{self._base}/agents", headers=self._headers()) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("agents", [])

    async def health(self) -> Dict[str, Any]:
        session = self._get_session()
        async with session.get(f"{self._base}/health", headers=self._headers()) as resp:
            resp.raise_for_status()
            return await resp.json()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }
        session = self._get_session()
        async with session.post(f"{self._base}/", json=payload, headers=self._headers()) as resp:
            resp.raise_for_status()
            data = await resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h
