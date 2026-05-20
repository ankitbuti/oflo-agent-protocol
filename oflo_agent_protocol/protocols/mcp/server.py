"""MCP 2024-11 compliant server built on FastAPI.

Exposes an Oflo agent's tools as MCP-compatible endpoints.
Compatible with Claude Desktop, Claude.ai, and any MCP client.

Endpoints:
  GET  /                    → server info + capabilities
  POST /tools/list          → list available tools
  POST /tools/call          → execute a tool
  POST /messages            → send a chat message to the agent
  GET  /health              → health check
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

logger = logging.getLogger(__name__)


class MCPServer:
    """
    Hosts an Oflo v2 agent as an MCP-compatible server.

    Implements the MCP 2024-11 JSON-RPC protocol subset required for
    tool listing and invocation by Claude and other LLM clients.
    """

    MCP_VERSION = "2024-11-05"

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        host: str = "0.0.0.0",
        port: int = 8080,
    ) -> None:
        self._name = name
        self._version = version
        self._host = host
        self._port = port
        self._agents: Dict[str, Any] = {}  # name → BaseAgentV2
        self._app = self._build_app()

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def register_agent(self, agent: Any, agent_name: Optional[str] = None) -> None:
        key = agent_name or agent.name
        self._agents[key] = agent
        logger.info("MCP: registered agent '%s'", key)

    def unregister_agent(self, agent_name: str) -> bool:
        return bool(self._agents.pop(agent_name, None))

    # ------------------------------------------------------------------
    # FastAPI app
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title=f"MCP: {self._name}", docs_url=None, redoc_url=None)

        @app.get("/")
        async def server_info() -> JSONResponse:
            return JSONResponse(content=self._server_capabilities())

        @app.get("/health")
        async def health() -> JSONResponse:
            return JSONResponse(content={
                "status": "healthy",
                "timestamp": time.time(),
                "agents": list(self._agents.keys()),
            })

        @app.post("/")
        async def jsonrpc_handler(request: Request) -> JSONResponse:
            body = await request.json()
            return await self._dispatch(body)

        @app.get("/agents")
        async def list_agents() -> JSONResponse:
            return JSONResponse(content={
                "agents": [
                    {"name": name, **agent.to_dict()}
                    for name, agent in self._agents.items()
                ]
            })

        return app

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, body: Dict[str, Any]) -> JSONResponse:
        req_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})

        handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "messages/create": self._handle_message,
            "ping": lambda p, rid: {"jsonrpc": "2.0", "id": rid, "result": {"pong": True}},
        }

        handler = handlers.get(method)
        if handler is None:
            return JSONResponse(content={
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": "Method not found", "data": method},
            })

        try:
            result = await handler(params, req_id)
            return JSONResponse(content=result)
        except Exception as exc:
            logger.exception("MCP handler error: %s", exc)
            return JSONResponse(content={
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32603, "message": str(exc)},
            })

    # ------------------------------------------------------------------
    # MCP method handlers
    # ------------------------------------------------------------------

    async def _handle_initialize(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": self.MCP_VERSION,
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {},
                    "prompts": {},
                },
                "serverInfo": {"name": self._name, "version": self._version},
            },
        }

    async def _handle_tools_list(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        agent_name = params.get("agent") or (next(iter(self._agents), None))
        tools: List[Dict[str, Any]] = []

        if agent_name and agent_name in self._agents:
            agent = self._agents[agent_name]
            for td in agent._tools.values():
                tools.append({
                    "name": f"{agent_name}.{td.name}",
                    "description": td.description,
                    "inputSchema": {
                        "type": "object",
                        "properties": td.parameters,
                        "required": td.required or [],
                    },
                })
        else:
            for aname, agent in self._agents.items():
                for td in agent._tools.values():
                    tools.append({
                        "name": f"{aname}.{td.name}",
                        "description": td.description,
                        "inputSchema": {
                            "type": "object",
                            "properties": td.parameters,
                            "required": td.required or [],
                        },
                    })

        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

    async def _handle_tools_call(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        tool_name: str = params.get("name", "")
        arguments: Dict[str, Any] = params.get("arguments", {})

        # tool_name format: "agent_name.tool_name" or just "tool_name"
        if "." in tool_name:
            agent_name, fn_name = tool_name.split(".", 1)
        else:
            agent_name = next(iter(self._agents), "")
            fn_name = tool_name

        agent = self._agents.get(agent_name)
        if agent is None:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32602, "message": f"Agent '{agent_name}' not found"},
            }

        td = agent._tools.get(fn_name)
        if td is None:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32602, "message": f"Tool '{fn_name}' not found on agent '{agent_name}'"},
            }

        try:
            import asyncio
            if asyncio.iscoroutinefunction(td.handler):
                result = await td.handler(**arguments)
            else:
                result = td.handler(**arguments)

            content = result if isinstance(result, str) else json.dumps(result)
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {"content": [{"type": "text", "text": content}], "isError": False},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                },
            }

    async def _handle_message(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        agent_name = params.get("agent") or (next(iter(self._agents), None))
        messages = params.get("messages", [])

        if not agent_name or agent_name not in self._agents:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32602, "message": f"Agent '{agent_name}' not found"},
            }

        agent = self._agents[agent_name]
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break

        reply = await agent.chat(user_text)

        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "id": f"msg_{uuid.uuid4().hex[:8]}",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": reply}],
                "model": getattr(agent._runtime, "model_id", "auto"),
                "stop_reason": "end_turn",
            },
        }

    def _server_capabilities(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "version": self._version,
            "mcp_version": self.MCP_VERSION,
            "capabilities": {
                "tools": True,
                "streaming": False,
                "agents": list(self._agents.keys()),
            },
        }

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    async def start(self) -> None:
        config = uvicorn.Config(self._app, host=self._host, port=self._port, log_level="warning")
        server = uvicorn.Server(config)
        asyncio.create_task(server.serve())
        logger.info("MCP server '%s' on http://%s:%d", self._name, self._host, self._port)

    @property
    def app(self) -> FastAPI:
        return self._app
