"""Google A2A-compliant HTTP server (JSON-RPC 2.0 over FastAPI).

Endpoints:
  GET  /.well-known/agent.json       → AgentCard discovery
  POST /                             → JSON-RPC dispatch (tasks/send, tasks/get, tasks/cancel)
  GET  /tasks/{id}/stream            → SSE streaming (tasks/sendSubscribe)

Usage::

    from oflo_agent_protocol.protocols.a2a.server import A2AServer
    from oflo_agent_protocol.protocols.a2a.types import AgentCard, AgentSkill

    server = A2AServer(
        card=AgentCard(name="Sales Agent", description="...", url="http://localhost:9000"),
        agent=my_agent,
        host="0.0.0.0",
        port=9000,
    )
    await server.start()
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Callable, Dict, Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

from oflo_agent_protocol.protocols.a2a.types import (
    A2AMessage,
    A2ATask,
    AgentCard,
    Artifact,
    TaskStatusUpdate,
    TextPart,
    jsonrpc_error,
    jsonrpc_response,
    A2A_ERRORS,
)

logger = logging.getLogger(__name__)


class A2AServer:
    """
    Hosts an Oflo agent as a Google A2A-compliant HTTP server.

    The `agent` parameter must have a `chat(text: str) -> str` coroutine.
    Any BaseAgentV2 instance satisfies this.
    """

    def __init__(
        self,
        card: AgentCard,
        agent: Any,
        host: str = "0.0.0.0",
        port: int = 9000,
    ) -> None:
        self._card = card
        self._agent = agent
        self._host = host
        self._port = port
        self._tasks: Dict[str, A2ATask] = {}
        self._app = self._build_app()

    # ------------------------------------------------------------------
    # FastAPI app construction
    # ------------------------------------------------------------------

    def _build_app(self) -> FastAPI:
        app = FastAPI(title=f"A2A: {self._card.name}", docs_url=None, redoc_url=None)

        @app.get("/.well-known/agent.json")
        async def agent_card() -> JSONResponse:
            return JSONResponse(content=self._card.to_dict())

        @app.post("/")
        async def jsonrpc_dispatch(request: Request) -> JSONResponse:
            body = await request.json()
            return await self._dispatch(body)

        @app.get("/tasks/{task_id}/stream")
        async def task_stream(task_id: str) -> StreamingResponse:
            return StreamingResponse(
                self._sse_stream(task_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )

        return app

    # ------------------------------------------------------------------
    # JSON-RPC dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, body: Dict[str, Any]) -> JSONResponse:
        req_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})

        handlers = {
            "tasks/send": self._handle_send,
            "tasks/get": self._handle_get,
            "tasks/cancel": self._handle_cancel,
            "tasks/sendSubscribe": self._handle_send_subscribe,
        }

        handler = handlers.get(method)
        if handler is None:
            code, msg = A2A_ERRORS["method_not_found"]
            return JSONResponse(content=jsonrpc_error(code, msg, req_id=req_id))

        try:
            result = await handler(params, req_id)
            return JSONResponse(content=result)
        except Exception as exc:
            logger.exception("A2A handler error: %s", exc)
            code, msg = A2A_ERRORS["internal_error"]
            return JSONResponse(content=jsonrpc_error(code, f"{msg}: {exc}", req_id=req_id))

    # ------------------------------------------------------------------
    # Method handlers
    # ------------------------------------------------------------------

    async def _handle_send(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        task_id = params.get("id", str(uuid.uuid4()))
        session_id = params.get("sessionId")
        message_data = params.get("message", {})

        user_msg = A2AMessage.text_message(
            role="user", text=self._extract_text(message_data)
        )
        task = A2ATask(
            task_id=task_id,
            session_id=session_id,
            status=TaskStatusUpdate(state="working"),
            history=[user_msg],
        )
        self._tasks[task_id] = task

        # Process with the underlying agent
        try:
            reply_text = await self._agent.chat(user_msg.text)
            agent_msg = A2AMessage.text_message(role="agent", text=reply_text)
            task.history.append(agent_msg)
            artifact = Artifact(name="reply", parts=[TextPart(text=reply_text)])
            task.artifacts.append(artifact)
            task.status = TaskStatusUpdate(
                state="completed",
                message=agent_msg,
            )
        except Exception as exc:
            task.status = TaskStatusUpdate(
                state="failed",
                message=A2AMessage.text_message(role="agent", text=str(exc)),
            )

        return jsonrpc_response(task.to_dict(), req_id=req_id)

    async def _handle_get(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        task_id = params.get("id")
        task = self._tasks.get(task_id)
        if task is None:
            code, msg = A2A_ERRORS["task_not_found"]
            return jsonrpc_error(code, msg, req_id=req_id)
        return jsonrpc_response(task.to_dict(), req_id=req_id)

    async def _handle_cancel(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        task_id = params.get("id")
        task = self._tasks.get(task_id)
        if task is None:
            code, msg = A2A_ERRORS["task_not_found"]
            return jsonrpc_error(code, msg, req_id=req_id)
        if task.status.state in ("completed", "failed", "canceled"):
            code, msg = A2A_ERRORS["task_not_cancelable"]
            return jsonrpc_error(code, msg, req_id=req_id)
        task.status = TaskStatusUpdate(state="canceled")
        return jsonrpc_response(task.to_dict(), req_id=req_id)

    async def _handle_send_subscribe(self, params: Dict[str, Any], req_id: Any) -> Dict[str, Any]:
        # Return task ID so caller can connect to /tasks/{id}/stream
        task_id = params.get("id", str(uuid.uuid4()))
        return jsonrpc_response({"taskId": task_id, "streamUrl": f"/tasks/{task_id}/stream"}, req_id=req_id)

    # ------------------------------------------------------------------
    # SSE streaming
    # ------------------------------------------------------------------

    async def _sse_stream(self, task_id: str) -> AsyncIterator[str]:
        task = self._tasks.get(task_id)
        if task is None:
            yield f"data: {json.dumps({'error': 'task not found'})}\n\n"
            return

        # Send current status
        yield f"data: {json.dumps(task.to_dict())}\n\n"

        # Poll until done (simple approach; replace with asyncio.Event for production)
        for _ in range(60):
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps({'status': task.status.state})}\n\n"
            if task.status.state in ("completed", "failed", "canceled"):
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(message_data: Dict[str, Any]) -> str:
        parts = message_data.get("parts", [])
        texts = []
        for p in parts:
            if isinstance(p, dict) and p.get("type") == "text":
                texts.append(p.get("text", ""))
            elif isinstance(p, str):
                texts.append(p)
        return " ".join(texts) or message_data.get("text", "")

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    async def start(self) -> None:
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        asyncio.create_task(server.serve())
        logger.info(
            "A2A server '%s' listening on http://%s:%d",
            self._card.name,
            self._host,
            self._port,
        )

    @property
    def app(self) -> FastAPI:
        return self._app
