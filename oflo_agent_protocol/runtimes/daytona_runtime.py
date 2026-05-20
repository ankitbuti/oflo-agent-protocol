"""Daytona sandbox runtime — executes agent tool calls inside isolated sandboxes.

Every agent can be assigned a Daytona sandbox so that:
  • Python code execution is fully isolated (no host contamination)
  • File system operations are sandboxed
  • Network calls are auditable
  • Snapshots allow instant rollback between tool calls
  • Each runtime session gets its own sandbox (created on first use, destroyed on close)

Docs: https://www.daytona.io/docs/en/python-sdk/

Usage::

    from oflo_agent_protocol.runtimes.daytona_runtime import (
        DaytonaSandboxRuntime, SandboxedAgentMixin
    )

    # Wrap Claude runtime with Daytona sandboxing for tool execution
    runtime = DaytonaSandboxRuntime(
        llm_runtime=ClaudeRuntime(),          # any BaseRuntime for LLM calls
        snapshot_id="python3.11-data-science", # optional: pre-built environment
        auto_destroy=True,
    )

    agent = BaseAgentV2("CodeAgent", runtime=runtime)

    @agent.tool(description="Execute Python code safely")
    async def run_python(code: str) -> dict:
        return await runtime.exec_code(code)

    @agent.tool(description="Run a shell command")
    async def run_command(command: str) -> str:
        return await runtime.exec_command(command)
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from oflo_agent_protocol.core.message import CanonicalMessage
from oflo_agent_protocol.core.types import TokenUsage
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)


class DaytonaSandbox:
    """
    Wraps a single Daytona sandbox with a clean async API.

    Lifecycle: create → use → [snapshot] → destroy
    """

    def __init__(
        self,
        sandbox: Any,  # daytona.Sandbox object
        daytona_client: Any,
    ) -> None:
        self._sandbox = sandbox
        self._daytona = daytona_client
        self._sandbox_id: str = getattr(sandbox, "id", "unknown")

    @property
    def sandbox_id(self) -> str:
        return self._sandbox_id

    # ------------------------------------------------------------------
    # Process execution
    # ------------------------------------------------------------------

    async def exec(self, command: str, timeout: int = 30) -> str:
        """Execute a shell command and return stdout."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self._sandbox.process.exec(command)
            )
            return getattr(response, "result", str(response))
        except Exception as exc:
            logger.error("Sandbox exec failed: %s", exc)
            raise RuntimeError(f"Sandbox command failed: {exc}") from exc

    async def exec_python(self, code: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Execute Python code in the sandbox.
        Returns {"output": str, "error": str | None, "exit_code": int}.
        """
        escaped = code.replace("'", "'\"'\"'")
        cmd = f"python3 -c '{escaped}'"
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self._sandbox.process.exec(cmd)
            )
            result = getattr(response, "result", "")
            exit_code = getattr(response, "exit_code", 0)
            return {"output": result, "error": None, "exit_code": exit_code}
        except Exception as exc:
            return {"output": "", "error": str(exc), "exit_code": 1}

    async def run_code_interpreter(self, code: str) -> Dict[str, Any]:
        """
        Use Daytona's code interpreter for richer Python execution
        (supports rich output, DataFrames, plots).
        """
        try:
            loop = asyncio.get_event_loop()
            # Try code_interpreter module if available
            if hasattr(self._sandbox, "code_interpreter"):
                result = await loop.run_in_executor(
                    None, lambda: self._sandbox.code_interpreter.exec(code)
                )
                return {
                    "output": getattr(result, "result", str(result)),
                    "error": getattr(result, "error", None),
                    "exit_code": 0,
                }
            # Fallback to process.exec
            return await self.exec_python(code)
        except Exception as exc:
            return {"output": "", "error": str(exc), "exit_code": 1}

    # ------------------------------------------------------------------
    # File system
    # ------------------------------------------------------------------

    async def write_file(self, path: str, content: str) -> None:
        """Write a file in the sandbox."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, lambda: self._sandbox.fs.upload_file(path, content.encode())
        )

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None, lambda: self._sandbox.fs.download_file(path)
        )
        return data.decode() if isinstance(data, bytes) else str(data)

    async def list_files(self, path: str = "/") -> List[str]:
        """List files in a sandbox directory."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._sandbox.fs.list_files(path)
            )
            return [str(f) for f in (result or [])]
        except Exception as exc:
            logger.warning("list_files error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Git
    # ------------------------------------------------------------------

    async def git_clone(self, repo_url: str, path: str = "/workspace") -> str:
        """Clone a git repository into the sandbox."""
        return await self.exec(f"git clone {repo_url} {path}")

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    async def snapshot(self, name: str) -> str:
        """Take a snapshot of the sandbox state. Returns snapshot ID."""
        try:
            loop = asyncio.get_event_loop()
            snap = await loop.run_in_executor(
                None, lambda: self._sandbox.snapshot.create(name=name)
            )
            snap_id = getattr(snap, "id", str(snap))
            logger.info("Sandbox snapshot: %s → %s", name, snap_id)
            return snap_id
        except AttributeError:
            logger.warning("Snapshot not supported by this sandbox version")
            return ""

    # ------------------------------------------------------------------
    # Install packages
    # ------------------------------------------------------------------

    async def pip_install(self, *packages: str) -> str:
        """Install Python packages in the sandbox."""
        pkgs = " ".join(packages)
        return await self.exec(f"pip install -q {pkgs}")

    async def destroy(self) -> None:
        """Remove the sandbox (irreversible)."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, lambda: self._daytona.remove(self._sandbox)
            )
            logger.info("Destroyed sandbox %s", self._sandbox_id)
        except Exception as exc:
            logger.warning("Sandbox destroy error: %s", exc)


class DaytonaSandboxRuntime(BaseRuntime):
    """
    Decorator runtime that runs an LLM (any BaseRuntime) for completions
    while routing tool execution through a Daytona sandbox.

    The sandbox is created lazily on first tool call and destroyed
    when the runtime is closed (or `auto_destroy=True`).
    """

    def __init__(
        self,
        llm_runtime: BaseRuntime,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        target: str = "us",
        snapshot_id: Optional[str] = None,
        auto_destroy: bool = True,
        sandbox_timeout: int = 300,  # seconds
    ) -> None:
        self._llm = llm_runtime
        self._api_key = api_key or os.getenv("DAYTONA_API_KEY", "")
        self._api_url = api_url or os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api")
        self._target = target
        self._snapshot_id = snapshot_id
        self._auto_destroy = auto_destroy
        self._sandbox_timeout = sandbox_timeout
        self._sandbox: Optional[DaytonaSandbox] = None
        self._daytona_client: Optional[Any] = None

    @property
    def provider_name(self) -> str:
        return f"daytona+{self._llm.provider_name}"

    @property
    def model_id(self) -> str:
        return getattr(self._llm, "model_id", "unknown")

    # ------------------------------------------------------------------
    # Sandbox lifecycle
    # ------------------------------------------------------------------

    async def _get_daytona_client(self) -> Any:
        if self._daytona_client:
            return self._daytona_client
        try:
            from daytona import AsyncDaytona, DaytonaConfig
        except ImportError:
            raise ImportError(
                "daytona package required. Install with: pip install daytona"
            )
        config = DaytonaConfig(
            api_key=self._api_key,
            api_url=self._api_url,
            target=self._target,
        )
        self._daytona_client = AsyncDaytona(config)
        return self._daytona_client

    async def get_sandbox(self) -> DaytonaSandbox:
        """Get (or lazily create) the agent's sandbox."""
        if self._sandbox:
            return self._sandbox

        try:
            client = await self._get_daytona_client()
            logger.info("Creating Daytona sandbox (snapshot=%s)…", self._snapshot_id)

            create_kwargs: Dict[str, Any] = {}
            if self._snapshot_id:
                create_kwargs["snapshot"] = self._snapshot_id

            raw_sandbox = await client.create(**create_kwargs)
            self._sandbox = DaytonaSandbox(sandbox=raw_sandbox, daytona_client=client)
            logger.info("Daytona sandbox ready: %s", self._sandbox.sandbox_id)
            return self._sandbox
        except Exception as exc:
            logger.error("Failed to create Daytona sandbox: %s", exc)
            raise

    async def exec_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code in the sandbox — callable from agent tools."""
        sb = await self.get_sandbox()
        return await sb.run_code_interpreter(code)

    async def exec_command(self, command: str) -> str:
        """Execute a shell command in the sandbox — callable from agent tools."""
        sb = await self.get_sandbox()
        return await sb.exec(command)

    async def close(self) -> None:
        """Shut down the sandbox and Daytona client."""
        if self._sandbox and self._auto_destroy:
            await self._sandbox.destroy()
            self._sandbox = None
        if self._daytona_client:
            try:
                await self._daytona_client.close()
            except Exception:
                pass
            self._daytona_client = None

    # ------------------------------------------------------------------
    # BaseRuntime interface (LLM calls are delegated to inner runtime)
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: List[CanonicalMessage],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> tuple[CanonicalMessage, TokenUsage]:
        return await self._llm.complete(
            messages, system=system, tools=tools,
            max_tokens=max_tokens, temperature=temperature, **kwargs
        )

    async def stream(
        self,
        messages: List[CanonicalMessage],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        async for chunk in self._llm.stream(
            messages, system=system, tools=tools,
            max_tokens=max_tokens, temperature=temperature, **kwargs
        ):
            yield chunk

    async def health_check(self) -> bool:
        llm_ok = await self._llm.health_check()
        # Don't fail health check if Daytona is unavailable — it's lazy
        return llm_ok


@asynccontextmanager
async def daytona_session(
    api_key: Optional[str] = None,
    snapshot_id: Optional[str] = None,
    auto_destroy: bool = True,
) -> AsyncIterator[DaytonaSandbox]:
    """
    Async context manager for a one-off Daytona sandbox session.

    Usage::

        async with daytona_session() as sb:
            result = await sb.exec_python("print(2 + 2)")
            print(result["output"])  # "4"
    """
    try:
        from daytona import AsyncDaytona, DaytonaConfig
    except ImportError:
        raise ImportError("pip install daytona")

    config = DaytonaConfig(
        api_key=api_key or os.getenv("DAYTONA_API_KEY", ""),
        api_url=os.getenv("DAYTONA_API_URL", "https://app.daytona.io/api"),
        target=os.getenv("DAYTONA_TARGET", "us"),
    )
    client = AsyncDaytona(config)
    create_kwargs: Dict[str, Any] = {}
    if snapshot_id:
        create_kwargs["snapshot"] = snapshot_id

    try:
        raw_sb = await client.create(**create_kwargs)
        sb = DaytonaSandbox(sandbox=raw_sb, daytona_client=client)
        yield sb
    finally:
        if auto_destroy and sb:
            await sb.destroy()
        await client.close()
