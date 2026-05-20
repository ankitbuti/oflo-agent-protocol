"""Composio connector — brings 300+ app integrations into every oflo agent.

Composio (https://composio.dev) provides ready-made tools for GitHub, Gmail,
Slack, Notion, Jira, Salesforce, Linear, and hundreds more. This module
converts those tools into native oflo ToolDefinition objects so they slot
seamlessly into the agentic loop of any BaseAgentV2 agent.

Architecture
────────────
• ComposioConnector   — user-facing class; manages sessions, injects tools.
• OFloComposioProvider— custom NonAgenticProvider that maps Composio tools
                        to ToolDefinition objects (new SDK path).
• ComposioToolKit      — convenience grouping of related apps (e.g. "devops").

Supported Composio SDK versions
────────────────────────────────
• New SDK  (composio>=0.7) — `Composio().tools.get(user_id, toolkits=[...])`
• Legacy   (composio-core) — `ComposioToolSet().get_tools(apps=[...])`

Both are detected at runtime; new SDK is preferred.

Quick start
───────────
    from oflo_agent_protocol.connectors import ComposioConnector

    connector = ComposioConnector(api_key=os.getenv("COMPOSIO_API_KEY"))

    # Inject GitHub + Gmail tools into an existing agent
    n = await connector.inject_into_agent(agent, toolkits=["github", "gmail"])
    print(f"Injected {n} Composio tools")

    # Start an OAuth flow for an app
    url = await connector.connect_app("github")
    print(f"Authorize here: {url}")

    # Execute a Composio action directly (no LLM needed)
    result = await connector.execute_action(
        "GITHUB_CREATE_ISSUE",
        {"owner": "my-org", "repo": "my-repo", "title": "Bug report"},
    )
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from oflo_agent_protocol.core.agent import BaseAgentV2
from oflo_agent_protocol.core.agent import ToolDefinition

logger = logging.getLogger(__name__)


# ── Pre-built toolkit groupings ────────────────────────────────────────────────

class ComposioToolKit:
    """Named collections of Composio app slugs for common use-cases."""

    DEVOPS = ["github", "gitlab", "jira", "linear", "notion"]
    COMMUNICATION = ["gmail", "slack", "discord", "telegram", "outlook"]
    PRODUCTIVITY = ["notion", "airtable", "asana", "trello", "monday"]
    DATA = ["googlesheets", "airtable", "hubspot", "salesforce", "snowflake"]
    CLOUD = ["aws", "gcp", "azure", "vercel", "netlify"]
    ECOMMERCE = ["shopify", "stripe", "woocommerce", "paypal"]
    ALL_POPULAR = [
        "github", "gmail", "slack", "notion", "jira",
        "linear", "airtable", "googlesheets", "hubspot",
    ]


# ── Custom Composio provider (new SDK) ─────────────────────────────────────────

def _make_oflo_provider_class() -> Any:
    """
    Lazily build OFloComposioProvider — only callable if composio ≥ 0.7
    with NonAgenticProvider support.
    """
    try:
        from composio.core.provider import NonAgenticProvider  # type: ignore
        from composio.types import Tool as ComposioTool  # type: ignore
    except ImportError:
        return None

    class OFloComposioProvider(NonAgenticProvider, name="oflo-agent-protocol"):  # type: ignore
        """
        Translates Composio tools into ToolDefinition objects for BaseAgentV2.

        This is used with:
            composio = Composio(provider=OFloComposioProvider())
            session  = composio.create(user_id="...")
            tool_defs = session.tools(toolkits=["github"])
        """

        def wrap_tool(self, tool: ComposioTool) -> ToolDefinition:  # type: ignore
            slug = _tool_slug(tool)
            desc = _tool_desc(tool)
            schema = _tool_schema(tool)
            properties: Dict[str, Any] = schema.get("properties", {})
            required: List[str] = schema.get("required", [])

            # Placeholder handler — real execution happens via execute_tool
            async def _placeholder(**kwargs: Any) -> Any:
                raise NotImplementedError(
                    f"Use ComposioConnector.execute_action('{slug}', ...) "
                    "for direct calls, or inject into an agent for LLM-driven execution."
                )

            return ToolDefinition(
                name=slug,
                description=desc,
                parameters=properties,
                handler=_placeholder,
                required=required,
            )

        def wrap_tools(
            self, tools: Sequence[ComposioTool]  # type: ignore
        ) -> List[ToolDefinition]:
            return [self.wrap_tool(t) for t in tools]

    return OFloComposioProvider


# ── ComposioConnector ──────────────────────────────────────────────────────────

class ComposioConnector:
    """
    Bridges Composio's 300+ app connectors into oflo-agent-protocol agents.

    Every tool injected via :meth:`inject_into_agent` becomes a first-class
    native tool: it appears in the agent's tool schema, participates in the
    agentic loop, and is tracked in the audit log.

    Parameters
    ──────────
    api_key     Composio API key (defaults to ``COMPOSIO_API_KEY`` env var).
    user_id     Identifies the end-user for Composio sessions / entity management.
                Defaults to ``"default"``.

    Example
    ───────
    ::

        connector = ComposioConnector()          # key from env
        await connector.inject_into_agent(
            agent,
            toolkits=["github", "slack"],
        )
        # The agent can now create GitHub issues and send Slack messages.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_id: str = "default",
    ) -> None:
        self._api_key = api_key or os.getenv("COMPOSIO_API_KEY", "")
        self._user_id = user_id
        self._client: Optional[Any] = None        # new SDK: Composio instance
        self._toolset: Optional[Any] = None       # legacy: ComposioToolSet
        self._sdk_version: str = "unknown"
        self._logger = logging.getLogger("oflo.composio")

    # ------------------------------------------------------------------
    # SDK detection & client initialisation
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Detect installed Composio SDK version and initialise the client."""
        if self._client or self._toolset:
            return

        # ── Try new SDK first (composio >= 0.7) ──────────────────────
        try:
            from composio import Composio  # type: ignore
            kwargs: Dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = Composio(**kwargs)
            self._sdk_version = "new"
            self._logger.debug("Composio new-SDK client initialised")
            return
        except (ImportError, Exception) as exc:
            self._logger.debug("New Composio SDK unavailable: %s", exc)

        # ── Fall back to legacy ComposioToolSet ───────────────────────
        try:
            from composio_openai import ComposioToolSet  # type: ignore
            self._toolset = ComposioToolSet(
                api_key=self._api_key or None,
                entity_id=self._user_id,
            )
            self._sdk_version = "legacy-openai"
            self._logger.debug("Composio legacy (composio_openai) client initialised")
            return
        except ImportError:
            pass

        try:
            from composio_langchain import ComposioToolSet  # type: ignore
            self._toolset = ComposioToolSet(
                api_key=self._api_key or None,
                entity_id=self._user_id,
            )
            self._sdk_version = "legacy-langchain"
            self._logger.debug("Composio legacy (composio_langchain) client initialised")
            return
        except ImportError:
            pass

        raise ImportError(
            "No Composio SDK found. Install with:\n"
            "  pip install composio          # new SDK (recommended)\n"
            "  pip install composio-openai   # legacy SDK\n"
            "  pip install composio-langchain # legacy SDK (LangChain flavour)"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def inject_into_agent(
        self,
        agent: BaseAgentV2,
        toolkits: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> int:
        """
        Fetch Composio tools and register them directly on *agent*.

        Parameters
        ──────────
        agent       Target BaseAgentV2 (or subclass) to inject tools into.
        toolkits    App slugs to include, e.g. ``["github", "slack"]``.
        actions     Specific action slugs, e.g. ``["GITHUB_CREATE_ISSUE"]``.
        search      Keyword search across Composio's action catalogue.
        limit       Cap the number of tools injected (default: all matched).
        tags        Filter by Composio action tags (e.g. ``["important"]``).

        Returns the number of tools injected.
        """
        tool_defs = await self._fetch_tool_definitions(
            toolkits=toolkits,
            actions=actions,
            search=search,
            limit=limit,
            tags=tags,
        )
        for td in tool_defs:
            agent._tools[td.name] = td
            self._logger.debug("Injected Composio tool '%s' into agent '%s'", td.name, agent.name)

        self._logger.info(
            "Injected %d Composio tool(s) into agent '%s'", len(tool_defs), agent.name
        )
        return len(tool_defs)

    async def connect_app(
        self,
        app_name: str,
        callback_url: Optional[str] = None,
        auth_config_id: Optional[str] = None,
    ) -> str:
        """
        Initiate an OAuth / API-key connection for *app_name*.

        Returns a redirect URL (for OAuth apps) or a status string.
        """
        loop = asyncio.get_event_loop()
        self._init_client()

        try:
            if self._client and self._sdk_version == "new":
                # New SDK: connected_accounts.initiate(...)
                kwargs: Dict[str, Any] = {
                    "user_id": self._user_id,
                    "app": app_name.upper(),
                }
                if auth_config_id:
                    kwargs["auth_config_id"] = auth_config_id
                if callback_url:
                    kwargs["callback_url"] = callback_url

                req = await loop.run_in_executor(
                    None,
                    lambda: self._client.connected_accounts.initiate(**kwargs),
                )
                return getattr(req, "redirect_url", str(req))

            elif self._toolset:
                # Legacy SDK: entity.initiate_connection(...)
                entity = await loop.run_in_executor(
                    None,
                    lambda: self._toolset.client.get_entity(self._user_id),
                )
                req = await loop.run_in_executor(
                    None,
                    lambda: entity.initiate_connection(app_name=app_name.upper()),
                )
                return getattr(req, "redirectUrl", getattr(req, "redirect_url", str(req)))

        except Exception as exc:
            self._logger.error("connect_app(%s) failed: %s", app_name, exc)
            raise RuntimeError(f"Failed to connect '{app_name}': {exc}") from exc

        return f"connected:{app_name}"

    async def list_connected_apps(self) -> List[Dict[str, Any]]:
        """Return all connected apps / accounts for this user."""
        loop = asyncio.get_event_loop()
        self._init_client()

        try:
            if self._client and self._sdk_version == "new":
                accounts = await loop.run_in_executor(
                    None,
                    lambda: self._client.connected_accounts.list(user_id=self._user_id),
                )
                return [_account_to_dict(a) for a in (accounts or [])]

            elif self._toolset:
                entity = await loop.run_in_executor(
                    None,
                    lambda: self._toolset.client.get_entity(self._user_id),
                )
                connections = await loop.run_in_executor(
                    None,
                    lambda: entity.get_connections(),
                )
                return [_account_to_dict(c) for c in (connections or [])]

        except Exception as exc:
            self._logger.warning("list_connected_apps failed: %s", exc)

        return []

    async def execute_action(
        self,
        action_slug: str,
        params: Dict[str, Any],
    ) -> Any:
        """
        Execute a Composio action directly (no LLM involved).

        Parameters
        ──────────
        action_slug     e.g. ``"GITHUB_CREATE_ISSUE"``
        params          Action input parameters dict.

        Returns the action result data.
        """
        loop = asyncio.get_event_loop()
        self._init_client()

        try:
            if self._client and self._sdk_version == "new":
                result = await loop.run_in_executor(
                    None,
                    lambda: self._client.tools.execute(
                        user_id=self._user_id,
                        slug=action_slug,
                        arguments=params,
                    ),
                )
                return _extract_result(result)

            elif self._toolset:
                result = await loop.run_in_executor(
                    None,
                    lambda: self._toolset.execute_action(
                        action=action_slug,
                        params=params,
                        entity_id=self._user_id,
                    ),
                )
                return _extract_result(result)

        except Exception as exc:
            self._logger.error("execute_action(%s) failed: %s", action_slug, exc)
            raise RuntimeError(
                f"Composio action '{action_slug}' failed: {exc}"
            ) from exc

    async def list_actions(
        self,
        toolkits: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Return available Composio actions as dicts (for discovery)."""
        loop = asyncio.get_event_loop()
        self._init_client()

        try:
            if self._client and self._sdk_version == "new":
                kwargs: Dict[str, Any] = {"user_id": self._user_id}
                if toolkits:
                    kwargs["toolkits"] = toolkits
                if search:
                    kwargs["search"] = search
                raw = await loop.run_in_executor(
                    None,
                    lambda: self._client.tools.get(**kwargs),
                )
                tools = raw or []
                return [
                    {
                        "slug": _tool_slug(t),
                        "description": _tool_desc(t),
                        "toolkit": _tool_toolkit(t),
                    }
                    for t in tools[:limit]
                ]

            elif self._toolset:
                kwargs2: Dict[str, Any] = {}
                if toolkits:
                    kwargs2["apps"] = [t.upper() for t in toolkits]
                raw2 = await loop.run_in_executor(
                    None,
                    lambda: self._toolset.get_tools(**kwargs2),
                )
                return [
                    {
                        "slug": _openai_tool_name(t),
                        "description": _openai_tool_desc(t),
                        "toolkit": "unknown",
                    }
                    for t in (raw2 or [])[:limit]
                ]

        except Exception as exc:
            self._logger.warning("list_actions failed: %s", exc)

        return []

    def describe(self) -> Dict[str, Any]:
        return {
            "user_id": self._user_id,
            "sdk_version": self._sdk_version,
            "initialized": bool(self._client or self._toolset),
        }

    # ------------------------------------------------------------------
    # Internal — tool fetching and conversion
    # ------------------------------------------------------------------

    async def _fetch_tool_definitions(
        self,
        toolkits: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ToolDefinition]:
        """Fetch tools from Composio and convert to ToolDefinition objects."""
        loop = asyncio.get_event_loop()
        self._init_client()

        # ── New SDK ───────────────────────────────────────────────────
        if self._client and self._sdk_version == "new":
            return await self._fetch_new_sdk(
                loop, toolkits=toolkits, actions=actions,
                search=search, limit=limit, tags=tags,
            )

        # ── Legacy SDK ────────────────────────────────────────────────
        if self._toolset:
            return await self._fetch_legacy_sdk(
                loop, toolkits=toolkits, actions=actions,
                search=search, limit=limit, tags=tags,
            )

        return []

    async def _fetch_new_sdk(
        self,
        loop: asyncio.AbstractEventLoop,
        toolkits: Optional[List[str]],
        actions: Optional[List[str]],
        search: Optional[str],
        limit: Optional[int],
        tags: Optional[List[str]],
    ) -> List[ToolDefinition]:
        kwargs: Dict[str, Any] = {"user_id": self._user_id}
        if toolkits:
            kwargs["toolkits"] = [t.upper() for t in toolkits]
        if actions:
            kwargs["actions"] = actions
        if search:
            kwargs["search"] = search
        if tags:
            kwargs["tags"] = tags

        raw = await loop.run_in_executor(
            None,
            lambda: self._client.tools.get(**kwargs),
        )
        raw = raw or []
        if limit:
            raw = raw[:limit]

        return [self._new_sdk_tool_to_def(t) for t in raw]

    def _new_sdk_tool_to_def(self, tool: Any) -> ToolDefinition:
        slug = _tool_slug(tool)
        desc = _tool_desc(tool)
        schema = _tool_schema(tool)
        properties: Dict[str, Any] = schema.get("properties", {})
        required: List[str] = schema.get("required", [])

        connector = self  # capture for closure

        async def handler(**kwargs: Any) -> Any:
            return await connector.execute_action(slug, kwargs)

        return ToolDefinition(
            name=slug,
            description=desc or slug,
            parameters=properties,
            handler=handler,
            required=required,
        )

    async def _fetch_legacy_sdk(
        self,
        loop: asyncio.AbstractEventLoop,
        toolkits: Optional[List[str]],
        actions: Optional[List[str]],
        search: Optional[str],
        limit: Optional[int],
        tags: Optional[List[str]],
    ) -> List[ToolDefinition]:
        """Convert OpenAI-format tools from the legacy SDK to ToolDefinition."""
        kwargs: Dict[str, Any] = {}

        if toolkits:
            # Legacy SDK uses App enum or string names
            try:
                from composio import App  # type: ignore
                kwargs["apps"] = [App(t.upper()) for t in toolkits]
            except Exception:
                kwargs["apps"] = [t.upper() for t in toolkits]

        if actions:
            try:
                from composio import Action  # type: ignore
                kwargs["actions"] = [Action(a) for a in actions]
            except Exception:
                kwargs["actions"] = actions

        if tags:
            kwargs["tags"] = tags

        raw: List[Any] = await loop.run_in_executor(
            None,
            lambda: self._toolset.get_tools(**kwargs),
        )
        raw = raw or []
        if limit:
            raw = raw[:limit]

        return [self._legacy_tool_to_def(t) for t in raw]

    def _legacy_tool_to_def(self, openai_tool: Any) -> ToolDefinition:
        """Convert an OpenAI-format function tool dict to ToolDefinition."""
        # OpenAI tool format: {"type": "function", "function": {...}}
        fn = openai_tool if isinstance(openai_tool, dict) else {}
        if "function" in fn:
            fn = fn["function"]

        name: str = fn.get("name", "unknown")
        desc: str = fn.get("description", "")
        params_schema: Dict[str, Any] = fn.get("parameters", {})
        properties: Dict[str, Any] = params_schema.get("properties", {})
        required: List[str] = params_schema.get("required", [])

        connector = self

        async def handler(**kwargs: Any) -> Any:
            return await connector.execute_action(name, kwargs)

        return ToolDefinition(
            name=name,
            description=desc or name,
            parameters=properties,
            handler=handler,
            required=required,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tool_slug(tool: Any) -> str:
    for attr in ("slug", "name", "action_name"):
        v = getattr(tool, attr, None)
        if v:
            return str(v)
    if isinstance(tool, dict):
        return tool.get("slug") or tool.get("name") or "unknown"
    return "unknown"


def _tool_desc(tool: Any) -> str:
    for attr in ("description", "display_name", "title"):
        v = getattr(tool, attr, None)
        if v:
            return str(v)
    if isinstance(tool, dict):
        return tool.get("description") or tool.get("display_name") or ""
    return ""


def _tool_toolkit(tool: Any) -> str:
    for attr in ("toolkit", "app", "app_name"):
        v = getattr(tool, attr, None)
        if v:
            return str(v)
    return "unknown"


def _tool_schema(tool: Any) -> Dict[str, Any]:
    for attr in ("input_parameters", "parameters", "input_schema", "schema"):
        v = getattr(tool, attr, None)
        if isinstance(v, dict):
            return v
    return {}


def _openai_tool_name(t: Any) -> str:
    if isinstance(t, dict):
        return t.get("function", {}).get("name") or t.get("name") or "unknown"
    return getattr(t, "name", "unknown")


def _openai_tool_desc(t: Any) -> str:
    if isinstance(t, dict):
        return t.get("function", {}).get("description") or t.get("description") or ""
    return getattr(t, "description", "")


def _account_to_dict(account: Any) -> Dict[str, Any]:
    if isinstance(account, dict):
        return account
    return {
        "id": getattr(account, "id", None),
        "app": getattr(account, "app_name", getattr(account, "app", None)),
        "status": getattr(account, "status", None),
    }


def _extract_result(result: Any) -> Any:
    """Normalise execution results — different SDK versions return different shapes."""
    if isinstance(result, dict):
        if not result.get("successful", True):
            raise RuntimeError(
                f"Composio action failed: {result.get('error', 'unknown error')}"
            )
        return result.get("data", result)
    # Some SDK versions wrap results in objects
    if hasattr(result, "data"):
        return result.data
    if hasattr(result, "response_data"):
        return result.response_data
    return result
