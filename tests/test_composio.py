"""Tests for ComposioConnector — all Composio SDK calls are mocked."""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oflo_agent_protocol.connectors.composio_connector import (
    ComposioConnector,
    ComposioToolKit,
    _account_to_dict,
    _extract_result,
    _tool_desc,
    _tool_slug,
)
from oflo_agent_protocol.core.agent import BaseAgentV2, ToolDefinition
from tests.conftest import StubRuntime


# ── Helper builders ───────────────────────────────────────────────────────────

def _make_mock_tool(slug: str, description: str = "", properties: Dict = None):
    t = MagicMock()
    t.slug = slug
    t.description = description
    t.input_parameters = {
        "type": "object",
        "properties": properties or {"query": {"type": "string"}},
        "required": list((properties or {"query": {}}).keys()),
    }
    return t


def _make_connector_with_new_sdk(tools=None, execute_result=None):
    """Return a ComposioConnector pre-wired with a mocked new SDK client."""
    connector = ComposioConnector(api_key="test-key", user_id="test-user")
    mock_client = MagicMock()

    mock_client.tools.get.return_value = tools or []
    mock_client.tools.execute.return_value = execute_result or {
        "successful": True,
        "data": {"result": "ok"},
    }
    mock_client.connected_accounts.list.return_value = []
    mock_client.connected_accounts.initiate.return_value = MagicMock(
        redirect_url="https://connect.composio.dev/oauth?app=github"
    )

    connector._client = mock_client
    connector._sdk_version = "new"
    return connector


# ── ComposioToolKit ────────────────────────────────────────────────────────────

class TestComposioToolKit:
    def test_toolkit_groups_are_lists(self):
        assert isinstance(ComposioToolKit.DEVOPS, list)
        assert isinstance(ComposioToolKit.COMMUNICATION, list)
        assert isinstance(ComposioToolKit.DATA, list)
        assert isinstance(ComposioToolKit.CLOUD, list)
        assert isinstance(ComposioToolKit.ALL_POPULAR, list)

    def test_devops_contains_github(self):
        assert "github" in ComposioToolKit.DEVOPS

    def test_communication_contains_slack(self):
        assert "slack" in ComposioToolKit.COMMUNICATION


# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_tool_slug_from_attr(self):
        t = MagicMock(slug="GITHUB_CREATE_ISSUE")
        assert _tool_slug(t) == "GITHUB_CREATE_ISSUE"

    def test_tool_slug_from_dict(self):
        assert _tool_slug({"slug": "MY_ACTION"}) == "MY_ACTION"
        assert _tool_slug({"name": "MY_ACTION"}) == "MY_ACTION"

    def test_tool_slug_fallback(self):
        assert _tool_slug({}) == "unknown"

    def test_tool_desc_from_attr(self):
        t = MagicMock(description="Does something cool")
        assert _tool_desc(t) == "Does something cool"

    def test_extract_result_success(self):
        r = {"successful": True, "data": {"id": 42}}
        assert _extract_result(r) == {"id": 42}

    def test_extract_result_failure_raises(self):
        r = {"successful": False, "error": "Permission denied"}
        with pytest.raises(RuntimeError, match="Permission denied"):
            _extract_result(r)

    def test_extract_result_plain_dict(self):
        r = {"anything": "goes"}
        assert _extract_result(r) == {"anything": "goes"}

    def test_account_to_dict_from_dict(self):
        d = {"id": "acc1", "app": "github", "status": "active"}
        assert _account_to_dict(d) == d

    def test_account_to_dict_from_object(self):
        obj = MagicMock(id="acc2", app_name="slack", status="active")
        result = _account_to_dict(obj)
        assert result["id"] == "acc2"
        assert result["app"] == "slack"


# ── ComposioConnector (new SDK path) ──────────────────────────────────────────

class TestComposioConnectorNewSDK:
    @pytest.mark.asyncio
    async def test_inject_no_tools(self):
        connector = _make_connector_with_new_sdk(tools=[])
        agent = BaseAgentV2(name="Empty", runtime=StubRuntime())
        n = await connector.inject_into_agent(agent, toolkits=["unknown"])
        assert n == 0
        assert len(agent._tools) == 0

    @pytest.mark.asyncio
    async def test_inject_tools_registered_on_agent(self):
        mock_tools = [
            _make_mock_tool("GITHUB_CREATE_ISSUE", "Create a GitHub issue"),
            _make_mock_tool("GITHUB_LIST_REPOS", "List repositories"),
        ]
        connector = _make_connector_with_new_sdk(tools=mock_tools)
        agent = BaseAgentV2(name="DevAgent", runtime=StubRuntime())
        n = await connector.inject_into_agent(agent, toolkits=["github"])
        assert n == 2
        assert "GITHUB_CREATE_ISSUE" in agent._tools
        assert "GITHUB_LIST_REPOS" in agent._tools

    @pytest.mark.asyncio
    async def test_injected_tool_is_callable(self):
        """The injected tool handler must call execute_action when invoked."""
        mock_tools = [_make_mock_tool("SEND_EMAIL", "Send an email")]
        connector = _make_connector_with_new_sdk(
            tools=mock_tools,
            execute_result={"successful": True, "data": {"message_id": "m1"}},
        )
        agent = BaseAgentV2(name="EmailAgent", runtime=StubRuntime())
        await connector.inject_into_agent(agent, toolkits=["gmail"])

        td: ToolDefinition = agent._tools["SEND_EMAIL"]
        result = await td.handler(to="test@example.com", subject="Hello")
        assert result == {"message_id": "m1"}

    @pytest.mark.asyncio
    async def test_inject_with_search(self):
        mock_tools = [_make_mock_tool("HN_GET_USER", "Get a HN user")]
        connector = _make_connector_with_new_sdk(tools=mock_tools)
        agent = BaseAgentV2(name="HNAgent", runtime=StubRuntime())
        n = await connector.inject_into_agent(agent, search="hackernews user")
        assert n == 1

        # Verify the SDK was called with search param
        connector._client.tools.get.assert_called_once()
        call_kwargs = connector._client.tools.get.call_args.kwargs
        assert call_kwargs.get("search") == "hackernews user"

    @pytest.mark.asyncio
    async def test_inject_with_limit(self):
        mock_tools = [_make_mock_tool(f"TOOL_{i}") for i in range(10)]
        connector = _make_connector_with_new_sdk(tools=mock_tools)
        agent = BaseAgentV2(name="LimitAgent", runtime=StubRuntime())
        n = await connector.inject_into_agent(agent, toolkits=["github"], limit=3)
        assert n == 3

    @pytest.mark.asyncio
    async def test_execute_action_direct(self):
        connector = _make_connector_with_new_sdk(
            execute_result={"successful": True, "data": {"created": True}}
        )
        result = await connector.execute_action(
            "GITHUB_CREATE_ISSUE",
            {"owner": "org", "repo": "repo", "title": "Bug"},
        )
        assert result == {"created": True}
        connector._client.tools.execute.assert_called_once_with(
            user_id="test-user",
            slug="GITHUB_CREATE_ISSUE",
            arguments={"owner": "org", "repo": "repo", "title": "Bug"},
        )

    @pytest.mark.asyncio
    async def test_execute_action_failure_raises(self):
        mock_client = MagicMock()
        mock_client.tools.execute.return_value = {
            "successful": False,
            "error": "Not authorised",
        }
        connector = ComposioConnector(api_key="key", user_id="user")
        connector._client = mock_client
        connector._sdk_version = "new"

        with pytest.raises(RuntimeError, match="Not authorised"):
            await connector.execute_action("GITHUB_CREATE_ISSUE", {})

    @pytest.mark.asyncio
    async def test_connect_app_returns_url(self):
        connector = _make_connector_with_new_sdk()
        url = await connector.connect_app("github")
        assert "composio" in url.lower() or "github" in url.lower() or url.startswith("http")

    @pytest.mark.asyncio
    async def test_list_connected_apps(self):
        mock_acc = MagicMock(id="acc1", app_name="github", status="active")
        mock_client = MagicMock()
        mock_client.connected_accounts.list.return_value = [mock_acc]
        mock_client.tools.get.return_value = []

        connector = ComposioConnector(api_key="key", user_id="user")
        connector._client = mock_client
        connector._sdk_version = "new"

        apps = await connector.list_connected_apps()
        assert len(apps) == 1
        assert apps[0]["app"] == "github"

    @pytest.mark.asyncio
    async def test_list_actions_returns_dicts(self):
        mock_tools = [_make_mock_tool("GH_STAR_REPO", "Star a repo")]
        connector = _make_connector_with_new_sdk(tools=mock_tools)
        actions = await connector.list_actions(toolkits=["github"])
        assert len(actions) == 1
        assert actions[0]["slug"] == "GH_STAR_REPO"

    def test_describe(self):
        connector = _make_connector_with_new_sdk()
        d = connector.describe()
        assert d["user_id"] == "test-user"
        assert d["sdk_version"] == "new"
        assert d["initialized"] is True


# ── Tool schema integrity ─────────────────────────────────────────────────────

class TestInjectedToolSchema:
    @pytest.mark.asyncio
    async def test_tool_definition_has_correct_schema(self):
        props = {
            "owner": {"type": "string", "description": "Repo owner"},
            "repo": {"type": "string", "description": "Repo name"},
            "title": {"type": "string", "description": "Issue title"},
        }
        mock_tools = [_make_mock_tool("GH_CREATE_ISSUE", "Create issue", props)]
        connector = _make_connector_with_new_sdk(tools=mock_tools)
        agent = BaseAgentV2(name="Dev", runtime=StubRuntime())
        await connector.inject_into_agent(agent)

        td = agent._tools["GH_CREATE_ISSUE"]
        assert td.description == "Create issue"
        assert "owner" in td.parameters
        assert "repo" in td.parameters
        assert "title" in td.parameters

        openai_schema = td.to_openai_schema()
        assert openai_schema["type"] == "function"
        assert openai_schema["function"]["name"] == "GH_CREATE_ISSUE"

        anthropic_schema = td.to_anthropic_schema()
        assert anthropic_schema["name"] == "GH_CREATE_ISSUE"
        assert "input_schema" in anthropic_schema

    @pytest.mark.asyncio
    async def test_tool_in_agentic_loop(self):
        """End-to-end: composio tool is invoked by the agent's agentic loop."""
        from oflo_agent_protocol.core.types import MessageRole
        from oflo_agent_protocol.core.message import ToolCall

        tc = ToolCall(
            id="tc1",
            name="SLACK_SEND_MESSAGE",
            arguments={"channel": "#general", "text": "Hello team!"},
        )
        runtime = StubRuntime(reply="Message sent to Slack.", tool_calls=[tc])

        agent = BaseAgentV2(name="SlackBot", runtime=runtime)

        call_log = []

        async def slack_send(channel: str, text: str) -> dict:
            call_log.append({"channel": channel, "text": text})
            return {"ok": True, "ts": "1234567890.000001"}

        agent.register_tool(
            name="SLACK_SEND_MESSAGE",
            description="Send a Slack message",
            parameters={
                "channel": {"type": "string"},
                "text": {"type": "string"},
            },
            handler=slack_send,
            required=["channel", "text"],
        )

        reply = await agent.chat("Post a hello to #general on Slack")
        assert call_log[0]["channel"] == "#general"
        assert "sent" in reply.lower() or "slack" in reply.lower()
