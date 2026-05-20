"""Oflo Agent Protocol — external app connectors.

Currently bundled:
  - Composio  (300+ app integrations: GitHub, Gmail, Slack, Notion, Jira, …)

Usage::

    from oflo_agent_protocol.connectors import ComposioConnector

    connector = ComposioConnector(api_key="...", user_id="alice")
    await connector.inject_into_agent(agent, toolkits=["github", "gmail"])
"""
from oflo_agent_protocol.connectors.composio_connector import (
    ComposioConnector,
    ComposioToolKit,
)

__all__ = ["ComposioConnector", "ComposioToolKit"]
