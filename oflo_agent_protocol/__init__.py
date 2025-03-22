"""
Oflo AI Agent Protocol - A protocol for building business AI agents with MCP integration.
"""

from .agent import BaseAgent, Message, FunctionDefinition, AgentStatus
from .factory import AgentFactory
from .mcp import MCPServer, MCPClient

__version__ = "0.1.0"
__author__ = "Ankit Buti"
__email__ = "ankit@oflo.ai"

__all__ = [
    "BaseAgent",
    "Message",
    "FunctionDefinition",
    "AgentStatus",
    "AgentFactory",
    "MCPServer",
    "MCPClient",
] 