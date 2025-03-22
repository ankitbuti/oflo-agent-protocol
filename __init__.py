"""
Oflo AI Agent Protocol
~~~~~~~~~~~~~~~~~~

A protocol for building AI agents with MCP integration.
"""

from .agent import Agent
from .resource import Resource
from .tool import Tool
from .version import __version__

__all__ = ["Agent", "Resource", "Tool", "__version__"] 

"""Version information."""
__version__ = "0.1.0"