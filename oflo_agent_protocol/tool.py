"""Tool implementation."""
from typing import Optional, Callable, Dict, Any
from mcp.tool import Tool as MCPTool

class Tool(MCPTool):
    """Wrapper around MCP Tool with additional functionality."""
    
    def __init__(self, name: str, handler: Callable, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(name, handler, description)
        self.metadata = metadata or {} 