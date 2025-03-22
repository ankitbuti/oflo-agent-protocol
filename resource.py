"""Resource implementation."""
from typing import Optional, Dict, Any
from mcp.resource import Resource as MCPResource

class Resource(MCPResource):
    """Wrapper around MCP Resource with additional functionality."""
    
    def __init__(self, name: str, description: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(name, description)
        self.metadata = metadata or {} 