"""Base Agent implementation."""
from typing import Optional, Dict, Any
from mcp.server import MCPServer
from .resource import Resource
from .tool import Tool

class Agent(MCPServer):
    """Base class for all Oflo agents."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self._tools: Dict[str, Tool] = {}
        self._resources: Dict[str, Resource] = {}

    async def register_tool(self, tool: Tool) -> None:
        """Register a new tool with the agent."""
        self._tools[tool.name] = tool
        await super().register_tool(tool)

    async def register_resource(self, resource: Resource) -> None:
        """Register a new resource with the agent."""
        self._resources[resource.name] = resource
        await super().register_resource(resource)

    def run(self) -> None:
        """Run the agent."""
        import asyncio
        asyncio.run(self._run())

    async def _run(self) -> None:
        """Internal run method."""
        await self.setup()
        await self.start()
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            await self.stop()

    async def setup(self) -> None:
        """Override this method to setup your agent."""
        pass 