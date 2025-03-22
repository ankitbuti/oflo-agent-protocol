import logging
import asyncio
import json
from oflo_agent_interface import MCPServer, MCPClient, MasterAgent

print("---\n\n--------oflo ai kickoff")

# Example usage
async def main():
    # Start an MCP server
    server = MCPServer(port=8080)
    await server.start()
    
    # Create a master agent
    master = MasterAgent()
    await master.initialize({"provider": "anthropic", "model_name": "claude-3-sonnet-20240229"})
    await server.register_agent(master)
    
    # Connect as a client
    async with MCPClient("http://localhost:8080") as client:
        # List all agents
        agents = await client.list_agents()
        print("Available agents:", json.dumps(agents, indent=2))
        
        # Send a chat message to the master agent
        response = await client.chat_completion(
            master.id,
            [{"role": "user", "content": "List all available agents"}]
        )
        print("Master agent response:", json.dumps(response, indent=2))
    
    # Keep the server running
    print("MCP Server is running. Press Ctrl+C to stop.")
    try:
        # Wait forever
        await asyncio.Future()
    except asyncio.CancelledError:
        # Clean up on cancellation
        for agent_id in list(server.agents.keys()):
            await server.agents[agent_id].terminate()
            await server.unregister_agent(agent_id)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())