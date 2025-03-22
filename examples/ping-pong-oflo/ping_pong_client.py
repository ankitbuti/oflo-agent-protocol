from mcp.client import MCPClient  # Updated import
import asyncio

async def test_ping_pong():
    client = MCPClient()  # Using MCPClient instead of Client
    await client.connect("ping-pong-server")
    
    # Use the play_ping_pong tool
    result = await client.invoke_tool("play_ping_pong", "ping")
    print(f"Result: {result}")  # Should print "pong"
    
    result = await client.invoke_tool("play_ping_pong", "pong")
    print(f"Result: {result}")  # Should print "ping"

if __name__ == "__main__":
    asyncio.run(test_ping_pong())