from mcp import Server, Resource, Tool
from slack_sdk import WebClient
import asyncio

# Initialize MCP Server
class PingPongServer(Server):
    def __init__(self):
        super().__init__("ping-pong-server")
        self.slack_client = WebClient(token="YOUR_SLACK_TOKEN")
        
    async def setup(self):
        # Register ping-pong resource
        await self.register_resource(
            Resource(
                "ping-pong",
                description="A simple ping-pong game resource"
            )
        )
        
        # Register tools
        await self.register_tool(
            Tool(
                "play_ping_pong",
                self.play_ping_pong,
                description="Play a round of ping pong"
            )
        )
        
    async def play_ping_pong(self, message: str):
        response = "pong" if message.lower() == "ping" else "ping"
        
        # Send notification to Slack
        try:
            self.slack_client.chat_postMessage(
                channel="#ping-pong",
                text=f"Received: {message}\nResponded: {response}"
            )
        except Exception as e:
            print(f"Error sending to Slack: {e}")
            
        return response

async def main():
    # Create and start the server
    server = PingPongServer()
    await server.start()
    
    try:
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await server.stop()

if __name__ == "__main__":
    asyncio.run(main())