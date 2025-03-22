from typing import Dict, Any, List
import asyncio
import logging
from oflo_agent_protocol import BaseAgent, Message, FunctionDefinition

class PingPongAgent(BaseAgent):
    """A simple ping-pong agent demonstrating the Oflo Agent Protocol."""
    
    def __init__(self):
        super().__init__(
            name="PingPongAgent",
            purpose="A simple agent that plays ping pong by responding to 'ping' with 'pong' and vice versa"
        )
        self.logger = logging.getLogger(__name__)
        
    @property
    def available_functions(self) -> List[FunctionDefinition]:
        """Define the functions this agent can use."""
        return [
            FunctionDefinition(
                name="play_ping_pong",
                description="Play a round of ping pong",
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The ping or pong message"
                        }
                    },
                    "required": ["message"]
                }
            )
        ]
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the agent."""
        try:
            # Register the play_ping_pong function as a tool
            self.register_function_as_tool(self.available_functions[0])
            self._status = "active"
            self.logger.info("PingPongAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize PingPongAgent: {e}")
            return False
    
    async def process_message(self, message: str) -> Message:
        """Process incoming messages and respond appropriately."""
        try:
            # Convert string message to Message object if needed
            if isinstance(message, str):
                message = Message(role="user", content=message)
            
            # Check if the message is ping or pong
            content = message.content.lower().strip()
            if content in ["ping", "pong"]:
                response = await self.call_function(
                    "play_ping_pong",
                    {"message": content}
                )
                return Message(role="assistant", content=response)
            else:
                return Message(
                    role="assistant",
                    content="Please send 'ping' or 'pong' to play!"
                )
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return Message(
                role="assistant",
                content="Sorry, I encountered an error processing your message."
            )
    
    async def play_ping_pong(self, parameters: Dict[str, Any]) -> str:
        """Handle the ping-pong game logic."""
        message = parameters.get("message", "").lower().strip()
        response = "pong" if message == "ping" else "ping"
        self.logger.info(f"Received: {message}, Responding: {response}")
        return response

# Export the agent class for use in the worker
__all__ = ['PingPongAgent'] 