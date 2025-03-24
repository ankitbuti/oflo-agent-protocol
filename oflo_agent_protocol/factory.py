from typing import Type, List, Dict, Any, Union
from agent import BaseAgent, AgentStatus, Message

from services.weaviate import WeaviateService

weaviate_service = WeaviateService(
    url=os.getenv('WEAVIATE_URL'),
    api_key=os.getenv('WEAVIATE_API_KEY')
)

class OfloAgentFactory:
    """Factory for creating agents based on a specified type."""
    
    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> Type[BaseAgent]:
        """Create an agent of the specified type."""
        if agent_type == "example_agent":
            return ExampleAgent(**kwargs)
        elif agent_type == "another_agent":
            return AnotherAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

class ExampleAgent(BaseAgent):
    """An example implementation of a BaseOfloAgent."""
    
    def __init__(self, id: str, name: str, capabilities: List[str]):
        self._id = id
        self._name = name
        self._capabilities = capabilities
        self._status = AgentStatus.INACTIVE

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    def capabilities(self) -> List[str]:
        return self._capabilities

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        self._status = AgentStatus.ACTIVE
        return True

    async def process_message(self, message: Union[str, Dict, Message]) -> Message:
        # Process the incoming message and return a response
        return Message(role="assistant", content="Processed message")

    async def call_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        # Call a specific function with the provided parameters
        return {"result": "Function called"}

class AnotherAgent(BaseAgent):
    """Another example implementation of a BaseOfloAgent."""
    
    def __init__(self, id: str, name: str, capabilities: List[str]):
        self._id = id
        self._name = name
        self._capabilities = capabilities
        self._status = AgentStatus.INACTIVE

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> AgentStatus:
        return self._status

    @property
    def capabilities(self) -> List[str]:
        return self._capabilities

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        self._status = AgentStatus.ACTIVE
        return True

    async def process_message(self, message: Union[str, Dict, Message]) -> Message:
        # Process the incoming message and return a response
        return Message(role="assistant", content="Another agent processed message")

    async def call_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        # Call a specific function with the provided parameters
        return {"result": "Another function called"}
