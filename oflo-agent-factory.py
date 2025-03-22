from typing import Type

class OfloAgentFactory:
    """Factory for creating agents based on a specified type."""
    
    @staticmethod
    def create_agent(agent_type: str, **kwargs) -> Type[BaseOfloAgent]:
        """Create an agent of the specified type."""
        if agent_type == "example_agent":
            return ExampleAgent(**kwargs)
        elif agent_type == "another_agent":
            return AnotherAgent(**kwargs)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

class ExampleAgent(BaseOfloAgent):
    """An example implementation of a BaseOfloAgent."""
    
    def __init__(self, id: str, name: str, capabilities: List[str]):
        self._id = id
        self._name = name
        self._capabilities = capabilities
        self._status = OfloAgentStatus.INACTIVE

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> OfloAgentStatus:
        return self._status

    @property
    def capabilities(self) -> List[str]:
        return self._capabilities

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        self._status = OfloAgentStatus.ACTIVE
        return True

    async def process_message(self, message: Union[str, Dict, OfloMessage]) -> OfloMessage:
        # Process the incoming message and return a response
        return OfloMessage(role="assistant", content="Processed message")

    async def call_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        # Call a specific function with the provided parameters
        return {"result": "Function called"}

class AnotherAgent(BaseOfloAgent):
    """Another example implementation of a BaseOfloAgent."""
    
    def __init__(self, id: str, name: str, capabilities: List[str]):
        self._id = id
        self._name = name
        self._capabilities = capabilities
        self._status = OfloAgentStatus.INACTIVE

    @property
    def id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> OfloAgentStatus:
        return self._status

    @property
    def capabilities(self) -> List[str]:
        return self._capabilities

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        self._status = OfloAgentStatus.ACTIVE
        return True

    async def process_message(self, message: Union[str, Dict, OfloMessage]) -> OfloMessage:
        # Process the incoming message and return a response
        return OfloMessage(role="assistant", content="Another agent processed message")

    async def call_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        # Call a specific function with the provided parameters
        return {"result": "Another function called"}
