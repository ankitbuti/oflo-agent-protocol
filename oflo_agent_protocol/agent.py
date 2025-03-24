from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Callable, Awaitable, Tuple
import json
import uuid
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
import aiohttp
from aiohttp import web
from services.weaviate import WeaviateService
from aimon import Detect


class AgentStatus(Enum):
    """Enumeration of possible agent statuses."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    INACTIVE = "inactive"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class FunctionDefinition:
    """Represents a function that an agent can call."""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str] = None


@dataclass
class AgentMemory:
    """Memory structure for agents."""
    short_term: Dict[str, Any] = None
    long_term: Dict[str, Any] = None
    buffer_size: int = 10

    def __post_init__(self):
        self.short_term = self.short_term or {}
        self.long_term = self.long_term or {}

    def store(self, key: str, value: Any, permanent: bool = False) -> None:
        """Store data in memory."""
        self.short_term[key] = value
        if permanent:
            self.long_term[key] = value
        
        # Trim short-term memory if it exceeds buffer size
        if len(self.short_term) > self.buffer_size:
            oldest_key = next(iter(self.short_term))
            del self.short_term[oldest_key]

    def retrieve(self, key: str, from_long_term: bool = False) -> Any:
        """Retrieve data from memory."""
        if from_long_term:
            return self.long_term.get(key)
        return self.short_term.get(key) or self.long_term.get(key)

    def clear_short_term(self) -> None:
        """Clear short-term memory."""
        self.short_term = {}


class Message:
    """Represents a message in the conversation."""
    def __init__(self, 
                 role: str, 
                 content: str, 
                 function_call: Optional[Dict] = None, 
                 tool_calls: Optional[List[Dict]] = None,
                 tool_results: Optional[List[Dict]] = None):
        self.role = role
        self.content = content
        self.function_call = function_call
        self.tool_calls = tool_calls or []
        self.tool_results = tool_results or []
        self.timestamp = datetime.now().isoformat()
        self.message_id = str(uuid.uuid4())

    def to_dict(self) -> Dict:
        """Convert message to dictionary format."""
        result = {
            "id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
        if self.function_call:
            result["function_call"] = self.function_call
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_results:
            result["tool_results"] = self.tool_results
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create a Message object from a dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            function_call=data.get("function_call"),
            tool_calls=data.get("tool_calls"),
            tool_results=data.get("tool_results")
        )
    
    @classmethod
    def from_mcp_request(cls, data: Dict) -> "Message":
        """Create a Message object from an MCP request."""
        messages = data.get("messages", [])
        if not messages:
            return cls(role="user", content="")
        
        last_message = messages[-1]
        return cls(
            role=last_message.get("role", "user"),
            content=last_message.get("content", ""),
            tool_calls=last_message.get("tool_calls")
        )


class IAgentInterface(ABC):
    """
    Interface for agent development.
    Defines the properties and methods that an agent should implement.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the agent."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the agent."""
        pass

    @property
    @abstractmethod
    def status(self) -> AgentStatus:
        """Current status of the agent."""
        pass

    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """List of capabilities the agent has."""
        pass

    @property
    @abstractmethod
    def available_functions(self) -> List[FunctionDefinition]:
        """List of functions the agent can call."""
        pass

    @abstractmethod
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the agent and prepare it for operation."""
        pass

    @abstractmethod
    async def process_message(self, message: Union[str, Dict, Message]) -> Message:
        """Process a message and return a response."""
        pass

    @abstractmethod
    async def call_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a function with the given parameters."""
        pass

    @abstractmethod
    async def terminate(self) -> bool:
        """Terminate the agent's operation."""
        pass


class BaseAgent(IAgentInterface):
    """
    Base class for agents, implementing common properties and methods.
    Compatible with OpenAI Function Calling API, Anthropic MCP, and Cloudflare Agent.
    """

    def __init__(self, weaviate_service: WeaviateService, name: str = None, purpose: str = None, model_config: Dict = None, ):
        self._id = str(uuid.uuid4())
        self._name = name or f"Agent-{self._id[:8]}"
        self._status = AgentStatus.INITIALIZING
        self._purpose = purpose or "A general-purpose assistant agent"
        self._memory = AgentMemory()
        self._conversation_history = []
        self._model_config = model_config or {}
        self._logger = logging.getLogger(f"agent.{self._name}")
        self.weaviate = weaviate_service
        
        # MCP specific attributes
        self._mcp_tools = []
        self._tool_handlers = {}
        
        # Setup AIMon detection with all guardrails
        self.detect = Detect(
            values_returned=['context', 'generated_text'],
            config={
                "hallucination": {"detector_name": "default"},
                "completeness": {"enabled": True},
                "conciseness": {"enabled": True},
                "toxicity": {"enabled": True},
                "instruction_adherence": {"enabled": True}
            }
        )
        
    @property
    def id(self) -> str:
        """Unique identifier for the agent."""
        return self._id

    @property
    def name(self) -> str:
        """Name of the agent."""
        return self._name

    @property
    def status(self) -> AgentStatus:
        """Current status of the agent."""
        return self._status

    @property
    def purpose(self) -> str:
        """System prompt or purpose of the agent."""
        return self._purpose

    @property
    def memory(self) -> AgentMemory:
        """Memory of the agent."""
        return self._memory

    @property
    def model_name(self) -> str:
        """Name of the model used by the agent."""
        return self._model_config.get("model_name", "")

    @property
    def model_provider(self) -> str:
        """Provider of the model (OpenAI, Anthropic, etc.)"""
        return self._model_config.get("provider", "")

    @property
    def model_url(self) -> str:
        """URL of the model API."""
        return self._model_config.get("url", "")

    @property
    def capabilities(self) -> List[str]:
        """List of capabilities the agent has."""
        return ["text_response", "function_calling"]

    @property
    def available_functions(self) -> List[FunctionDefinition]:
        """List of functions the agent can call."""
        return []

    @property
    def history(self) -> List[Dict]:
        """Conversation history."""
        return [msg.to_dict() for msg in self._conversation_history]

    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the agent and prepare it for operation."""
        if config:
            self._model_config.update(config)
        
        # Validate required configuration
        required_fields = ["provider", "model_name"]
        if not all(field in self._model_config for field in required_fields):
            self._status = AgentStatus.ERROR
            self._logger.error(f"Missing required configuration: {required_fields}")
            return False
            
        for func_def in self.available_functions:
            self.register_tool(func_def)
            
        return True

    @detect
    async def process_message(self, message: Union[str, Dict, Message], context: str = "") -> Message:
        """
        Process a message with AIMon guardrails.
        The @detect decorator will automatically check for:
        - Hallucinations
        - Completeness
        - Conciseness
        - Toxicity
        - Instruction adherence
        """
        try:
            # Convert the message to a Message object if it's not already
            if isinstance(message, str):
                msg = Message(role="user", content=message)
            elif isinstance(message, dict):
                msg = Message(
                    role=message.get("role", "user"),
                    content=message.get("content", ""),
                    function_call=message.get("function_call"),
                    tool_calls=message.get("tool_calls"),
                    tool_results=message.get("tool_results")
                )
            else:
                msg = message
            
            # Add to conversation history
            self._conversation_history.append(msg)
            
            # Check for tool calls and execute them if necessary
            if msg.tool_calls:
                tool_results = []
                for tool_call in msg.tool_calls:
                    tool_id = tool_call.get("id")
                    tool_name = tool_call.get("function", {}).get("name")
                    tool_args = tool_call.get("function", {}).get("arguments", "{}")
                    
                    try:
                        # Parse arguments
                        args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
                        
                        # Call the tool
                        result = await self.call_function(tool_name, args)
                        
                        # Add to results
                        tool_results.append({
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps(result) if not isinstance(result, str) else result
                        })
                    except Exception as e:
                        self._logger.error(f"Error executing tool {tool_name}: {str(e)}")
                        tool_results.append({
                            "tool_call_id": tool_id,
                            "role": "tool",
                            "name": tool_name,
                            "content": json.dumps({"error": str(e)})
                        })
                
                # Create a tool response message
                if tool_results:
                    tool_response = Message(
                        role="assistant",
                        content="",
                        tool_results=tool_results
                    )
                    self._conversation_history.append(tool_response)
            
            # Process the message (to be implemented by subclasses)
            response = await self._generate_response(msg)
            
            # Add response to conversation history
            self._conversation_history.append(response)
            
            return context, response
        except Exception as e:
            self._logger.error(f"Error processing message: {e}")
            return context, Message(
                role="assistant",
                content="I encountered an error processing your message."
            )

    async def _generate_response(self, message: Message) -> Message:
        """Generate a response to the message."""
        # This should be implemented by subclasses
        return Message(role="assistant", content="This method should be implemented by subclasses.")

    async def call_function(self, function_name: str, parameters: Dict[str, Any]) -> Any:
        """Call a function with the given parameters."""
        self._logger.info(f"Calling function {function_name} with parameters: {parameters}")
        
        # First check for registered tool handlers (MCP approach)
        if function_name in self._tool_handlers:
            try:
                handler = self._tool_handlers[function_name]
                return await handler(parameters)
            except Exception as e:
                self._logger.error(f"Error in tool handler for {function_name}: {str(e)}")
                return {"error": str(e)}
        
        # Then check for method-based function implementations
        for func_def in self.available_functions:
            if func_def.name == function_name:
                # Function exists, now check for implementation
                method_name = f"_func_{function_name}"
                if hasattr(self, method_name) and callable(getattr(self, method_name)):
                    try:
                        return await getattr(self, method_name)(**parameters)
                    except Exception as e:
                        self._logger.error(f"Error in function {function_name}: {str(e)}")
                        return {"error": str(e)}
                
                return {"error": f"Function {function_name} is defined but not implemented."}
        
        self._logger.warning(f"Function {function_name} is not available")
        return {"error": f"Function {function_name} is not available."}
        
    def register_tool(self, name: str, description: str, parameters: Dict, 
                     handler: Callable[[Dict[str, Any]], Awaitable[Any]]) -> None:
        """
        Register a tool handler function for MCP compatibility.
        
        Args:
            name: Tool name
            description: Tool description
            parameters: JSON Schema for the tool parameters
            handler: Async function to handle the tool call
        """
        # Register the tool definition
        self._mcp_tools.append({
            "name": name,
            "description": description,
            "input_schema": parameters
        })
        
        # Register the handler
        self._tool_handlers[name] = handler
        
        self._logger.info(f"Registered tool: {name}")
        
    def register_function_as_tool(self, function_def: FunctionDefinition) -> None:
        """Register a function definition as a tool."""
        method_name = f"_func_{function_def.name}"
        if hasattr(self, method_name) and callable(getattr(self, method_name)):
            # Create a wrapper that routes the call to the function
            async def tool_handler(parameters: Dict[str, Any]) -> Any:
                return await getattr(self, method_name)(**parameters)
            
            # Register the tool
            self._mcp_tools.append({
                "name": function_def.name,
                "description": function_def.description,
                "input_schema": function_def.parameters
            })
            
            # Register the handler
            self._tool_handlers[function_def.name] = tool_handler
            
            self._logger.info(f"Registered function as tool: {function_def.name}")

    async def terminate(self) -> bool:
        """Terminate the agent's operation."""
        self._status = AgentStatus.TERMINATED
        # Clean up resources
        self._memory.clear_short_term()
        return True

    def to_openai_format(self) -> Dict:
        """Convert agent to OpenAI compatible format."""
        functions = []
        for func in self.available_functions:
            functions.append({
                "name": func.name,
                "description": func.description,
                "parameters": func.parameters
            })
            
        return {
            "id": self.id,
            "name": self.name,
            "description": self.purpose,
            "functions": functions
        }
    
    def to_anthropic_format(self) -> Dict:
        """Convert agent to Anthropic MCP compatible format."""
        # Combine registered tools and available functions
        tools = self._mcp_tools.copy()
        
        # Add tools from available functions that aren't already registered
        registered_tool_names = {tool["name"] for tool in tools}
        for func in self.available_functions:
            if func.name not in registered_tool_names:
                tools.append({
                    "name": func.name,
                    "description": func.description,
                    "input_schema": func.parameters
                })
            
        return {
            "id": self.id,
            "name": self.name,
            "system_prompt": self.purpose,
            "tools": tools
        }
    
    def to_cloudflare_format(self) -> Dict:
        """Convert agent to Cloudflare Agent compatible format."""
        capabilities = []
        for func in self.available_functions:
            capabilities.append({
                "name": func.name,
                "description": func.description,
                "schema": {
                    "type": "object",
                    "properties": func.parameters.get("properties", {}),
                    "required": func.required or []
                }
            })
            
        return {
            "agent_id": self.id,
            "name": self.name,
            "description": self.purpose,
            "capabilities": capabilities
        }


class MasterAgent(BaseAgent):
    """
    Master agent that can coordinate other agents.
    """
    
    def __init__(self, name: str = "Master", purpose: str = None):
        super().__init__(name=name, purpose=purpose or "Coordinates other agents and manages task delegation")
        self.agents = {}  # Dictionary to store child agents
        
    @property
    def available_functions(self) -> List[FunctionDefinition]:
        return [
            FunctionDefinition(
                name="delegate_task",
                description="Delegate a task to a specific agent",
                parameters={
                    "type": "object",
                    "properties": {
                        "agent_id": {
                            "type": "string",
                            "description": "ID of the agent to delegate the task to"
                        },
                        "task": {
                            "type": "string",
                            "description": "Task description"
                        }
                    }
                },
                required=["agent_id", "task"]
            ),
            FunctionDefinition(
                name="list_agents",
                description="List all available agents",
                parameters={
                    "type": "object",
                    "properties": {}
                }
            )
        ]
        
    async def _generate_response(self, message: Message) -> Message:
        """Generate a response to the message."""
        # In a real implementation, this would call the model API
        # For now, we'll just return a placeholder
        content = f"Master agent acknowledged: {message.content}"
        return Message(role="assistant", content=content)
    
    async def register_agent(self, agent: BaseAgent) -> bool:
        """Register a new agent with the master agent."""
        if agent.id in self.agents:
            return False
        
        self.agents[agent.id] = agent
        return True
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the master agent."""
        if agent_id not in self.agents:
            return False
        
        del self.agents[agent_id]
        return True
    
    async def _func_delegate_task(self, agent_id: str, task: str) -> Dict:
        """Delegate a task to a specific agent."""
        if agent_id not in self.agents:
            return {"success": False, "error": f"Agent {agent_id} not found"}
        
        agent = self.agents[agent_id]
        if agent.status != AgentStatus.ACTIVE:
            return {"success": False, "error": f"Agent {agent_id} is not active"}
        
        try:
            response = await agent.process_message(task)
            return {
                "success": True,
                "agent_id": agent_id,
                "response": response.to_dict()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _func_list_agents(self) -> Dict:
        """List all available agents."""
        agent_list = []
        for agent_id, agent in self.agents.items():
            agent_list.append({
                "id": agent.id,
                "name": agent.name,
                "status": agent.status.value,
                "capabilities": agent.capabilities
            })
        
        return {"agents": agent_list}


class MCPServer:
    """
    Model Context Protocol Server implementation.
    Allows agents to be exposed as MCP-compatible endpoints.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("mcp_server")
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Set up API routes."""
        self.app.add_routes([
            web.post("/v1/{agent_id}/chat/completions", self._handle_chat_completion),
            web.get("/v1/models", self._handle_list_models),
            web.get("/v1/agents", self._handle_list_agents),
            web.post("/v1/agents", self._handle_register_agent),
            web.delete("/v1/agents/{agent_id}", self._handle_unregister_agent),
            web.get("/health", self._handle_health_check)
        ])
    
    async def start(self) -> None:
        """Start the MCP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        self.logger.info(f"MCP Server started on http://{self.host}:{self.port}")
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the MCP server."""
        self.agents[agent.id] = agent
        self.logger.info(f"Registered agent: {agent.name} (ID: {agent.id})")
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the MCP server."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False
    
    async def _handle_chat_completion(self, request: web.Request) -> web.Response:
        """Handle chat completion requests."""
        agent_id = request.match_info.get("agent_id")
        if agent_id not in self.agents:
            return web.json_response(
                {"error": f"Agent {agent_id} not found"}, 
                status=404
            )
        
        agent = self.agents[agent_id]
        if agent.status != AgentStatus.ACTIVE:
            return web.json_response(
                {"error": f"Agent {agent_id} is not active"}, 
                status=400
            )
        
        try:
            data = await request.json()
            
            # Handle MCP request format
            message = Message.from_mcp_request(data)
            response = await agent.process_message(message)
            
            # Convert to MCP response format
            mcp_response = {
                "id": f"chatcmpl-{str(uuid.uuid4())}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": agent.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": response.role,
                            "content": response.content
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # Would be calculated in a real implementation
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }
            
            # Add tool calls if present
            if response.tool_calls:
                mcp_response["choices"][0]["message"]["tool_calls"] = response.tool_calls
            
            # Add function call if present (legacy format)
            if response.function_call:
                mcp_response["choices"][0]["message"]["function_call"] = response.function_call
                mcp_response["choices"][0]["finish_reason"] = "function_call"
            
            return web.json_response(mcp_response)
            
        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}")
            return web.json_response(
                {"error": f"Error processing request: {str(e)}"}, 
                status=500
            )
    
    async def _handle_list_models(self, request: web.Request) -> web.Response:
        """Handle model listing requests."""
        models = []
        for agent_id, agent in self.agents.items():
            models.append({
                "id": f"{agent.model_name}-{agent_id[:8]}",
                "name": agent.model_name,
                "provider": agent.model_provider,
                "owned_by": "Oflo",
                "capabilities": agent.capabilities
            })
        
        return web.json_response({
            "object": "list",
            "data": models
        })
    
    async def _handle_list_agents(self, request: web.Request) -> web.Response:
        """Handle agent listing requests."""
        agents_list = []
        for agent_id, agent in self.agents.items():
            agents_list.append({
                "id": agent.id,
                "name": agent.name,
                "status": agent.status.value,
                "capabilities": agent.capabilities,
                "model": agent.model_name,
                "provider": agent.model_provider
            })
        
        return web.json_response({
            "object": "list",
            "data": agents_list
        })
    
    async def _handle_register_agent(self, request: web.Request) -> web.Response:
        """Handle agent registration requests."""
        try:
            data = await request.json()
            agent_type = data.get("type", "base")
            name = data.get("name")
            purpose = data.get("purpose")
            model_config = data.get("model_config", {})
            
            # Create the appropriate agent type
            if agent_type == "marketing":
                agent = MarketingAgent(name=name, purpose=purpose)
            elif agent_type == "sales":
                agent = SalesAgent(name=name, purpose=purpose)
            elif agent_type == "master":
                agent = MasterAgent(name=name, purpose=purpose)
            else:  # Default to base agent
                agent = BaseAgent(name=name, purpose=purpose)
            
            # Initialize the agent
            success = await agent.initialize(model_config)
            if not success:
                return web.json_response(
                    {"error": "Failed to initialize agent"}, 
                    status=400
                )
            
            # Register the agent
            await self.register_agent(agent)
            
            return web.json_response({
                "id": agent.id,
                "name": agent.name,
                "status": agent.status.value,
                "message": "Agent registered successfully"
            })
            
        except Exception as e:
            self.logger.error(f"Error registering agent: {str(e)}")
            return web.json_response(
                {"error": f"Error registering agent: {str(e)}"}, 
                status=500
            )
    
    async def _handle_unregister_agent(self, request: web.Request) -> web.Response:
        """Handle agent unregistration requests."""
        agent_id = request.match_info.get("agent_id")
        success = await self.unregister_agent(agent_id)
        
        if success:
            return web.json_response({
                "success": True,
                "message": f"Agent {agent_id} unregistered successfully"
            })
        else:
            return web.json_response(
                {"error": f"Agent {agent_id} not found"}, 
                status=404
            )
    
    async def _handle_health_check(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "agents_count": len(self.agents)
        })


class MCPClient:
    """
    Client for interacting with MCP-compatible services.
    Can be used to communicate with remote agents.
    """
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def chat_completion(self, agent_id: str, messages: List[Dict]) -> Dict:
        """Send a chat completion request to an agent."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/v1/{agent_id}/chat/completions"
        data = {
            "messages": messages,
            "model": "agent-model"  # Placeholder, will be determined by the server
        }
        
        headers = await self._get_headers()
        async with self.session.post(url, json=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Error from MCP server: {error_text}")
            
            return await response.json()
    
    async def list_agents(self) -> List[Dict]:
        """List all available agents."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/v1/agents"
        headers = await self._get_headers()
        
        async with self.session.get(url, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Error from MCP server: {error_text}")
            
            result = await response.json()
            return result.get("data", [])
    
    async def register_agent(self, agent_config: Dict) -> Dict:
        """Register a new agent."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        url = f"{self.base_url}/v1/agents"
        headers = await self._get_headers()
        
        async with self.session.post(url, json=agent_config, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"Error from MCP server: {error_text}")
            
            return await response.json()
