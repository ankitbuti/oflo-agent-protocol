from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.agents import AgentAction, AgentFinish
import operator
from functools import partial

from ..agent import BaseAgent, Message, AgentStatus
from ..factory import OfloAgentFactory
from services.weaviate import WeaviateService

class AgentState(TypedDict):
    """State maintained between agent steps."""
    messages: List[BaseMessage]
    actions: List[AgentAction]
    next_agent: str
    current_agent: str
    task_status: str
    shared_memory: Dict[str, Any]

class LangGraphAdapter:
    """
    Adapter to make Oflo agents compatible with LangGraph's framework.
    Enables building complex agent workflows and state machines.
    """
    
    def __init__(self, weaviate_service: WeaviateService):
        self.weaviate = weaviate_service
        self.factory = OfloAgentFactory()
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_ids: Dict[str, str] = {}
        self._tools: Dict[str, Any] = {}
        
    async def add_agent(self,
                       name: str,
                       purpose: str,
                       model_config: Dict[str, Any] = None,
                       tools: List[Any] = None) -> str:
        """Add an agent to the graph."""
        # Create Oflo agent
        agent = self.factory.create_agent(
            name=name,
            purpose=purpose,
            model_config=model_config or {},
            capabilities=tools or []
        )
        
        # Store agent in Weaviate
        agent_id = await self.weaviate.store_agent(agent)
        
        self._agents[name] = agent
        self._agent_ids[name] = agent_id
        
        if tools:
            self._tools.update({tool.__name__: tool for tool in tools})
            
        return agent_id

    def _create_agent_node(self, agent_name: str):
        """Create a node function for the agent in the graph."""
        agent = self._agents[agent_name]
        agent_id = self._agent_ids[agent_name]
        
        async def agent_node(state: AgentState) -> dict:
            """Process agent's step in the workflow."""
            # Update agent status
            await self.weaviate.update_agent_status(agent_id, AgentStatus.WORKING)
            
            try:
                # Convert last message to Oflo format
                last_message = state["messages"][-1]
                oflo_message = Message(
                    content=last_message.content,
                    role="user" if isinstance(last_message, HumanMessage) else "assistant"
                )
                
                # Process through Oflo agent
                response = await agent.process_message(oflo_message)
                
                # Store conversation in Weaviate
                await self.weaviate.store_message(agent_id, oflo_message)
                await self.weaviate.store_message(agent_id, response)
                
                # Update agent status
                await self.weaviate.update_agent_status(agent_id, AgentStatus.IDLE)
                
                # Add response to state
                state["messages"].append(AIMessage(content=response.content))
                
                # Check for next agent or completion
                if "TASK_COMPLETE" in response.content:
                    return {"next_agent": END}
                elif any(agent in response.content for agent in self._agents.keys()):
                    next_agent = next(agent for agent in self._agents.keys() 
                                    if agent in response.content)
                    return {"next_agent": next_agent}
                else:
                    return {"next_agent": state.get("next_agent", END)}
                
            except Exception as e:
                await self.weaviate.update_agent_status(agent_id, AgentStatus.ERROR)
                raise e
                
        return agent_node

    def _create_tool_node(self):
        """Create a node for tool execution in the graph."""
        tool_executor = ToolExecutor(self._tools)
        
        async def tool_node(state: AgentState) -> dict:
            """Execute tools based on agent actions."""
            actions = state["actions"]
            if not actions:
                return {"next_agent": state["current_agent"]}
                
            action = actions[-1]
            
            if isinstance(action, AgentFinish):
                return {"next_agent": END}
                
            # Execute tool
            observation = await tool_executor.execute(action.tool, action.tool_input)
            
            # Add result to messages
            state["messages"].append(
                HumanMessage(content=f"Tool {action.tool} returned: {observation}")
            )
            
            return {"next_agent": state["current_agent"]}
            
        return tool_node

    def create_workflow(self, 
                       entry_point: str,
                       conditional_edges: Dict[str, Dict[str, Any]] = None) -> StateGraph:
        """
        Create a workflow graph with the registered agents.
        
        Args:
            entry_point: Name of the agent to start with
            conditional_edges: Dict of agent names to their edge conditions
        """
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Add agent nodes
        for agent_name in self._agents:
            workflow.add_node(agent_name, self._create_agent_node(agent_name))
            
        # Add tool execution node if tools exist
        if self._tools:
            workflow.add_node("tools", self._create_tool_node())
            
        # Add conditional edges
        if conditional_edges:
            for agent_name, edges in conditional_edges.items():
                for target, condition in edges.items():
                    workflow.add_conditional_edges(
                        agent_name,
                        partial(operator.eq, "next_agent"),
                        {
                            target: target,
                            END: END
                        }
                    )
        else:
            # Default sequential flow
            workflow.add_edge(entry_point, "tools")
            workflow.add_edge("tools", entry_point)
            
        # Set entry point
        workflow.set_entry_point(entry_point)
        
        return workflow.compile()

    async def execute_workflow(self, 
                             workflow: StateGraph,
                             initial_message: str,
                             shared_memory: Dict[str, Any] = None) -> AgentState:
        """Execute a compiled workflow."""
        # Initialize state
        state = AgentState(
            messages=[HumanMessage(content=initial_message)],
            actions=[],
            next_agent="",
            current_agent="",
            task_status="in_progress",
            shared_memory=shared_memory or {}
        )
        
        # Execute workflow
        final_state = await workflow.invoke(state)
        
        return final_state

    async def get_agent_memory(self, agent_name: str, query: str = None) -> List[Dict]:
        """Get an agent's memory from Weaviate."""
        if agent_name not in self._agent_ids:
            raise ValueError(f"Unknown agent: {agent_name}")
            
        return await self.weaviate.search_memory(
            query=query,
            agent_id=self._agent_ids[agent_name]
        )

    async def share_knowledge(self,
                            content: str,
                            source_agent: str,
                            target_agents: List[str],
                            knowledge_type: str = "shared",
                            confidence: float = 1.0) -> str:
        """Share knowledge between agents through the knowledge graph."""
        if source_agent not in self._agent_ids:
            raise ValueError(f"Unknown source agent: {source_agent}")
            
        target_ids = []
        for agent in target_agents:
            if agent not in self._agent_ids:
                raise ValueError(f"Unknown target agent: {agent}")
            target_ids.append(self._agent_ids[agent])
            
        return await self.weaviate.add_knowledge(
            content=content,
            knowledge_type=knowledge_type,
            source=f"agent:{self._agent_ids[source_agent]}",
            confidence=confidence,
            agent_ids=target_ids
        ) 