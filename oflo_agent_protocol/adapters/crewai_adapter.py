from typing import List, Dict, Any, Optional
from crewai import Agent as CrewAgent, Task, Crew
from ..agent import BaseAgent, Message, AgentStatus
from ..factory import OfloAgentFactory
from services.weaviate import WeaviateService

class CrewAIAdapter:
    """
    Adapter to make Oflo agents compatible with CrewAI's framework.
    Allows using Oflo agents within CrewAI's orchestration system.
    """
    
    def __init__(self, weaviate_service: WeaviateService):
        self.weaviate = weaviate_service
        self.factory = OfloAgentFactory()
        self._crews: Dict[str, Crew] = {}
        
    async def create_agent(self, 
                    name: str,
                    role: str,
                    goal: str,
                    backstory: str = None,
                    tools: List[Any] = None,
                    llm_config: Dict[str, Any] = None,
                    **kwargs) -> CrewAgent:
        """
        Create a CrewAI agent that wraps an Oflo agent.
        
        Args:
            name: Agent name
            role: Agent role/purpose
            goal: Agent's goal
            backstory: Agent's backstory
            tools: List of tools available to the agent
            llm_config: Language model configuration
            **kwargs: Additional agent configuration
        """
        # Create Oflo agent
        oflo_agent = self.factory.create_agent(
            name=name,
            purpose=role,
            model_config=llm_config or {},
            capabilities=tools or []
        )
        
        # Store agent in Weaviate
        agent_id = await self.weaviate.store_agent(oflo_agent)
        
        # Create CrewAI agent wrapper
        crew_agent = CrewAgent(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory or role,
            tools=tools or [],
            llm_config=llm_config or {},
            **kwargs
        )
        
        # Attach Oflo agent to CrewAI agent
        crew_agent._oflo_agent = oflo_agent
        crew_agent._oflo_agent_id = agent_id
        
        # Override CrewAI agent's execute method to use Oflo agent
        original_execute = crew_agent.execute
        
        async def execute_wrapper(task: Task) -> str:
            # Update agent status
            await self.weaviate.update_agent_status(agent_id, AgentStatus.WORKING)
            
            try:
                # Convert task to message
                message = Message(
                    content=f"{task.description}\nContext: {task.context or ''}",
                    role="user"
                )
                
                # Process message through Oflo agent
                response = await oflo_agent.process_message(message)
                
                # Store conversation in Weaviate
                await self.weaviate.store_message(agent_id, message)
                await self.weaviate.store_message(agent_id, response)
                
                # Update agent status
                await self.weaviate.update_agent_status(agent_id, AgentStatus.IDLE)
                
                return response.content
                
            except Exception as e:
                await self.weaviate.update_agent_status(agent_id, AgentStatus.ERROR)
                raise e
                
        crew_agent.execute = execute_wrapper
        return crew_agent
        
    def create_crew(self, 
                   agents: List[CrewAgent],
                   tasks: List[Task],
                   crew_name: str = None,
                   process: str = "sequential",
                   **kwargs) -> Crew:
        """
        Create a CrewAI crew with Oflo agents.
        
        Args:
            agents: List of CrewAI agents (with Oflo agents attached)
            tasks: List of tasks for the crew
            crew_name: Name of the crew
            process: Process type ('sequential' or 'hierarchical')
            **kwargs: Additional crew configuration
        """
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=process,
            **kwargs
        )
        
        if crew_name:
            self._crews[crew_name] = crew
            
        return crew
        
    def get_crew(self, crew_name: str) -> Optional[Crew]:
        """Get a previously created crew by name."""
        return self._crews.get(crew_name)
        
    async def get_agent_memory(self, agent: CrewAgent, query: str = None) -> List[Dict]:
        """Get an agent's memory from Weaviate."""
        if not hasattr(agent, '_oflo_agent_id'):
            raise ValueError("Agent is not an Oflo-adapted agent")
            
        if query:
            return await self.weaviate.search_memory(
                query=query,
                agent_id=agent._oflo_agent_id
            )
        else:
            # Get all memory
            return await self.weaviate.search_memory(
                query="",
                agent_id=agent._oflo_agent_id,
                limit=1000
            )
            
    async def get_agent_conversation(self, agent: CrewAgent) -> List[Dict]:
        """Get an agent's conversation history from Weaviate."""
        if not hasattr(agent, '_oflo_agent_id'):
            raise ValueError("Agent is not an Oflo-adapted agent")
            
        return await self.weaviate.get_agent_conversation(agent._oflo_agent_id)
        
    async def share_knowledge(self, 
                            content: str,
                            source_agent: CrewAgent,
                            target_agents: List[CrewAgent],
                            knowledge_type: str = "shared",
                            confidence: float = 1.0) -> str:
        """Share knowledge between agents through the knowledge graph."""
        if not hasattr(source_agent, '_oflo_agent_id'):
            raise ValueError("Source agent is not an Oflo-adapted agent")
            
        target_ids = []
        for agent in target_agents:
            if not hasattr(agent, '_oflo_agent_id'):
                raise ValueError("Target agent is not an Oflo-adapted agent")
            target_ids.append(agent._oflo_agent_id)
            
        return await self.weaviate.add_knowledge(
            content=content,
            knowledge_type=knowledge_type,
            source=f"agent:{source_agent._oflo_agent_id}",
            confidence=confidence,
            agent_ids=target_ids
        ) 