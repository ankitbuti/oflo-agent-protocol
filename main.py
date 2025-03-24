import logging
import asyncio
import json
from oflo_agent_protocol import MasterAgent, MCPServer

print("---\n\n--------oflo ai kickoff")

import asyncio
from crewai import Task
from dotenv import load_dotenv
import os

from oflo_agent_protocol.adapters.crewai_adapter import CrewAIAdapter
from services.weaviate import WeaviateService

# Load environment variables
load_dotenv()

# Initialize services
weaviate_service = WeaviateService(url="...", api_key="...")
crew_adapter = CrewAIAdapter(weaviate_service)



# Example usage
async def main():
    # Start an MCP server
    server = MCPServer(port=8080)
    await server.start()
    
    # Create a master agent
    master = MasterAgent()
    await master.initialize({"provider": "anthropic", "model_name": "claude-3-sonnet-20240229"})
    await server.register_agent(master)
   
       # Initialize services
    weaviate_service = WeaviateService(
        url=os.getenv('WEAVIATE_URL'),
        api_key=os.getenv('WEAVIATE_API_KEY')
    )
    
    # Create CrewAI adapter
    crew_adapter = CrewAIAdapter(weaviate_service)
    
    # Create research agent
    researcher = await crew_adapter.create_agent(
        name="Research Expert",
        role="Senior Research Analyst",
        goal="Find and analyze information about given topics",
        backstory="Expert at gathering and analyzing information from various sources",
        llm_config={
            "model": "gpt-4-turbo-preview",
            "temperature": 0.7
        }
    )
    
    # Create writing agent
    writer = await crew_adapter.create_agent(
        name="Content Writer",
        role="Professional Content Writer",
        goal="Create engaging and informative content",
        backstory="Experienced writer specializing in clear and compelling content",
        llm_config={
            "model": "gpt-4-turbo-preview",
            "temperature": 0.8
        }
    )
    
    # Create tasks
    research_task = Task(
        description="Research the latest developments in AI agent frameworks",
        expected_output="A comprehensive summary of current AI agent frameworks",
        context="Focus on practical applications and real-world use cases"
    )
    
    writing_task = Task(
        description="Write a blog post about AI agent frameworks",
        expected_output="A well-structured blog post about AI agent frameworks",
        context="Use the research provided to create engaging content"
    )
    
    # Create crew
    crew = crew_adapter.create_crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        crew_name="content_creation_crew",
        process="sequential"  # Tasks will be executed in sequence
    )
    
    # Execute crew's tasks
    result = await crew.kickoff()
    
            # Create agents
        agent = await crew_adapter.create_agent(
            name="My Agent",
            role="Specific Role",
            goal="Agent's Goal",
            llm_config={"model": "gpt-4"}
        )

        # Create and run crews
        crew = crew_adapter.create_crew(
            agents=[agent],
            tasks=[task],
            process="sequential"
        )
        result = await crew.kickoff()
    
    # Get agent memories
    research_memory = await crew_adapter.get_agent_memory(
        agent=researcher,
        query="AI frameworks"
    )
    
    # Share knowledge between agents
    await crew_adapter.share_knowledge(
        content="Key insights about AI frameworks",
        source_agent=researcher,
        target_agents=[writer],
        knowledge_type="research_findings",
        confidence=0.9
    )
    
    # Get conversation history
    writer_conversation = await crew_adapter.get_agent_conversation(writer)
    
    print("Crew execution result:", result)
    print("\nResearch memories:", research_memory)
    print("\nWriter conversation:", writer_conversation)

    
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