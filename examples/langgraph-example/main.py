import asyncio
from dotenv import load_dotenv
import os
from typing import Dict, Any

from oflo_agent_protocol.adapters.langgraph_adapter import LangGraphAdapter
from services.weaviate import WeaviateService

# Load environment variables
load_dotenv()

# Example tool
async def search_web(query: str) -> str:
    """Simulated web search tool."""
    return f"Found results for: {query}"

async def main():
    # Initialize services
    weaviate_service = WeaviateService(
        url=os.getenv('WEAVIATE_URL'),
        api_key=os.getenv('WEAVIATE_API_KEY')
    )
    
    # Create LangGraph adapter
    graph_adapter = LangGraphAdapter(weaviate_service)
    
    # Add agents to the graph
    await graph_adapter.add_agent(
        name="researcher",
        purpose="Research and analyze information",
        model_config={
            "model": "gpt-4-turbo-preview",
            "temperature": 0.7
        },
        tools=[search_web]
    )
    
    await graph_adapter.add_agent(
        name="writer",
        purpose="Write content based on research",
        model_config={
            "model": "gpt-4-turbo-preview",
            "temperature": 0.8
        }
    )
    
    await graph_adapter.add_agent(
        name="editor",
        purpose="Edit and improve content",
        model_config={
            "model": "gpt-4-turbo-preview",
            "temperature": 0.6
        }
    )
    
    # Define conditional edges for the workflow
    conditional_edges = {
        "researcher": {
            "writer": lambda state: "research_complete" in state["shared_memory"]
        },
        "writer": {
            "editor": lambda state: "draft_complete" in state["shared_memory"]
        }
    }
    
    # Create workflow
    workflow = graph_adapter.create_workflow(
        entry_point="researcher",
        conditional_edges=conditional_edges
    )
    
    # Initialize shared memory
    shared_memory = {
        "topic": "AI agent frameworks",
        "style": "technical blog post",
        "target_audience": "developers"
    }
    
    # Execute workflow
    final_state = await graph_adapter.execute_workflow(
        workflow=workflow,
        initial_message="Research and write a blog post about AI agent frameworks",
        shared_memory=shared_memory
    )
    
    # Get agent memories
    research_memory = await graph_adapter.get_agent_memory(
        agent_name="researcher",
        query="AI frameworks"
    )
    
    # Share knowledge between agents
    await graph_adapter.share_knowledge(
        content="Key insights about AI frameworks",
        source_agent="researcher",
        target_agents=["writer", "editor"],
        knowledge_type="research_findings",
        confidence=0.9
    )
    
    # Print results
    print("Workflow execution completed!")
    print("\nFinal messages:")
    for msg in final_state["messages"]:
        print(f"{msg.type}: {msg.content}")
    
    print("\nResearch memories:", research_memory)
    print("\nShared memory:", final_state["shared_memory"])

if __name__ == "__main__":
    asyncio.run(main()) 