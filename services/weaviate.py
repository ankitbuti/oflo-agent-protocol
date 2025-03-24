from typing import Dict, List, Optional, Union, Any
import weaviate
from weaviate.util import generate_uuid5
import json
import logging
from datetime import datetime
from dataclasses import asdict

from oflo_agent_protocol.agent import AgentStatus, Message, AgentMemory, BaseAgent

logger = logging.getLogger(__name__)

class WeaviateService:
    """
    Weaviate middleware service for managing agent data, states, and memory.
    Provides vector search and knowledge graph capabilities for the Oflo Agent Protocol.
    """
    
    def __init__(self, url: str, api_key: Optional[str] = None):
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=weaviate.AuthApiKey(api_key) if api_key else None
        )
        self._setup_schema()
    
    def _setup_schema(self) -> None:
        """Setup the Weaviate schema for agents, memory, and knowledge graph."""
        
        # Agent class for storing agent metadata and state
        agent_class = {
            "class": "Agent",
            "description": "Represents an agent in the system",
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": True
                }
            },
            "properties": [
                {
                    "name": "name",
                    "dataType": ["string"],
                    "description": "Name of the agent"
                },
                {
                    "name": "purpose",
                    "dataType": ["text"],
                    "description": "Purpose/system prompt of the agent",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "status",
                    "dataType": ["string"],
                    "description": "Current status of the agent"
                },
                {
                    "name": "model_config",
                    "dataType": ["text"],
                    "description": "Model configuration as JSON"
                },
                {
                    "name": "capabilities",
                    "dataType": ["string[]"],
                    "description": "List of agent capabilities"
                },
                {
                    "name": "last_active",
                    "dataType": ["date"],
                    "description": "Last activity timestamp"
                }
            ]
        }
        
        # Memory class for storing agent memory entries
        memory_class = {
            "class": "Memory",
            "description": "Represents an agent's memory entry",
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": True
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Memory content",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "type",
                    "dataType": ["string"],
                    "description": "Memory type (short_term/long_term)"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When the memory was created"
                },
                {
                    "name": "key",
                    "dataType": ["string"],
                    "description": "Memory key identifier"
                },
                {
                    "name": "belongsTo",
                    "dataType": ["Agent"],
                    "description": "Reference to the agent this memory belongs to"
                }
            ]
        }
        
        # Message class for conversation history
        message_class = {
            "class": "Message",
            "description": "Represents a message in agent conversations",
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": True
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Message content",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "role",
                    "dataType": ["string"],
                    "description": "Message role (user/assistant/system/function)"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When the message was sent"
                },
                {
                    "name": "function_call",
                    "dataType": ["text"],
                    "description": "Function call data as JSON"
                },
                {
                    "name": "tool_calls",
                    "dataType": ["text"],
                    "description": "Tool calls data as JSON"
                },
                {
                    "name": "belongsTo",
                    "dataType": ["Agent"],
                    "description": "Reference to the agent this message belongs to"
                }
            ]
        }
        
        # Knowledge class for agent knowledge graph
        knowledge_class = {
            "class": "Knowledge",
            "description": "Represents a knowledge node in the agent knowledge graph",
            "vectorizer": "text2vec-transformers",
            "moduleConfig": {
                "text2vec-transformers": {
                    "vectorizeClassName": True
                }
            },
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Knowledge content",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": False,
                            "vectorizePropertyName": False
                        }
                    }
                },
                {
                    "name": "type",
                    "dataType": ["string"],
                    "description": "Type of knowledge"
                },
                {
                    "name": "source",
                    "dataType": ["string"],
                    "description": "Source of the knowledge"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When the knowledge was added"
                },
                {
                    "name": "confidence",
                    "dataType": ["number"],
                    "description": "Confidence score of the knowledge"
                },
                {
                    "name": "relatedTo",
                    "dataType": ["Knowledge"],
                    "description": "References to related knowledge nodes"
                },
                {
                    "name": "usedBy",
                    "dataType": ["Agent"],
                    "description": "References to agents using this knowledge"
                }
            ]
        }

        # Create schema classes
        for class_obj in [agent_class, memory_class, message_class, knowledge_class]:
            try:
                if not self.client.schema.exists(class_obj["class"]):
                    self.client.schema.create_class(class_obj)
                    logger.info(f"Created schema class: {class_obj['class']}")
            except Exception as e:
                logger.error(f"Error creating schema class {class_obj['class']}: {str(e)}")

    async def store_agent(self, agent: BaseAgent) -> str:
        """Store agent data in Weaviate."""
        agent_data = {
            "name": agent.name,
            "purpose": agent.purpose,
            "status": agent.status.value,
            "model_config": json.dumps(agent._model_config),
            "capabilities": agent.capabilities,
            "last_active": datetime.now().isoformat()
        }
        
        # Generate deterministic UUID based on agent ID
        uuid = generate_uuid5(agent.id)
        
        try:
            self.client.data_object.create(
                "Agent",
                agent_data,
                uuid
            )
            return uuid
        except Exception as e:
            logger.error(f"Error storing agent: {str(e)}")
            raise

    async def store_memory(self, agent_id: str, memory: AgentMemory) -> None:
        """Store agent memory in Weaviate."""
        # Store short-term memory
        for key, value in memory.short_term.items():
            memory_data = {
                "content": json.dumps(value) if not isinstance(value, str) else value,
                "type": "short_term",
                "key": key,
                "timestamp": datetime.now().isoformat(),
                "belongsTo": [{"beacon": f"weaviate://localhost/Agent/{agent_id}"}]
            }
            
            try:
                self.client.data_object.create("Memory", memory_data)
            except Exception as e:
                logger.error(f"Error storing short-term memory: {str(e)}")
        
        # Store long-term memory
        for key, value in memory.long_term.items():
            memory_data = {
                "content": json.dumps(value) if not isinstance(value, str) else value,
                "type": "long_term",
                "key": key,
                "timestamp": datetime.now().isoformat(),
                "belongsTo": [{"beacon": f"weaviate://localhost/Agent/{agent_id}"}]
            }
            
            try:
                self.client.data_object.create("Memory", memory_data)
            except Exception as e:
                logger.error(f"Error storing long-term memory: {str(e)}")

    async def store_message(self, agent_id: str, message: Message) -> str:
        """Store a conversation message in Weaviate."""
        message_data = {
            "content": message.content,
            "role": message.role,
            "timestamp": message.timestamp,
            "belongsTo": [{"beacon": f"weaviate://localhost/Agent/{agent_id}"}]
        }
        
        if message.function_call:
            message_data["function_call"] = json.dumps(message.function_call)
        if message.tool_calls:
            message_data["tool_calls"] = json.dumps(message.tool_calls)
            
        uuid = generate_uuid5(message.message_id)
        
        try:
            self.client.data_object.create(
                "Message",
                message_data,
                uuid
            )
            return uuid
        except Exception as e:
            logger.error(f"Error storing message: {str(e)}")
            raise

    async def add_knowledge(self, content: str, knowledge_type: str, source: str,
                          confidence: float = 1.0, related_ids: List[str] = None,
                          agent_ids: List[str] = None) -> str:
        """Add a knowledge node to the knowledge graph."""
        knowledge_data = {
            "content": content,
            "type": knowledge_type,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence
        }
        
        # Add references to related knowledge
        if related_ids:
            knowledge_data["relatedTo"] = [
                {"beacon": f"weaviate://localhost/Knowledge/{id}"} for id in related_ids
            ]
            
        # Add references to agents
        if agent_ids:
            knowledge_data["usedBy"] = [
                {"beacon": f"weaviate://localhost/Agent/{id}"} for id in agent_ids
            ]
            
        try:
            uuid = self.client.data_object.create("Knowledge", knowledge_data)
            return uuid
        except Exception as e:
            logger.error(f"Error adding knowledge: {str(e)}")
            raise

    async def search_memory(self, query: str, agent_id: str = None, 
                          memory_type: str = None, limit: int = 10) -> List[Dict]:
        """Search agent memory using vector similarity."""
        where_filter = None
        if agent_id or memory_type:
            where_filter = {
                "operator": "And",
                "operands": []
            }
            if agent_id:
                where_filter["operands"].append({
                    "path": ["belongsTo", "Agent", "id"],
                    "operator": "Equal",
                    "valueString": agent_id
                })
            if memory_type:
                where_filter["operands"].append({
                    "path": ["type"],
                    "operator": "Equal",
                    "valueString": memory_type
                })

        try:
            result = (
                self.client.query
                .get("Memory", ["content", "type", "key", "timestamp"])
                .with_where(where_filter)
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
                .do()
            )
            
            return result["data"]["Get"]["Memory"]
        except Exception as e:
            logger.error(f"Error searching memory: {str(e)}")
            return []

    async def search_knowledge(self, query: str, knowledge_type: str = None,
                             min_confidence: float = 0.0, limit: int = 10) -> List[Dict]:
        """Search knowledge graph using vector similarity."""
        where_filter = None
        if knowledge_type or min_confidence > 0.0:
            where_filter = {
                "operator": "And",
                "operands": []
            }
            if knowledge_type:
                where_filter["operands"].append({
                    "path": ["type"],
                    "operator": "Equal",
                    "valueString": knowledge_type
                })
            if min_confidence > 0.0:
                where_filter["operands"].append({
                    "path": ["confidence"],
                    "operator": "GreaterThanEqual",
                    "valueNumber": min_confidence
                })

        try:
            result = (
                self.client.query
                .get("Knowledge", ["content", "type", "source", "confidence", "timestamp"])
                .with_where(where_filter)
                .with_near_text({"concepts": [query]})
                .with_limit(limit)
                .do()
            )
            
            return result["data"]["Get"]["Knowledge"]
        except Exception as e:
            logger.error(f"Error searching knowledge: {str(e)}")
            return []

    async def get_agent_state(self, agent_id: str) -> Optional[Dict]:
        """Get agent state from Weaviate."""
        try:
            result = (
                self.client.query
                .get("Agent", ["name", "purpose", "status", "model_config", 
                             "capabilities", "last_active"])
                .with_id(agent_id)
                .do()
            )
            
            if result["data"]["Get"]["Agent"]:
                return result["data"]["Get"]["Agent"][0]
            return None
        except Exception as e:
            logger.error(f"Error getting agent state: {str(e)}")
            return None

    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        """Update agent status in Weaviate."""
        try:
            self.client.data_object.update(
                "Agent",
                {
                    "status": status.value,
                    "last_active": datetime.now().isoformat()
                },
                agent_id
            )
            return True
        except Exception as e:
            logger.error(f"Error updating agent status: {str(e)}")
            return False

    async def get_agent_conversation(self, agent_id: str, limit: int = 100) -> List[Dict]:
        """Get agent conversation history."""
        try:
            result = (
                self.client.query
                .get("Message", ["content", "role", "timestamp", 
                               "function_call", "tool_calls"])
                .with_where({
                    "path": ["belongsTo", "Agent", "id"],
                    "operator": "Equal",
                    "valueString": agent_id
                })
                .with_sort({"path": ["timestamp"], "order": "desc"})
                .with_limit(limit)
                .do()
            )
            
            return result["data"]["Get"]["Message"]
        except Exception as e:
            logger.error(f"Error getting conversation history: {str(e)}")
            return []
