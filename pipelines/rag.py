import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

# Vector RAG components
@dataclass
class Document:
    """Represents a document in the RAG system."""
    id: str
    content: str
    metadata: Dict[str, Any] = None
    embedding: Optional[np.ndarray] = None

class VectorStore:
    """Base class for vector stores used in RAG."""
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.documents = {}
        self.embeddings = {}
        self.logger = logging.getLogger("vector_store")
    
    async def add_document(self, document: Document) -> str:
        """Add a document to the vector store."""
        if document.embedding is None:
            self.logger.warning(f"Document {document.id} has no embedding")
            return None
        
        self.documents[document.id] = document
        self.embeddings[document.id] = document.embedding
        return document.id
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents using cosine similarity."""
        if not self.embeddings:
            return []
        
        results = []
        for doc_id, embedding in self.embeddings.items():
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append((self.documents[doc_id], similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class ShapRAG:
    """
    SHAP (SHapley Additive exPlanations) RAG implementation.
    Provides explainability for RAG results by attributing importance to retrieved documents.
    """
    
    def __init__(self, vector_store: VectorStore, embedding_model: Any):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.logger = logging.getLogger("shap_rag")
    
    async def query(self, query: str, top_k: int = 5, num_samples: int = 20) -> Dict[str, Any]:
        """
        Query the RAG system with SHAP explanations.
        
        Args:
            query: The query string
            top_k: Number of top documents to retrieve
            num_samples: Number of permutation samples for SHAP calculation
            
        Returns:
            Dictionary containing retrieved documents and their SHAP values
        """
        # Embed the query
        query_embedding = await self._get_embedding(query)
        
        # Get top documents
        results = await self.vector_store.search(query_embedding, top_k=top_k)
        if not results:
            return {"documents": [], "shap_values": []}
        
        # Calculate SHAP values
        documents = [doc for doc, _ in results]
        shap_values = await self._calculate_shap_values(query, documents, num_samples)
        
        return {
            "documents": [{"id": doc.id, "content": doc.content, "metadata": doc.metadata} for doc in documents],
            "shap_values": shap_values,
            "similarities": [sim for _, sim in results]
        }
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the embedding model."""
        # This is a placeholder - implement based on your embedding model
        try:
            embedding = self.embedding_model.embed(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return np.zeros(self.vector_store.embedding_dim)
    
    async def _calculate_shap_values(self, query: str, documents: List[Document], num_samples: int) -> List[float]:
        """
        Calculate SHAP values for documents using permutation sampling.
        
        This is a simplified implementation of SHAP for RAG:
        1. Create random subsets of documents
        2. For each subset, calculate the "utility" (e.g., relevance score)
        3. Calculate marginal contributions of each document
        4. Average the marginal contributions to get SHAP values
        """
        n = len(documents)
        shap_values = [0.0] * n
        
        # Generate random permutations for sampling
        for _ in range(num_samples):
            # Random permutation of document indices
            perm = np.random.permutation(n)
            
            # Calculate marginal contributions
            prev_score = 0
            for i, idx in enumerate(perm):
                # Subset of documents up to and including current document
                subset_docs = [documents[perm[j]] for j in range(i+1)]
                
                # Calculate utility score for this subset
                current_score = await self._calculate_utility(query, subset_docs)
                
                # Marginal contribution is the difference in utility
                marginal = current_score - prev_score
                shap_values[idx] += marginal
                prev_score = current_score
        
        # Average the contributions
        shap_values = [val / num_samples for val in shap_values]
        return shap_values
    
    async def _calculate_utility(self, query: str, documents: List[Document]) -> float:
        """
        Calculate utility of a set of documents for answering the query.
        This is a placeholder - implement based on your specific needs.
        """
        if not documents:
            return 0.0
        
        # Simple implementation: average similarity to query
        query_embedding = await self._get_embedding(query)
        similarities = [
            self.vector_store._cosine_similarity(query_embedding, doc.embedding)
            for doc in documents
        ]
        return sum(similarities) / len(similarities)


# Graph RAG components
class Node:
    """Represents a node in the knowledge graph."""
    
    def __init__(self, id: str, type: str, properties: Dict[str, Any] = None):
        self.id = id
        self.type = type
        self.properties = properties or {}
        self.embedding = None
    
    def set_embedding(self, embedding: np.ndarray) -> None:
        """Set the embedding for this node."""
        self.embedding = embedding


class Edge:
    """Represents an edge in the knowledge graph."""
    
    def __init__(self, source_id: str, target_id: str, type: str, properties: Dict[str, Any] = None):
        self.source_id = source_id
        self.target_id = target_id
        self.type = type
        self.properties = properties or {}


class KnowledgeGraph:
    """Simple knowledge graph implementation for RAG."""
    
    def __init__(self):
        self.nodes = {}  # id -> Node
        self.edges = []  # List of Edge objects
        self.node_connections = {}  # id -> list of connected node ids
        self.logger = logging.getLogger("knowledge_graph")
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        if node.id not in self.node_connections:
            self.node_connections[node.id] = []
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            self.logger.warning(f"Cannot add edge: source or target node not found")
            return
        
        self.edges.append(edge)
        
        # Update connections
        if edge.source_id not in self.node_connections:
            self.node_connections[edge.source_id] = []
        if edge.target_id not in self.node_connections:
            self.node_connections[edge.target_id] = []
        
        self.node_connections[edge.source_id].append(edge.target_id)
        self.node_connections[edge.target_id].append(edge.source_id)
    
    def get_neighbors(self, node_id: str, max_distance: int = 1) -> List[Node]:
        """Get neighboring nodes up to a certain distance."""
        if node_id not in self.nodes:
            return []
        
        visited = set([node_id])
        neighbors = []
        current_level = [node_id]
        
        for _ in range(max_distance):
            next_level = []
            for current_id in current_level:
                for neighbor_id in self.node_connections.get(current_id, []):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbors.append(self.nodes[neighbor_id])
                        next_level.append(neighbor_id)
            
            if not next_level:
                break
            
            current_level = next_level
        
        return neighbors


class GraphRAG:
    """
    Graph-based RAG implementation.
    Uses a knowledge graph to enhance retrieval with structural information.
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph, embedding_model: Any):
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model
        self.logger = logging.getLogger("graph_rag")
    
    async def query(self, query: str, top_k: int = 5, max_hops: int = 2) -> Dict[str, Any]:
        """
        Query the graph-based RAG system.
        
        Args:
            query: The query string
            top_k: Number of top nodes to retrieve initially
            max_hops: Maximum number of hops for graph traversal
            
        Returns:
            Dictionary containing retrieved nodes and their relevance scores
        """
        # Embed the query
        query_embedding = await self._get_embedding(query)
        
        # Find initial nodes by embedding similarity
        initial_nodes = await self._find_similar_nodes(query_embedding, top_k)
        if not initial_nodes:
            return {"nodes": [], "scores": []}
        
        # Expand with graph traversal
        expanded_nodes = await self._expand_with_graph(initial_nodes, max_hops)
        
        # Re-rank all nodes
        ranked_nodes = await self._rank_nodes(query_embedding, expanded_nodes)
        
        return {
            "nodes": [{"id": node.id, "type": node.type, "properties": node.properties} 
                     for node, _ in ranked_nodes],
            "scores": [score for _, score in ranked_nodes]
        }
    
    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using the embedding model."""
        # This is a placeholder - implement based on your embedding model
        try:
            embedding = self.embedding_model.embed(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return np.zeros(768)  # Assuming 768-dim embeddings
    
    async def _find_similar_nodes(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[Node, float]]:
        """Find nodes with embeddings similar to the query."""
        results = []
        
        for node_id, node in self.knowledge_graph.nodes.items():
            if node.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, node.embedding)
                results.append((node, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    async def _expand_with_graph(self, initial_nodes: List[Tuple[Node, float]], max_hops: int) -> List[Node]:
        """Expand the initial set of nodes using graph traversal."""
        expanded = set()
        for node, _ in initial_nodes:
            expanded.add(node.id)
        
        # Add initial nodes
        result = [node for node, _ in initial_nodes]
        
        # Expand with neighbors
        for node, _ in initial_nodes:
            neighbors = self.knowledge_graph.get_neighbors(node.id, max_distance=max_hops)
            for neighbor in neighbors:
                if neighbor.id not in expanded:
                    expanded.add(neighbor.id)
                    result.append(neighbor)
        
        return result
    
    async def _rank_nodes(self, query_embedding: np.ndarray, nodes: List[Node]) -> List[Tuple[Node, float]]:
        """Rank nodes by relevance to the query."""
        results = []
        
        for node in nodes:
            if node.embedding is not None:
                # Base score is embedding similarity
                similarity = self._cosine_similarity(query_embedding, node.embedding)
                
                # Could add other ranking factors here
                # For example, node importance in the graph, etc.
                
                results.append((node, similarity))
            else:
                # Nodes without embeddings get a low score
                results.append((node, 0.1))
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
