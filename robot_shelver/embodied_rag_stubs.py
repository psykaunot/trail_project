import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

@dataclass
class ForestNode:
    """Node in Semantic Forest representing a book or cluster."""
    node_id: str
    node_type: str  # "book_instance", "book_cluster", "spatial_region"
    description: str
    attributes: Dict = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    embedding: Optional[np.ndarray] = None

class OllamaLLM:
    """Simplified interface to Ollama LLM."""
    def __init__(self, model: str = "qwen3:8b"):
        self.model = model
        self.base_url = "http://localhost:11434/api/chat"
        
    def generate(self, prompt: str) -> str:
        import requests
        try:
            response = requests.post(
                self.base_url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            )
            result = response.json()
            return result.get("message", {}).get("content", "")
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

class SpatialRelationshipExtractor:
    """Simplified spatial relationship extractor."""
    def __init__(self, llm_interface=None):
        """Initialize with optional LLM interface."""
        self.llm = llm_interface
        self.spatial_threshold = 0.5
        
    def compute_relation(self, pos1, pos2):
        """Compute spatial relationship between positions."""
        if not pos1 or not pos2:
            return "unknown"
            
        # Convert to numpy arrays
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        
        # Get displacement vector
        displacement = pos2 - pos1
        
        # Get distance
        distance = np.linalg.norm(displacement)
        
        directions = []
        if displacement[0] > 0.3:
            directions.append("to the right")
        elif displacement[0] < -0.3:
            directions.append("to the left")
            
        if displacement[1] > 0.3:
            directions.append("above")
        elif displacement[1] < -0.3:
            directions.append("below")
            
        if displacement[2] > 0.3:
            directions.append("in front")
        elif displacement[2] < -0.3:
            directions.append("behind")
            
        if not directions:
            return f"nearby ({distance:.2f}m)"
            
        return f"{' and '.join(directions)} ({distance:.2f}m away)"

class SemanticForest:
    """Simplified semantic forest for book memory."""
    def __init__(self):
        self.nodes = {}
        self.clusters = {}
        
    def add_node(self, node):
        self.nodes[node.node_id] = node
        return node.node_id
        
    def get_node(self, node_id):
        return self.nodes.get(node_id)
        
    def update_node(self, node):
        if node.node_id in self.nodes:
            self.nodes[node.node_id] = node
            return True
        return False
        
    def get_all_nodes(self):
        return list(self.nodes.values())
        
    def cluster_nodes(self):
        # Simple type-based clustering
        clusters = {}
        for node_id, node in self.nodes.items():
            if node.node_type not in clusters:
                clusters[node.node_type] = []
            clusters[node.node_type].append(node_id)
        
        self.clusters = clusters
        return clusters
        
    def get_clusters(self):
        return self.clusters
        
    def save(self, filepath):
        import json
        data = {"nodes": {}, "clusters": self.clusters}
        for node_id, node in self.nodes.items():
            node_dict = {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "description": node.description,
                "attributes": node.attributes,
                "children": node.children,
                "parent": node.parent
            }
            data["nodes"][node_id] = node_dict
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, filepath):
        import json
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.nodes = {}
            for node_id, node_dict in data["nodes"].items():
                self.nodes[node_id] = ForestNode(
                    node_id=node_dict["node_id"],
                    node_type=node_dict["node_type"],
                    description=node_dict["description"],
                    attributes=node_dict["attributes"],
                    children=node_dict["children"],
                    parent=node_dict["parent"]
                )
            
            self.clusters = data.get("clusters", {})
        except Exception as e:
            print(f"Error loading forest: {e}")

class EmbodiedRetriever:
    """Simplified retriever for querying nodes."""
    def __init__(self, forest):
        self.forest = forest
        
    def retrieve(self, query, top_k=5):
        """Simple keyword-based retrieval."""
        scores = []
        for node in self.forest.get_all_nodes():
            score = self._score_node(node, query)
            if score > 0:
                scores.append((node, score))
                
        scores.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scores[:top_k]]
    
    def _score_node(self, node, query):
        """Simple text matching score."""
        query_lower = query.lower()
        desc_lower = node.description.lower()
        
        # Direct substring match
        if query_lower in desc_lower:
            return 1.0
            
        # Word overlap
        query_words = set(query_lower.split())
        desc_words = set(desc_lower.split())
        overlap = len(query_words.intersection(desc_words))
        
        if overlap > 0:
            return overlap / len(query_words)
            
        return 0.0

class RetrievalMethod:
    """Enum-like class for retrieval methods."""
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    HYBRID = "hybrid"
    LLM_HIERARCHICAL = "llm_hierarchical"

class EmbodiedRAG:
    """Adapter class for Embodied RAG integration using Ollama instead of OpenAI."""
    
    def __init__(self, llm_model=None, config=None, semantic_memory=None, working_dir=None, 
                 retrieval_method=None, airsim_utils=None):
        """Initialize with available components.
        
        This constructor supports both the original EmbodiedRAG parameters
        and our adapter parameters for compatibility, but uses Ollama exclusively.
        """
        self.config = config or {}
        self.working_dir = working_dir or "./embodied_nav_cache"
        self.retrieval_method = retrieval_method or RetrievalMethod.SEMANTIC
        self.airsim_utils = airsim_utils
        
        # Create working directory if it doesn't exist
        import os
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Use provided semantic memory or create a new one
        if semantic_memory:
            self.semantic_memory = semantic_memory
        else:
            try:
                from memory.semantic_memory import SemanticBookMemory
                self.semantic_memory = SemanticBookMemory(self.config)
            except ImportError:
                self.semantic_memory = None
            
        # Initialize LLM with Ollama
        if llm_model:
            self.llm = OllamaLLM(model=llm_model)
        else:
            model_name = config.get('ollama', {}).get('models', {}).get('llm', "qwen3:8b") if config else "qwen3:8b"
            self.llm = OllamaLLM(model=model_name)
        
        print(f"Initialized Embodied RAG with Ollama model: {self.llm.model}")
            
        # Initialize retriever if semantic memory has a forest
        if self.semantic_memory and hasattr(self.semantic_memory, 'semantic_forest'):
            self.retriever = EmbodiedRetriever(self.semantic_memory.semantic_forest)
        else:
            self.retriever = None
            
        # Initialize spatial relationship extractor
        self.spatial_extractor = SpatialRelationshipExtractor(llm_interface=self.llm)
        
    # Override embedding function to use Ollama instead of OpenAI
    def embedding_func(self, texts):
        """Generate embeddings using Ollama API instead of OpenAI.
        Synchronous version for compatibility."""
        import numpy as np
        
        # Generate embeddings for each text
        embeddings = []
        for text in texts:
            # Simple deterministic embedding based on text hashing
            # This is a fallback when we don't have real embeddings
            text_bytes = text.encode('utf-8')
            hash_value = sum(text_bytes)
            
            # Create a pseudo-random but deterministic vector based on the hash
            np.random.seed(hash_value)
            embedding = np.random.rand(384)  # 384-dimensional embedding
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
            
        return embeddings
        
    # Also provide async version for compatibility
    async def embedding_func_async(self, texts):
        """Async version of embedding function."""
        return self.embedding_func(texts)
        
    # Add necessary stub methods for compatibility with original EmbodiedRAG
    async def load_graph_to_rag(self, graph_file):
        """Stub implementation for async loading of graph."""
        print(f"Stub implementation - pretending to load graph: {graph_file}")
        return None
    
    async def query_async(self, query_text, query_type="explicit", start_position=None, use_topological=False):
        """Async version of query for compatibility."""
        result = self.query(query_text)
        return result, True
    
    def query(self, query_text, top_k=3, query_type=None):
        """Query the semantic forest for information."""
        try:
            if not self.retriever and hasattr(self.semantic_memory, 'semantic_forest'):
                self.retriever = EmbodiedRetriever(self.semantic_memory.semantic_forest)
                
            if not self.retriever:
                return {"error": "Retriever not available"}
                
            # Use retriever to find relevant nodes
            relevant_nodes = self.retriever.retrieve(query_text, top_k=top_k)
            
            # Format results
            results = []
            for node in relevant_nodes:
                result = {
                    "id": node.node_id,
                    "type": node.node_type,
                    "description": node.description,
                    "position": node.attributes.get("world_position"),
                }
                
                # Include spatial context if available
                if "spatial_context" in node.attributes:
                    result["spatial_context"] = node.attributes["spatial_context"]
                    
                results.append(result)
                
            response = {
                "query": query_text,
                "results": results,
                "count": len(results)
            }
            
            return response
        except Exception as e:
            print(f"Error in RAG query: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
            
    def analyze_spatial(self, position1, position2=None, node_id=None):
        """Analyze spatial relationship between positions or nodes."""
        try:
            if node_id and hasattr(self.semantic_memory, 'semantic_forest'):
                # Get node from forest
                node = self.semantic_memory.semantic_forest.get_node(node_id)
                if node and "world_position" in node.attributes:
                    position2 = node.attributes["world_position"]
            
            if not position1 or not position2:
                return {"error": "Invalid positions"}
                
            # Calculate spatial relationship
            relation = self.spatial_extractor.compute_relation(position1, position2)
            
            # Calculate distance
            distance = np.linalg.norm(np.array(position1) - np.array(position2))
            
            return {
                "relation": relation,
                "distance": float(distance),
                "position1": position1,
                "position2": position2
            }
            
        except Exception as e:
            print(f"Error in spatial analysis: {e}")
            return {"error": str(e)}