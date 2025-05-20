import sys
import os
import time
import json
import uuid
import traceback
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from memory.book_matcher import BookMatcher
from memory.spatial_relations import SpatialRelationTracker, SpatialRelation, RelationType

# Add Embodied_RAG to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../Embodied_RAG'))

# Use actual Embodied_RAG if available, fall back to stubs if not
try:
    # Import from actual Embodied_RAG implementation
    from embodied_nav.spatial_relationship_extractor import SpatialRelationshipExtractor
    from embodied_nav.embodied_retriever import EmbodiedRetriever
    from embodied_nav.ollama_llm import OllamaLLM
    print("Using actual Embodied_RAG implementation")
    USE_ACTUAL_RAG = True
except ImportError:
    # Fall back to stub implementation
    from embodied_rag_stubs import ForestNode, SemanticForest, EmbodiedRetriever, OllamaLLM, SpatialRelationshipExtractor
    print("Using stub Embodied_RAG implementation")
    USE_ACTUAL_RAG = False

# Define ForestNode if we're using the actual implementation
if USE_ACTUAL_RAG:
    @dataclass
    class ForestNode:
        """Node in Semantic Forest representing a book or cluster."""
        node_id: str
        node_type: str  # "book_instance", "book_cluster", "spatial_region"
        description: str
        attributes: Dict = None
        children: List[str] = None
        parent: Optional[str] = None
        embedding: Optional[Any] = None
        
        def __post_init__(self):
            if self.attributes is None:
                self.attributes = {}
            if self.children is None:
                self.children = []

    # Create SemanticForest if not available
    class SemanticForest:
        """Adaptation layer for Embodied_RAG graph to forest structure."""
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
            """Save forest to disk with comprehensive Vector3 handling."""
            try:
                # Create simpler data structure with just nodes
                data = {"nodes": {}}

                # Process each node carefully
                for node_id, node in self.nodes.items():
                    # Process attributes recursively
                    processed_attributes = self._process_for_json(node.attributes) if node.attributes else {}

                    # Create processed node entry
                    node_dict = {
                        "node_id": node.node_id,
                        "node_type": node.node_type,
                        "description": node.description,
                        "attributes": processed_attributes,
                        "children": node.children,
                        "parent": node.parent
                    }
                    data["nodes"][node_id] = node_dict

                # Save data with simple format
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                    print(f"Successfully saved to {filepath}")

            except Exception as e:
                print(f"Error in save method: {e}")
                traceback.print_exc()

                # Save minimal backup
                try:
                    backup_data = {
                        "error": str(e),
                        "timestamp": time.time(),
                        "backup": True
                    }
                    with open(filepath, 'w') as f:
                        json.dump(backup_data, f)
                        print(f"Saved minimal backup to {filepath}")
                except Exception as e2:
                    print(f"Backup save also failed: {e2}")

        def _process_for_json(self, obj):
            """Process any object to make it JSON serializable."""
            # Handle None
            if obj is None:
                return None

            # Handle Vector3
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Vector3':
                return [float(obj[0]), float(obj[1]), float(obj[2])]

            # Handle Quaternion
            if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Quaternion':
                return {
                    "scalar": float(obj.scalar),
                    "vector": [float(obj.vector.x), float(obj.vector.y), float(obj.vector.z)]
                }

            # Handle lists recursively
            if isinstance(obj, list):
                return [self._process_for_json(item) for item in obj]

            # Handle dictionaries recursively
            if isinstance(obj, dict):
                return {k: self._process_for_json(v) for k, v in obj.items()}

            # Return other types as is
            return obj
                
        def load(self, filepath):
            try:
                # Check if file exists
                if not os.path.exists(filepath):
                    print(f"Warning: File {filepath} does not exist. Initializing with empty forest.")
                    self.nodes = {}
                    return
                    
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.nodes = {}
                
                # Handle different forest data structures
                # Standard format with 'nodes' key
                if "nodes" in data:
                    nodes_data = data["nodes"]
                    print(f"Loading forest with {len(nodes_data)} nodes from '{filepath}'")
                    
                    # Process nodes
                    for node_id, node_dict in nodes_data.items():
                        try:
                            self.nodes[node_id] = ForestNode(
                                node_id=node_dict["node_id"],
                                node_type=node_dict["node_type"],
                                description=node_dict["description"],
                                attributes=node_dict["attributes"],
                                children=node_dict["children"],
                                parent=node_dict["parent"]
                            )
                        except KeyError as ke:
                            print(f"Warning: Missing key in node {node_id}: {ke}")
                            # Skip this node but continue processing others
                            continue
                
                # Alternative format with 'semantic_forest_nodes' key
                elif "semantic_forest_nodes" in data:
                    nodes_data = data["semantic_forest_nodes"]
                    print(f"Loading forest with {len(nodes_data)} nodes (semantic_forest_nodes format) from '{filepath}'")
                    
                    # Process semantic_forest_nodes
                    for node_id, node_dict in nodes_data.items():
                        try:
                            self.nodes[node_id] = ForestNode(
                                node_id=node_dict["node_id"],
                                node_type=node_dict["node_type"],
                                description=node_dict["description"],
                                attributes=node_dict["attributes"],
                                children=node_dict["children"],
                                parent=node_dict["parent"]
                            )
                        except KeyError as ke:
                            print(f"Warning: Missing key in node {node_id}: {ke}")
                            # Skip this node but continue processing others
                            continue
                else:
                    print(f"Warning: No recognized node structure found in {filepath}. Initializing with empty forest.")
                    # In this case, we keep the empty self.nodes dictionary
                
                # Load clusters if available
                self.clusters = data.get("clusters", {})
                
                print(f"Successfully loaded {len(self.nodes)} nodes from {filepath}")
                
            except json.JSONDecodeError as je:
                print(f"Error decoding JSON from forest file: {je}")
                # Initialize with empty data rather than failing
                self.nodes = {}
            except Exception as e:
                print(f"Error loading forest: {e}")
                import traceback
                traceback.print_exc()
                # Initialize with empty data rather than failing
                self.nodes = {}

class SemanticMemory:
    """Main semantic memory class for storing and retrieving book information."""
    def __init__(self, config=None):
        self.config = config or {}
        self.semantic_forest = SemanticForest()
        self.memories = {}
        self.recent_discoveries = []
        self.last_novel_discovery = time.time()
        self.mission_start_time = time.time()
        self.retriever = None
        
        # Initialize Ollama LLM interface with default model if config is None
        if self.config:
            model = self.config.get('ollama', {}).get('models', {}).get('llm', "qwen3:8b")
        else:
            model = "qwen3:8b"
        self.ollama_llm = OllamaLLM(model=model)
        
        # Initialize spatial relationship extractor
        self.spatial_extractor = SpatialRelationshipExtractor(llm_interface=self.ollama_llm)
        
        # Initialize enhanced book matcher
        self.book_matcher = BookMatcher(self.config)
        
        # Initialize spatial relationship tracker
        self.spatial_tracker = SpatialRelationTracker(self.config)
        
        # Load configuration parameters
        self.novelty_window = self.config.get('agent', {}).get('novelty_window', 30)
        self.novelty_count_threshold = self.config.get('agent', {}).get('novelty_count_threshold', 3)
        self.coverage_threshold = self.config.get('agent', {}).get('coverage_threshold', 0.9)
        
        # Enhanced matching parameters
        self.matching_threshold = self.config.get('matching', {}).get('threshold', 0.7)
        self.confidence_update_rate = self.config.get('matching', {}).get('confidence_update_rate', 0.8)
        
        # Book observation statistics
        self.observation_stats = {
            'total_observations': 0,
            'unique_books': 0,
            'duplicate_matches': 0,
            'observation_history': []
        }
        
        # Load existing memory if available
        memory_path = self.config.get('paths', {}).get('memory_save', 'semantic_forest_save.json')
        self.load(memory_path)
        
        # Initialize retriever
        self._initialize_retriever()
        
    def _initialize_retriever(self):
        """Initialize the retriever for querying the semantic forest."""
        try:
            # Initialize retriever for semantic querying
            self.retriever = EmbodiedRetriever(self.semantic_forest)
            print("Semantic retriever initialized")
        except Exception as e:
            print(f"Error initializing retriever: {e}")
            self.retriever = None
    
    def add_observation(self, book_data, camera_state):
        """Add a book observation to semantic memory."""
        # Create a unique book ID with timestamp and UUID
        current_time = time.time()
        book_id = f"book_{int(current_time*1000)}_{str(uuid.uuid4())[:8]}"
        
        # Extract physical properties with expanded attribute set
        description = book_data.get('description', 'Unknown book')
        world_position = book_data.get('world_position')
        
        # Enhance with additional metadata
        observation_id = f"obs_{int(current_time*1000)}"
        physical_properties = self._extract_physical_properties(book_data)
        visual_features = self._extract_visual_features(book_data)
        
        # Create enhanced forest node
        node = ForestNode(
            node_id=book_id,
            node_type="book_instance",
            description=description,
            attributes={
                # Core properties
                "world_position": world_position,
                "bbox": book_data.get('bbox'),
                "timestamp": current_time,
                "creation_time": current_time,  # When first observed (immutable)
                "camera_position": camera_state,
                "confidence": book_data.get("confidence", 1.0),
                "detection_method": book_data.get("detection_method", "unknown"),
                
                # Enhanced properties
                "physical_properties": physical_properties,
                "visual_features": visual_features,
                "observation_history": [{
                    "id": observation_id,
                    "timestamp": current_time,
                    "camera_state": camera_state,
                    "confidence": book_data.get("confidence", 1.0)
                }]
            }
        )
        
        # Extract enhanced spatial relationships if there are other books
        if world_position and len(self.semantic_forest.get_all_nodes()) > 0:
            spatial_context = self._extract_spatial_context(
                position=world_position,
                source_id=book_id,
                book_description=description
            )
            if spatial_context:
                node.attributes["spatial_context"] = spatial_context
                
                # Store a simplified version for quick access
                if spatial_context.get("nearby_books"):
                    node.attributes["nearby_book_ids"] = [
                        book["book_id"] for book in spatial_context["nearby_books"]
                    ]
                
                # Store arrangement information
                if spatial_context.get("arrangements"):
                    node.attributes["arrangements"] = spatial_context["arrangements"]
        
        # Update tracking statistics
        self.observation_stats['total_observations'] += 1
        self.observation_stats['observation_history'].append({
            "id": observation_id,
            "timestamp": current_time,
            "camera_state": {
                "pan": camera_state.get("pan"),
                "tilt": camera_state.get("tilt")
            }
        })
        
        # Keep observation history limited to prevent memory growth
        max_history = 100
        if len(self.observation_stats['observation_history']) > max_history:
            self.observation_stats['observation_history'] = \
                self.observation_stats['observation_history'][-max_history:]
        
        # Check if this is a unique book using enhanced matching
        is_novel = True
        similar_node = self._find_similar_node(node)
        
        if similar_node:
            # Update existing node with new observation data
            is_novel = False
            book_id = similar_node.node_id
            self._update_node(similar_node, book_data, camera_state)
        else:
            # This is a new unique book
            self.semantic_forest.add_node(node)
            self.recent_discoveries.append(book_id)
            self.last_novel_discovery = current_time
            
            # Update statistics
            self.observation_stats['unique_books'] += 1
            
            # Store in memories dictionary for legacy compatibility
            self.memories[book_id] = node
            
            # Record novelty event
            self._record_novelty_event(book_id, node)
            
            # Update clusters periodically when we have enough books
            if len(self.semantic_forest.get_all_nodes()) >= 3:
                self.update_clusters()
                
            if self.config.get('debug', False):
                print(f"New unique book added: {description[:50]}...")
        
        return book_id, is_novel
        
    def _extract_physical_properties(self, book_data):
        """Extract physical properties from book data."""
        properties = {}
        
        # Size dimensions if available
        if "bbox" in book_data:
            bbox = book_data.get("bbox")
            if bbox and len(bbox) == 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                properties["width_ratio"] = width
                properties["height_ratio"] = height
                properties["aspect_ratio"] = width / height if height > 0 else 0
                
        # Extract info from description
        description = book_data.get("description", "")
        
        # Parse description for size keywords
        size_keywords = ["small", "medium", "large", "tiny", "huge"]
        for keyword in size_keywords:
            if keyword in description.lower():
                properties["size_category"] = keyword
                break
                
        # Parse colors from description
        colors = ["red", "green", "blue", "yellow", "black", "white", 
                 "brown", "orange", "purple", "pink", "gray", "grey"]
        for color in colors:
            if color in description.lower():
                properties["color"] = color
                break
                
        # Get detection confidence
        properties["confidence"] = book_data.get("confidence", 0.0)
        
        return properties
        
    def _extract_visual_features(self, book_data):
        """Extract visual features from book data."""
        # This would normally involve image processing like color histograms, 
        # shape analysis, etc. For now, we'll use basic properties
        features = {}
        
        # Use detection method and quality
        features["detection_method"] = book_data.get("detection_method", "unknown")
        features["detection_quality"] = book_data.get("detection_quality", "unknown")
        
        # For future: add visual embedding if available
        if "visual_embedding" in book_data:
            features["embedding"] = book_data.get("visual_embedding")
        
        return features
        
    def _record_novelty_event(self, book_id, node):
        """Record details about a new book discovery."""
        # This could be expanded to include more contextual information
        # about where and when the book was discovered
        novelty_event = {
            "book_id": book_id,
            "timestamp": time.time(),
            "description": node.description,
            "position": node.attributes.get("world_position"),
            "camera_state": node.attributes.get("camera_position"),
            "confidence": node.attributes.get("confidence")
        }
        
        # We could log this to a separate file for analysis later
        if self.config.get('debug', False):
            print(f"Novelty event recorded: {novelty_event}")
        
    def _find_similar_node(self, new_node):
        """Find an existing node that likely represents the same book."""
        # Get all book instance nodes from the forest
        all_book_nodes = [node for node in self.semantic_forest.get_all_nodes() 
                         if node.node_type == "book_instance"]
        
        if not all_book_nodes:
            return None  # No existing books to compare against
        
        # Define extremely strict spatial proximity thresholds for guaranteed deduplication
        EXACT_MATCH_THRESHOLD = 0.15    # 15cm = exact same book, no question
        STRICT_PROXIMITY_THRESHOLD = 0.25  # 25cm = very likely same book
        CLOSE_PROXIMITY_THRESHOLD = 0.35   # 35cm = possibly same book, needs verification
        
        # First try extremely strict position-based matching if we have coordinates
        if 'world_position' in new_node.attributes and new_node.attributes['world_position']:
            new_pos = np.array(new_node.attributes['world_position'])
            new_confidence = new_node.attributes.get('confidence', 0.0)
            
            # Find books within spatial proximity - categorized by confidence
            exact_matches = []    # Guaranteed matches (very close)
            strict_matches = []   # Very likely matches
            possible_matches = [] # Possible matches needing verification
            
            for node in all_book_nodes:
                if 'world_position' in node.attributes and node.attributes['world_position']:
                    existing_pos = np.array(node.attributes['world_position'])
                    distance = np.linalg.norm(new_pos - existing_pos)
                    existing_confidence = node.attributes.get('confidence', 0.0)
                    
                    # Compute weighted match score - higher confidence observations have more weight
                    combined_confidence = (new_confidence + existing_confidence) / 2
                    match_quality = 1.0 - (distance / CLOSE_PROXIMITY_THRESHOLD)
                    match_score = match_quality * (0.5 + 0.5 * combined_confidence)
                    
                    # Categorize by distance with match score
                    if distance < EXACT_MATCH_THRESHOLD:
                        exact_matches.append((node, distance, match_score))
                        print(f"EXACT position match at {distance:.3f}m (score: {match_score:.3f})")
                    elif distance < STRICT_PROXIMITY_THRESHOLD:
                        strict_matches.append((node, distance, match_score))
                        print(f"STRICT position match at {distance:.3f}m (score: {match_score:.3f})")
                    elif distance < CLOSE_PROXIMITY_THRESHOLD:
                        possible_matches.append((node, distance, match_score))
                        print(f"POSSIBLE position match at {distance:.3f}m (score: {match_score:.3f})")
            
            # First try exact matches (extremely close books)
            if exact_matches:
                # Sort by match score
                exact_matches.sort(key=lambda x: x[2], reverse=True)
                best_match, distance, score = exact_matches[0]
                
                # Found exact position match
                print(f"Using EXACT position match at {distance:.3f}m with score {score:.3f}")
                
                # Update observation statistics
                self.observation_stats['duplicate_matches'] += 1
                return best_match
                
            # Then try strict matches with additional verification
            if strict_matches:
                # Sort by match score
                strict_matches.sort(key=lambda x: x[2], reverse=True)
                best_match, distance, score = strict_matches[0]
                
                # Also check description similarity as a sanity check
                desc_similarity = self.book_matcher._description_similarity(
                    new_node.description, 
                    best_match.description
                )
                
                # If descriptions are minimally similar, accept the match
                if desc_similarity > 0.3:
                    print(f"Using STRICT position match at {distance:.3f}m with score {score:.3f}")
                    print(f"  Description similarity: {desc_similarity:.3f}")
                    
                    # Update observation statistics
                    self.observation_stats['duplicate_matches'] += 1
                    return best_match
                else:
                    print(f"WARNING: Descriptions differ too much for books at {distance:.3f}m")
                    print(f"  Desc similarity: {desc_similarity:.3f}")
                    
                    # If confidence is high enough, trust position more than description
                    # (Most likely same book with incorrect description recognition)
                    if new_confidence > 0.7 or existing_confidence > 0.7:
                        print(f"  Using match despite description mismatch")
                        self.observation_stats['duplicate_matches'] += 1
                        return best_match
            
            # Finally try possible matches with careful verification
            if possible_matches:
                # Sort by match score
                possible_matches.sort(key=lambda x: x[2], reverse=True)
                best_match, distance, score = possible_matches[0]
                
                # Run full book matcher comparison for more careful validation
                is_match, match_score = self.book_matcher.is_same_book(
                    new_node, best_match, threshold=self.matching_threshold
                )
                
                if is_match:
                    print(f"Confirmed POSSIBLE position match at {distance:.3f}m with combined score {match_score:.3f}")
                    
                    # Update observation statistics
                    self.observation_stats['duplicate_matches'] += 1
                    return best_match
                else:
                    print(f"Rejected possible match at {distance:.3f}m with score {match_score:.3f}")
        
        # If no position-based match, fall back to full semantic/visual matching with book matcher
        best_match, match_score = self.book_matcher.get_best_matching_node(
            new_node, 
            all_book_nodes,
            min_threshold=self.matching_threshold
        )
        
        if best_match and match_score >= self.matching_threshold:
            # Found matching book
            print(f"Found semantic-matching book with score {match_score:.2f}")
            
            # Calculate distance for logging
            if ('world_position' in new_node.attributes and 
                'world_position' in best_match.attributes):
                new_pos = new_node.attributes['world_position']
                match_pos = best_match.attributes['world_position']
                distance = np.linalg.norm(np.array(new_pos) - np.array(match_pos))
            
            # Update observation statistics
            self.observation_stats['duplicate_matches'] += 1
            return best_match
            
        # No match found - this is a new unique book
        return None
        
    
    def _update_node(self, node, book_data, camera_state):
        """Update an existing node with new observation data."""
        current_time = time.time()
        
        # Create unique observation ID for this update
        observation_id = f"obs_{int(current_time*1000)}"
        
        # Update basic tracking information
        node.attributes["last_seen"] = current_time
        node.attributes["observation_count"] = node.attributes.get("observation_count", 0) + 1
        
        # Get confidence values
        new_confidence = book_data.get("confidence", 0)
        old_confidence = node.attributes.get("confidence", 0)
        
        # Enhanced position update logic for improved stability
        if book_data.get("world_position"):
            new_pos = book_data.get("world_position")
            
            if "world_position" in node.attributes and node.attributes["world_position"]:
                old_pos = node.attributes["world_position"]
                
                # Calculate position difference for logging and decision making
                pos_difference = np.linalg.norm(np.array(new_pos) - np.array(old_pos))
                
                # Record position change for analysis
                if "position_history" not in node.attributes:
                    node.attributes["position_history"] = []
                    
                # Add this position to history
                node.attributes["position_history"].append({
                    "position": new_pos,
                    "timestamp": current_time,
                    "confidence": new_confidence,
                    "distance_from_previous": float(pos_difference)
                })
                
                # Limit position history size
                max_pos_history = 5
                if len(node.attributes["position_history"]) > max_pos_history:
                    node.attributes["position_history"] = node.attributes["position_history"][-max_pos_history:]
                
                # Position update logic with tiered approach based on distance
                if pos_difference < 0.1:  # Extremely close - basically same position
                    # Use average for stability
                    avg_pos = [
                        (new_pos[0] + old_pos[0]) / 2,
                        (new_pos[1] + old_pos[1]) / 2,
                        (new_pos[2] + old_pos[2]) / 2
                    ]
                    node.attributes["world_position"] = avg_pos
                    print(f"Position nearly identical (diff: {pos_difference:.3f}m), using average")
                    
                elif pos_difference < 0.3:  # Close positions, weighted average
                    # Positions are similar, use confidence-weighted average
                    # More heavily weight the high confidence
                    total_confidence = new_confidence + old_confidence
                    if total_confidence > 0:
                        weight_new = new_confidence / total_confidence
                        weight_old = old_confidence / total_confidence
                    else:
                        weight_new = 0.5
                        weight_old = 0.5
                        
                    # Ensure more recent observation gets some minimum weight
                    weight_new = max(0.3, weight_new)
                    weight_old = 1.0 - weight_new
                    
                    avg_pos = [
                        (new_pos[0] * weight_new) + (old_pos[0] * weight_old),
                        (new_pos[1] * weight_new) + (old_pos[1] * weight_old),
                        (new_pos[2] * weight_new) + (old_pos[2] * weight_old)
                    ]
                    node.attributes["world_position"] = avg_pos
                    print(f"Position similar (diff: {pos_difference:.3f}m), using weighted average")
                    
                elif pos_difference < 0.5:  # Moderately different positions
                    # Check confidence to decide
                    if new_confidence > old_confidence * 1.2:  # New is significantly more confident
                        node.attributes["world_position"] = new_pos
                        print(f"Using new position (diff: {pos_difference:.3f}m)")
                    elif old_confidence > new_confidence * 1.2:  # Old is significantly more confident
                        # Keep old position
                        print(f"Keeping existing position (diff: {pos_difference:.3f}m)")
                    else:
                        # Confidences similar - conservatively use 25/75 weighting favoring old for stability
                        avg_pos = [
                            (new_pos[0] * 0.25) + (old_pos[0] * 0.75),
                            (new_pos[1] * 0.25) + (old_pos[1] * 0.75),
                            (new_pos[2] * 0.25) + (old_pos[2] * 0.75)
                        ]
                        node.attributes["world_position"] = avg_pos
                        print(f"Moderately different positions (diff: {pos_difference:.3f}m)")
                else:
                    # Large position difference (>0.5m)
                    if new_confidence > old_confidence * 1.5:  # New is MUCH more confident
                        node.attributes["world_position"] = new_pos
                        print(f"Accepting new position with large difference ({pos_difference:.3f}m)")
                    else:
                        # Reject new position, too different
                        print(f"Rejecting new position with large difference ({pos_difference:.3f}m)")
            else:
                # No existing position, use the new one
                node.attributes["world_position"] = new_pos
                print(f"Setting initial position: {new_pos}")
                
        # Update the confidence value last, after we've used it for position decisions
        if new_confidence > 0 or old_confidence > 0:
            # Blend confidences with optimized weighting
            # Higher weights for higher values and more recent observations
            if new_confidence > old_confidence:
                # New observation has higher confidence, give it more weight
                updated_confidence = (new_confidence * 0.8) + (old_confidence * 0.2)
            else:
                # Old observation had higher confidence, retain more of it
                updated_confidence = (new_confidence * 0.4) + (old_confidence * 0.6)
                
            node.attributes["confidence"] = updated_confidence
            print(f"Updated confidence: {old_confidence:.2f} -> {updated_confidence:.2f}")
                
        # Update bounding box if the new one is more confident
        if "bbox" in book_data and new_confidence > old_confidence:
            node.attributes["bbox"] = book_data.get("bbox")
        
        # Update or create observation history
        if "observation_history" not in node.attributes:
            node.attributes["observation_history"] = []
            
        # Add this observation to the history
        node.attributes["observation_history"].append({
            "id": observation_id,
            "timestamp": current_time,
            "camera_state": camera_state,
            "confidence": new_confidence,
            "world_position": book_data.get("world_position")
        })
        
        # Limit history length to prevent unbounded growth
        max_history = 10
        if len(node.attributes["observation_history"]) > max_history:
            node.attributes["observation_history"] = node.attributes["observation_history"][-max_history:]
        
        # Update physical properties with most confident values
        if "physical_properties" not in node.attributes:
            node.attributes["physical_properties"] = self._extract_physical_properties(book_data)
        elif new_confidence > old_confidence * 1.1:  # Significantly more confident
            # Update physical properties with new data
            new_properties = self._extract_physical_properties(book_data)
            for key, value in new_properties.items():
                if value is not None:
                    node.attributes["physical_properties"][key] = value
        
        # Update the node in the forest
        self.semantic_forest.update_node(node)
    
    def _check_novelty(self, new_node: ForestNode) -> bool:
        """Check if this book is novel compared to existing nodes"""
        if not new_node.attributes.get("world_position"):
            return True  # Conservative: treat as novel if no position
            
        # Query forest for similar books
        query = f"Book at position {new_node.attributes['world_position']}"
        similar_nodes = self.retriever.retrieve(query, top_k=3)
        
        # Check spatial distance
        position_threshold = 0.5  # default threshold, 0.5 meters
        if self.config and 'semantic_forest' in self.config and 'clustering' in self.config['semantic_forest']:
            position_threshold = self.config['semantic_forest']['clustering'].get('distance_threshold', 0.5)
        new_pos = np.array(new_node.attributes["world_position"])
        
        for node in similar_nodes:
            if node.attributes.get("world_position"):
                existing_pos = np.array(node.attributes["world_position"])
                distance = np.linalg.norm(new_pos - existing_pos)
                
                if distance < position_threshold:
                    # Check description similarity
                    if self._descriptions_similar(new_node.description, node.description):
                        return False  # Not novel
                        
        return True
    
    def _descriptions_similar(self, desc1: str, desc2: str) -> bool:
        """Check if two descriptions refer to the same book"""
        # Simple implementation - can be enhanced with LLM
        if not desc1 or not desc2:
            return False
            
        # Normalize and compare
        desc1_lower = desc1.lower().strip()
        desc2_lower = desc2.lower().strip()
        
        # Exact match or substring
        if desc1_lower == desc2_lower:
            return True
        if desc1_lower in desc2_lower or desc2_lower in desc1_lower:
            return True
            
        return False
    
    def update_clusters(self):
        """Periodically update book clusters."""
        if not hasattr(self, 'last_cluster_update'):
            self.last_cluster_update = time.time()
            return

        # Update clusters every 30 seconds
        if time.time() - self.last_cluster_update < 30:
            return

        self.last_cluster_update = time.time()

        # Perform clustering based on physical proximity
        self.semantic_forest.cluster_nodes()
        print(f"Updated semantic clusters: {len(self.semantic_forest.get_clusters())} clusters")

        

    def _update_forest_structure(self):
        """Enhanced clustering with spatial and semantic grouping."""
        try:
            # Step 1: Get all book instance nodes
            nodes = self.semantic_forest.get_all_nodes()
            book_nodes = [n for n in nodes if n.node_type == "book_instance"]

            if len(book_nodes) < 3:
                return  # Not enough books to cluster

            # Step 2: Calculate spatial distances between books
            distance_matrix = np.zeros((len(book_nodes), len(book_nodes)))
            for i, node1 in enumerate(book_nodes):
                pos1 = node1.attributes.get("world_position")
                if not pos1:
                    continue

                for j, node2 in enumerate(book_nodes):
                    if i == j:
                        continue

                    pos2 = node2.attributes.get("world_position")
                    if not pos2:
                        continue

                    # Calculate Euclidean distance
                    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
                    distance_matrix[i, j] = distance

            # Step 3: Apply hierarchical clustering
            threshold = 0.5  # default threshold, 0.5 meters
            if self.config and 'semantic_forest' in self.config and 'clustering' in self.config['semantic_forest']:
                threshold = self.config['semantic_forest']['clustering'].get('distance_threshold', 0.5)
            clusters = []
            used_indices = set()

            for i in range(len(book_nodes)):
                if i in used_indices:
                    continue

                cluster = [i]
                used_indices.add(i)

                for j in range(len(book_nodes)):
                    if j in used_indices:
                        continue

                    # Check if distance is below threshold
                    if distance_matrix[i, j] < threshold:
                        cluster.append(j)
                        used_indices.add(j)

                min_cluster_size = 2  # default minimum size for a cluster
                if self.config and 'semantic_forest' in self.config and 'clustering' in self.config['semantic_forest']:
                    min_cluster_size = self.config['semantic_forest']['clustering'].get('min_cluster_size', 2)
                
                if len(cluster) >= min_cluster_size:
                    clusters.append([book_nodes[idx].node_id for idx in cluster])

            # Step 4: Create cluster nodes
            for cluster_idx, node_ids in enumerate(clusters):
                self._create_cluster_node(cluster_idx, node_ids)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error updating forest structure: {e}")  

    def _create_cluster_node(self, cluster_idx: int, node_ids: List[str]):
        """Create a cluster node with summary of books."""
        try:
            # Get nodes
            nodes = [self.semantic_forest.get_node(nid) for nid in node_ids]
            nodes = [n for n in nodes if n]  # Filter None

            if not nodes:
                return

            # Generate cluster summary
            summary = self._generate_cluster_summary(nodes)

            # Get average position
            positions = [n.attributes.get("world_position") for n in nodes 
                        if n.attributes.get("world_position")]

            avg_position = None
            if positions:
                avg_position = np.mean(np.array(positions), axis=0).tolist()

            # Create cluster node
            cluster_node = ForestNode(
                node_id=f"cluster_{cluster_idx}_{int(time.time())}",
                node_type="book_cluster",
                description=summary,
                attributes={
                    "member_nodes": node_ids,
                    "size": len(node_ids),
                    "world_position": avg_position,
                    "timestamp": time.time()
                }
            )

            # Add to forest
            self.semantic_forest.add_node(cluster_node)

        except Exception as e:
            print(f"Error creating cluster node: {e}")



    def save(self, filepath):
        """Save memory data directly to file"""
        try:
            # Create data structure directly from semantic forest nodes
            data = {
                "semantic_forest_nodes": {},
                "metadata": {
                    "timestamp": time.time(),
                    "unique_books": self.observation_stats.get('unique_books', 0)
                }
            }

            # Get nodes directly from semantic forest
            all_nodes = self.semantic_forest.get_all_nodes()

            # Process each node
            for node in all_nodes:
                node_data = {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "description": node.description,
                    "attributes": self._process_for_json(getattr(node, 'attributes', {})),
                    "children": getattr(node, 'children', []),
                    "parent": getattr(node, 'parent', None)
                }
                data["semantic_forest_nodes"][node.node_id] = node_data

            # Save directly to file with json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"Successfully saved semantic forest to {filepath}")

        except Exception as e:
            print(f"Error saving forest: {e}")
            import traceback
            traceback.print_exc()

            # Try simple backup
            try:
                backup_data = {"backup": True, "timestamp": time.time()}
                with open(filepath, 'w') as f:
                    json.dump(backup_data, f)

                print(f"Saved backup data to {filepath}")
            except Exception as e2:
                print(f"Even backup save failed: {e2}")

    def _process_for_json(self, obj):
        """Process any object to make it JSON serializable"""
        # Handle None
        if obj is None:
            return None

        # Handle Vector3
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Vector3':
            return [float(obj[0]), float(obj[1]), float(obj[2])]

        # Handle Quaternion 
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Quaternion':
            return {
                "scalar": float(obj.scalar),
                "vector": [float(obj.vector.x), float(obj.vector.y), float(obj.vector.z)]
            }

        # Handle numpy arrays
        if hasattr(obj, 'dtype') and hasattr(obj, 'tolist'):
            return obj.tolist()

        # Handle numpy scalars
        if hasattr(obj, 'dtype') and hasattr(obj, 'item'):
            return obj.item()

        # Handle lists recursively
        if isinstance(obj, list):
            return [self._process_for_json(item) for item in obj]

        # Handle dictionaries recursively  
        if isinstance(obj, dict):
            return {k: self._process_for_json(v) for k, v in obj.items()}

        # Return other types as is
        return obj
    
    def _generate_cluster_summary(self, nodes: List[ForestNode]) -> str:
        """Use LLM to generate summary of book cluster"""
        descriptions = [n.description for n in nodes]
        positions = [n.attributes.get("world_position") for n in nodes if n.attributes.get("world_position")]
        
        prompt = f"""Summarize these book observations into a single description:
        Books: {descriptions}
        Positions: {positions}
        
        Provide a concise summary describing the collection."""
        
        try:
            summary = self.ollama_llm.generate(prompt)
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Cluster of {len(nodes)} books"
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """Query semantic forest for information."""
        if not self.retriever:
            self._initialize_retriever()
            if not self.retriever:
                return [{"error": "Retriever not available"}]
        
        try:
            # Use retrievers to find relevant nodes
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
            
            return results
        except Exception as e:
            print(f"Error in query: {e}")
            import traceback
            traceback.print_exc()
            return [{"error": str(e)}]            

    def get_memory_stats(self) -> Dict:
        """Get current memory statistics including picked books"""
        all_nodes = self.semantic_forest.get_all_nodes()
        book_nodes = [n for n in all_nodes if n.node_type == 'book_instance']
        cluster_nodes = [n for n in all_nodes if n.node_type == 'book_cluster']
        
        # Count books by status
        picked_books = [n for n in book_nodes if n.attributes.get("status") == "picked"]
        available_books = [n for n in book_nodes if n.attributes.get("status") != "picked"]
        
        recent_window = 30  # seconds
        current_time = time.time()

        # Count recent discoveries more accurately
        recent_discoveries = []
        for node in book_nodes:
            if node.attributes.get("timestamp", 0) > current_time - recent_window:
                recent_discoveries.append(node.node_id)

        return {
            "total_unique_books": len(book_nodes),
            "picked_books": len(picked_books),
            "available_books": len(available_books),
            "total_clusters": len(cluster_nodes),
            "total_nodes": len(all_nodes),
            "total_observations": sum(node.attributes.get("observation_count", 1) for node in book_nodes),
            "recent_discoveries": len(recent_discoveries),
            "time_since_last_novel": current_time - self.last_novel_discovery,
            "search_coverage": self._estimate_coverage(),
            "mission_duration": current_time - self.mission_start_time
        }
    
    def _estimate_coverage(self) -> float:
        """Estimate search coverage based on spatial distribution"""
        try:
            nodes = self.semantic_forest.get_all_nodes()
            positions = []
            
            for node in nodes:
                if node.attributes.get("world_position"):
                    positions.append(node.attributes["world_position"])
                    
            if not positions:
                return 0.0
                
            # Simple coverage estimation based on spatial spread
            positions = np.array(positions)
            x_range = positions[:, 0].max() - positions[:, 0].min()
            z_range = positions[:, 2].max() - positions[:, 2].min()
            
            # Estimate based on expected room size
            expected_area = 100  # m²
            covered_area = x_range * z_range
            
            return min(1.0, covered_area / expected_area)
            
        except Exception as e:
            print(f"Error estimating coverage: {e}")
            return 0.0
    
    def _extract_spatial_context(self, position, source_id=None, book_description=None):
        """Extract spatial relationships to nearby books."""
        if not position:
            return None
            
        context = {
            "nearby_books": [],
            "alignments": [],
            "arrangements": {}
        }
        position_array = np.array(position)
        
        # Track processed books to avoid duplication
        processed_books = set()
        if source_id:
            processed_books.add(source_id)
        
        # Get all book instance nodes
        book_nodes = [node for node in self.semantic_forest.get_all_nodes() 
                     if node.node_type == "book_instance" and 
                     "world_position" in node.attributes and
                     node.attributes.get("world_position") and
                     node.node_id not in processed_books]
        
        # Process each book that has a position
        for node in book_nodes:
            other_position = node.attributes.get("world_position")
            if not other_position:
                continue
                
            # Skip if it's the same position
            other_array = np.array(other_position)
            if np.array_equal(position_array, other_array):
                continue
                
            # Calculate distance
            distance = np.linalg.norm(position_array - other_array)
            
            # Only include books within proximity threshold
            proximity_threshold = self.config.get('spatial', {}).get('proximity_threshold', 5.0)
            if distance <= proximity_threshold:
                try:
                    # First use advanced spatial relation tracker
                    if source_id:
                        # Infer detailed spatial relations
                        relations = self.spatial_tracker.infer_spatial_relations(
                            source_id=source_id,
                            source_pos=position,
                            target_id=node.node_id,
                            target_pos=other_position,
                            source_attrs={"description": book_description} if book_description else None,
                            target_attrs={"description": node.description}
                        )
                        
                        # Add relations to the tracker
                        for relation in relations:
                            self.spatial_tracker.add_relation(relation)
                        
                        # Convert to human-readable format for the most significant relation
                        if relations:
                            # Sort by confidence
                            top_relations = sorted(relations, key=lambda r: r.confidence, reverse=True)
                            primary_relation = self.spatial_tracker.format_relation_for_llm(top_relations[0])
                        else:
                            primary_relation = f"is {distance:.2f}m away from"
                    else:
                        # Fallback to basic relation if we don't have a source ID
                        primary_relation = f"is {distance:.2f}m away from"
                        
                    # Add relationship information
                    context["nearby_books"].append({
                        "book_id": node.node_id,
                        "description": node.description,
                        "relation": primary_relation,
                        "distance": float(distance)
                    })
                except Exception as e:
                    print(f"Error computing spatial relation: {e}")
                    
        # If we have a source ID, get any special arrangements this book is part of
        if source_id:
            try:
                # Get all existing arrangements
                arrangements = self.spatial_tracker.identify_book_arrangements()
                
                # Find arrangements involving this book
                involved_arrangements = {
                    arrangement_type: [arr for arr in arrs if source_id in arr]
                    for arrangement_type, arrs in arrangements.items()
                }
                
                # Add to context if found
                for arr_type, arr_list in involved_arrangements.items():
                    if arr_list:
                        # Prepare list of books in shared arrangements
                        shared_books = []
                        for arr in arr_list:
                            for other_id in arr:
                                if other_id != source_id and other_id not in shared_books:
                                    # Find details for this book
                                    other_node = self.semantic_forest.get_node(other_id)
                                    if other_node:
                                        shared_books.append({
                                            "book_id": other_id,
                                            "description": other_node.description,
                                            "arrangement": arr_type.rstrip('s')  # Remove plural
                                        })
                        
                        if shared_books:
                            context["arrangements"][arr_type] = shared_books
            except Exception as e:
                print(f"Error identifying book arrangements: {e}")
        
        return context

    def should_terminate_search(self) -> Tuple[bool, str]:
        """Determine if book search should be terminated."""
        current_time = time.time()
        mission_duration = current_time - self.mission_start_time
        
        # Early termination prevention - minimum search time requirement
        min_search_time = self.config.get('search', {}).get('min_duration', 60)
        if mission_duration < min_search_time:
            return False, f"Search still in initial phase ({mission_duration:.1f}/{min_search_time}s)"

        # Get comprehensive memory statistics
        stats = self.get_memory_stats()
        search_metrics = self._calculate_search_metrics()
        time_since_last_novel = current_time - self.last_novel_discovery
        
        # 0. Check if a book has been picked - higher priority termination
        picked_books = []
        for node in self.semantic_forest.get_all_nodes():
            if (node.node_type == "book_instance" and 
                node.attributes.get("status") == "picked"):
                picked_books.append(node)
                
        if picked_books:
            return True, f"Successfully picked a book: {picked_books[0].description[:30]}..."
        
        # 1. Termination based on lack of novelty (exploration efficiency)
        novelty_window = self._get_adaptive_novelty_window(stats, mission_duration)
        novelty_threshold = self._get_adaptive_novelty_threshold(stats, mission_duration)
        
        # Count recent unique discoveries
        if time_since_last_novel > novelty_window:
            recent_window_start = current_time - novelty_window
            
            # Improved counting of recent discoveries using observation timestamps
            recent_count = 0
            for node in self.semantic_forest.get_all_nodes():
                if (node.node_type == "book_instance" and 
                    node.attributes.get("timestamp", 0) > recent_window_start):
                    recent_count += 1
            
            # Check if novelty rate is below threshold
            if recent_count < novelty_threshold:
                return True, f"Low novelty rate: Only {recent_count} new books in last {novelty_window:.1f}s (threshold: {novelty_threshold})"
        
        # 2. Termination based on spatial coverage (exploration breadth)
        coverage_threshold = self.config.get('search', {}).get('coverage_threshold', 0.9)
        if stats["search_coverage"] > coverage_threshold:
            return True, f"High spatial coverage: {stats['search_coverage']:.1%} (threshold: {coverage_threshold:.1%})"
        
        # 3. Termination based on diminishing returns (efficiency)
        if search_metrics['efficiency'] < 0.1 and mission_duration > min_search_time * 2:
            return True, f"Low search efficiency: {search_metrics['efficiency']:.2f} books/minute"
        
        # 4. Termination based on finding sufficient books (goal satisfaction)
        min_books_threshold = self.config.get('search', {}).get('min_books', 3)
        if stats["total_unique_books"] >= min_books_threshold:
            # If we have enough books and haven't found new ones recently
            if time_since_last_novel > novelty_window:
                return True, f"Found sufficient books ({stats['total_unique_books']}) with no recent discoveries"
        
        # 5. Termination based on observation saturation (repeated observations)
        if search_metrics['observation_rate'] > 0 and search_metrics['uniqueness_rate'] < 0.1:
            # Many observations but very few new unique books
            if mission_duration > min_search_time * 1.5:
                return True, f"Low uniqueness rate: {search_metrics['uniqueness_rate']:.2f} with {stats['total_observations']} observations"
        
        # By default, continue searching
        return False, self._generate_continuation_reason(stats, search_metrics)
    
    def _calculate_search_metrics(self) -> Dict:
        """Calculate advanced search metrics for termination decisions."""
        current_time = time.time()
        mission_duration_minutes = (current_time - self.mission_start_time) / 60.0
        
        total_observations = self.observation_stats['total_observations']
        unique_books = self.observation_stats['unique_books']
        
        # Calculate rates (per minute)
        observation_rate = total_observations / mission_duration_minutes if mission_duration_minutes > 0 else 0
        discovery_rate = unique_books / mission_duration_minutes if mission_duration_minutes > 0 else 0
        
        # Calculate efficiency metrics
        uniqueness_rate = unique_books / total_observations if total_observations > 0 else 0
        efficiency = discovery_rate  # Books found per minute
        
        # Calculate rate of change (acceleration/deceleration)
        recent_window = 60  # Look at last minute
        recent_window_start = current_time - recent_window
        
        recent_obs_count = sum(1 for obs in self.observation_stats['observation_history'] 
                              if obs['timestamp'] > recent_window_start)
        
        recent_obs_rate = recent_obs_count / (recent_window / 60) if recent_window > 0 else 0
        rate_change = recent_obs_rate - observation_rate if observation_rate > 0 else 0
        
        return {
            'total_observations': total_observations,
            'unique_books': unique_books,
            'observation_rate': observation_rate,  # Observations per minute
            'discovery_rate': discovery_rate,      # Unique books per minute
            'uniqueness_rate': uniqueness_rate,    # Proportion of observations that yield unique books
            'efficiency': efficiency,              # Overall search efficiency
            'recent_obs_rate': recent_obs_rate,    # Recent observation rate
            'rate_change': rate_change             # Change in observation rate (acceleration/deceleration)
        }
    
    def _get_adaptive_novelty_window(self, stats: Dict, mission_duration: float) -> float:
        """
        Get adaptive novelty window based on search progress.
        Starts with a short window and increases as search progresses.
        """
        base_window = self.config.get('agent', {}).get('novelty_window', 30)
        
        # Scale window based on mission duration
        if mission_duration < 120:  # First 2 minutes
            return base_window
        elif mission_duration < 300:  # 2-5 minutes
            return base_window * 1.5
        else:  # 5+ minutes
            return base_window * 2.0
    
    def _get_adaptive_novelty_threshold(self, stats: Dict, mission_duration: float) -> int:
        """
        Get adaptive novelty threshold based on search progress.
        More forgiving early in the search, stricter later.
        """
        base_threshold = self.config.get('agent', {}).get('novelty_count_threshold', 3)
        
        # Early in search: require fewer discoveries
        if mission_duration < 120:  # First 2 minutes
            return max(1, base_threshold - 1)
        elif mission_duration < 300:  # 2-5 minutes
            return base_threshold
        else:  # 5+ minutes
            return base_threshold + 1  # Require more discoveries to continue
    
    def _generate_continuation_reason(self, stats: Dict, metrics: Dict) -> str:
        """Generate an informative reason for continuing the search."""
        if stats["total_unique_books"] == 0:
            return "No books found yet, continuing search"
        
        # If we have picked books, use that in the reason
        if "picked_books" in stats and stats["picked_books"] > 0:
            if stats["search_coverage"] < 0.5:
                return f"Already picked {stats['picked_books']} book(s), but continuing to explore more areas"
            return f"Already picked {stats['picked_books']} book(s), searching for additional books"
        
        # If we have available books that could be picked
        if "available_books" in stats and stats["available_books"] > 0:
            best_books = self._get_best_books_to_pick(max_count=2)
            if best_books:
                book_desc = best_books[0].description[:30] + "..." if len(best_books[0].description) > 30 else best_books[0].description
                return f"Found {stats['available_books']} book(s) including '{book_desc}', continue search to find more or pick"
        
        min_books = self.config.get('search', {}).get('min_books', 3)
        if stats["total_unique_books"] < min_books:
            return f"Searching for more books ({stats['total_unique_books']}/{min_books})"
        
        if metrics['efficiency'] > 0.5:
            return f"Good search efficiency ({metrics['efficiency']:.1f} books/min), continuing"
        
        if stats["search_coverage"] < 0.5:
            return f"Low area coverage ({stats['search_coverage']:.1%}), continuing exploration"
        
        return "Continue searching for more books"
    
    def _get_best_books_to_pick(self, max_count=3):
        """Get a list of the best books to pick based on confidence and accessibility."""
        book_nodes = [n for n in self.semantic_forest.get_all_nodes() 
                      if n.node_type == 'book_instance' and n.attributes.get('status') != 'picked']
        
        # Skip if no books available
        if not book_nodes:
            return []
            
        # Sort by confidence (highest first)
        book_nodes.sort(key=lambda n: n.attributes.get('confidence', 0), reverse=True)
        
        # Return top N books
        return book_nodes[:max_count]
    
    def record_pick_action(self, position=None):
        """Record that a book was picked from the given position."""
        if position is None:
            return False

        # Find books near this position
        nodes = self.semantic_forest.get_all_nodes()
        picked_book = None
        min_distance = float('inf')

        # Enhanced search for the closest book to the pick position
        for node in nodes:
            if node.node_type != "book_instance":
                continue

            # Skip if already picked
            if node.attributes.get('status') == 'picked':
                continue

            # Skip if no position information
            if not node.attributes or "world_position" not in node.attributes:
                continue

            # Calculate distance
            book_pos = node.attributes['world_position']

            # Convert Vector3 to list if needed
            if hasattr(book_pos, '__class__') and book_pos.__class__.__name__ == 'Vector3':
                book_pos = [book_pos[0], book_pos[1], book_pos[2]]

            try:
                distance = np.linalg.norm(np.array(position) - np.array(book_pos))

                # Log all potential matches within 1m for debugging
                if distance < 1.0:
                    print(f"Potential book match at {distance:.3f}m: {node.description[:30]}...")

                # If this is the closest book so far
                if distance < min_distance and distance < 0.5:  # Within 50cm
                    min_distance = distance
                    picked_book = node
            except Exception as e:
                print(f"Error calculating distance: {e}")
                continue
            
        # Update the book status if found
        if picked_book:
            # Mark the book as picked
            picked_book.attributes['status'] = 'picked'
            picked_book.attributes['pick_time'] = time.time()
            picked_book.attributes['pick_position'] = position

            # Record the pick action in observation history
            if 'observation_history' not in picked_book.attributes:
                picked_book.attributes['observation_history'] = []

            picked_book.attributes['observation_history'].append({
                'id': f"pick_{int(time.time()*1000)}",
                'timestamp': time.time(),
                'action': 'pick',
                'position': position,
                'distance': float(min_distance)
            })

            # Record pick in position history too
            if 'position_history' not in picked_book.attributes:
                picked_book.attributes['position_history'] = []

            picked_book.attributes['position_history'].append({
                'position': position,
                'timestamp': time.time(),
                'action': 'pick',
                'confidence': 1.0,  # High confidence since this is a direct action
                'distance_from_previous': float(min_distance)
            })

            # Update the node in the forest
            self.semantic_forest.update_node(picked_book)
            print(f"Recorded pick action for book: {picked_book.description[:30]}... at distance {min_distance:.3f}m")
            return True

        print("No matching book found to record pick action")
        return False
        
    def record_place_action(self, pickup_position=None, placement_position=None):
        """
        Record that a previously picked book was placed at a new position.
        
        Args:
            pickup_position: [x, y, z] position where the book was picked from
            placement_position: [x, y, z] position where the book was placed
            
        Returns:
            bool: True if a book was found and updated, False otherwise
        """
        if pickup_position is None or placement_position is None:
            print("Cannot record placement without both pickup and placement positions")
            return False
        
        # Find the book that was picked from this position
        nodes = self.semantic_forest.get_all_nodes()
        placed_book = None
        
        # First try to find by status and pick position
        for node in nodes:
            if (node.node_type == "book_instance" and 
                node.attributes.get('status') == 'picked'):
                
                # If we have the pick position recorded
                if node.attributes.get('pick_position'):
                    pick_pos = node.attributes['pick_position']
                    try:
                        distance = np.linalg.norm(np.array(pickup_position) - np.array(pick_pos))
                        
                        # If this is a close match to our pickup position
                        if distance < 0.5:  # Within 50cm
                            placed_book = node
                            print(f"Found picked book matching pickup position (distance: {distance:.3f}m)")
                            break
                    except Exception as e:
                        print(f"Error comparing pick positions: {e}")
        
        # If we didn't find by pick position, try most recently picked
        if not placed_book:
            # Find all picked books
            picked_books = [n for n in nodes if n.node_type == "book_instance" and 
                           n.attributes.get('status') == 'picked']
            
            if picked_books:
                # Sort by pick time (most recent first)
                picked_books.sort(key=lambda n: n.attributes.get('pick_time', 0), reverse=True)
                placed_book = picked_books[0]
                print(f"Using most recently picked book as fallback")
        
        # Update the book status if found
        if placed_book:
            # Mark the book as placed
            placed_book.attributes['status'] = 'placed'
            placed_book.attributes['place_time'] = time.time()
            placed_book.attributes['place_position'] = placement_position
            
            # Record the place action in observation history
            if 'observation_history' not in placed_book.attributes:
                placed_book.attributes['observation_history'] = []
                
            placed_book.attributes['observation_history'].append({
                'id': f"place_{int(time.time()*1000)}",
                'timestamp': time.time(),
                'action': 'place',
                'position': placement_position,
                'previous_position': pickup_position
            })
            
            # Record placement in position history too
            if 'position_history' not in placed_book.attributes:
                placed_book.attributes['position_history'] = []
                
            # Calculate distance from pickup position
            distance_moved = 0.0
            try:
                distance_moved = np.linalg.norm(np.array(placement_position) - np.array(pickup_position))
            except Exception:
                pass
                
            placed_book.attributes['position_history'].append({
                'position': placement_position,
                'timestamp': time.time(),
                'action': 'place',
                'confidence': 1.0,  # High confidence since this is a direct action
                'distance_from_previous': float(distance_moved)
            })
            
            # Update the world position to the placement position
            placed_book.attributes['world_position'] = placement_position
            
            # Update the node in the forest
            self.semantic_forest.update_node(placed_book)
            print(f"Recorded place action for book: {placed_book.description[:30]}...")
            print(f"  Book moved {distance_moved:.3f}m from pickup to placement")
            return True
            
        print("No matching book found to record place action")
        return False
    
    def save(self, filepath):
        """Save memory data directly to file."""
        try:
            # Create data structure directly from semantic forest nodes
            data = {
                "semantic_forest_nodes": {},
                "metadata": {
                    "timestamp": time.time(),
                    "unique_books": self.observation_stats.get('unique_books', 0)
                }
            }

            # Get nodes directly from semantic forest
            all_nodes = self.semantic_forest.get_all_nodes()

            # Process each node
            for node in all_nodes:
                node_data = {
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "description": node.description,
                    "attributes": self._process_for_json(getattr(node, 'attributes', {})),
                    "children": getattr(node, 'children', []),
                    "parent": getattr(node, 'parent', None)
                }
                data["semantic_forest_nodes"][node.node_id] = node_data

            # Save directly to file with json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"Successfully saved semantic forest to {filepath}")

        except Exception as e:
            print(f"Error saving forest: {e}")

            # Try simple backup
            try:
                backup_data = {"backup": True, "timestamp": time.time()}
                with open(filepath, 'w') as f:
                    json.dump(backup_data, f)

                print(f"Saved backup data to {filepath}")
            except Exception as e2:
                print(f"Even backup save failed: {e2}")


    def _convert_for_serialization(self, obj):
        """Recursively convert values for JSON serialization."""
        # Handle Vector3 objects
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Vector3':
            return [obj[0], obj[1], obj[2]]

        # Handle lists that might contain Vector3 objects
        elif isinstance(obj, list):
            return [self._convert_for_serialization(item) for item in obj]

        # Handle dictionaries that might contain Vector3 objects
        elif isinstance(obj, dict):
            return {k: self._convert_for_serialization(v) for k, v in obj.items()}

        # Return other types as is
        else:
            return obj
    
    def load(self, filepath: str):
        """Load forest from disk with enhanced error handling"""
        try:
            # Check if file exists first
            if not os.path.exists(filepath):
                print(f"Warning: Semantic forest file {filepath} not found. Starting with empty forest.")
                # Initialize with empty forest rather than failing
                return
            
            # Try to load the forest with robust error handling
            try:
                self.semantic_forest.load(filepath)
                # Verify forest loaded correctly
                try:
                    nodes = self.semantic_forest.get_all_nodes()
                    nodes_count = len(nodes) if nodes else 0
                    print(f"Semantic forest loaded from {filepath} with {nodes_count} nodes")
                except Exception as nodes_error:
                    # If nodes are missing, reset the nodes dict to empty
                    print(f"Warning: Error accessing forest nodes: {nodes_error}")
                    print("Initializing empty nodes dictionary")
                    if not hasattr(self.semantic_forest, 'nodes') or self.semantic_forest.nodes is None:
                        self.semantic_forest.nodes = {}
                    print(f"Semantic forest loaded from {filepath} with 0 nodes")
            except Exception as forest_error:
                # If forest load fails, create a fresh forest
                print(f"Warning: Could not load semantic forest: {forest_error}")
                print("Initializing fresh semantic forest")
                self.semantic_forest = SemanticForest()
        except Exception as e:
            # Catch-all for any other issues
            print(f"Error during semantic forest loading process: {e}")
            import traceback
            traceback.print_exc()
            # Create empty forest to allow the agent to function
            self.semantic_forest = SemanticForest()
    
    def update_robot_position(self, position):
        """Update the robot's position in the semantic forest."""
        if position is None:
            return
            
        # First, check if we already have a robot node
        robot_node = None
        for node in self.semantic_forest.get_all_nodes():
            if node.node_type == "robot":
                robot_node = node
                break
                
        current_time = time.time()
        
        if robot_node:
            # Update existing robot node
            robot_node.attributes["world_position"] = position
            robot_node.attributes["last_updated"] = current_time
            robot_node.attributes["position_history"].append({
                "position": position,
                "timestamp": current_time
            })
            
            # Keep position history limited to prevent memory growth
            max_history = 100
            if len(robot_node.attributes["position_history"]) > max_history:
                robot_node.attributes["position_history"] = robot_node.attributes["position_history"][-max_history:]
                
            # Update the node in the forest
            self.semantic_forest.update_node(robot_node)
        else:
            # Create a new robot node
            robot_node = ForestNode(
                node_id="robot_agent",
                node_type="robot",
                description="Robot Agent",
                attributes={
                    "world_position": position,
                    "creation_time": current_time,
                    "last_updated": current_time,
                    "position_history": [{
                        "position": position,
                        "timestamp": current_time
                    }]
                }
            )
            
            # Add the node to the forest
            self.semantic_forest.add_node(robot_node)
            
        # Update visualization tool reference (if it exists)
        try:
            from tools import visualize_semantic_forest
            if hasattr(visualize_semantic_forest, 'robot_position'):
                visualize_semantic_forest.robot_position = position
        except Exception as e:
            # It's okay if this fails, just means the visualization tool isn't available
            pass

class VectorJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Vector3, numpy arrays, and other special types."""
    def default(self, obj):
        # Handle Vector3 objects from magnum
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Vector3':
            return [float(obj[0]), float(obj[1]), float(obj[2])]
        
        # Handle magnum.Quaternion
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Quaternion':
            return {
                "scalar": float(obj.scalar),
                "vector": [float(obj.vector.x), float(obj.vector.y), float(obj.vector.z)]
            }
            
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # Handle numpy numeric types
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                           np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
            
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
            
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Let the base class handle it (or raise TypeError)
        return super().default(obj)