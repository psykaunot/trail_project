"""
Enhanced spatial relationship extraction and management for book memory.
Provides more detailed and accurate spatial relationship tracking between books.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import math
from dataclasses import dataclass
from enum import Enum
import time

class RelationType(Enum):
    """Types of spatial relationships between objects."""
    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left of"
    RIGHT_OF = "right of"
    IN_FRONT_OF = "in front of"
    BEHIND = "behind"
    NEAR = "near"
    FAR = "far"
    INSIDE = "inside"
    CONTAINING = "containing"
    ON_TOP_OF = "on top of"
    UNDERNEATH = "underneath"
    ALIGNED_WITH = "aligned with"
    STACKED_WITH = "stacked with"
    ADJACENT_TO = "adjacent to"
    GROUP = "grouped with"

@dataclass
class SpatialRelation:
    """Detailed spatial relationship between two objects."""
    source_id: str
    target_id: str
    relation_type: RelationType
    distance: float
    angle: float  # In radians
    confidence: float
    timestamp: float
    
    @property
    def is_reciprocal(self) -> bool:
        """Check if this relationship has a natural reciprocal relationship."""
        reciprocal_pairs = {
            RelationType.ABOVE: RelationType.BELOW,
            RelationType.LEFT_OF: RelationType.RIGHT_OF,
            RelationType.IN_FRONT_OF: RelationType.BEHIND,
            RelationType.CONTAINING: RelationType.INSIDE,
            RelationType.ON_TOP_OF: RelationType.UNDERNEATH
        }
        return self.relation_type in reciprocal_pairs
    
    def get_reciprocal(self) -> 'SpatialRelation':
        """Get the reciprocal relationship (if applicable)."""
        reciprocal_pairs = {
            RelationType.ABOVE: RelationType.BELOW,
            RelationType.BELOW: RelationType.ABOVE,
            RelationType.LEFT_OF: RelationType.RIGHT_OF,
            RelationType.RIGHT_OF: RelationType.LEFT_OF,
            RelationType.IN_FRONT_OF: RelationType.BEHIND,
            RelationType.BEHIND: RelationType.IN_FRONT_OF,
            RelationType.CONTAINING: RelationType.INSIDE,
            RelationType.INSIDE: RelationType.CONTAINING,
            RelationType.ON_TOP_OF: RelationType.UNDERNEATH,
            RelationType.UNDERNEATH: RelationType.ON_TOP_OF
        }
        
        if self.relation_type not in reciprocal_pairs:
            return self  # No reciprocal for this relationship
        
        return SpatialRelation(
            source_id=self.target_id,
            target_id=self.source_id,
            relation_type=reciprocal_pairs[self.relation_type],
            distance=self.distance,
            angle=(self.angle + math.pi) % (2 * math.pi),  # Opposite direction
            confidence=self.confidence,
            timestamp=self.timestamp
        )

class SpatialRelationTracker:
    """Manages and analyzes spatial relationships between books."""
    
    def __init__(self, config=None):
        """
        Initialize the spatial relationship tracker.
        
        Args:
            config: Configuration dictionary with tracking parameters
        """
        self.config = config or {}
        self.relations = []
        self.groupings = {}  # Groups of books that are related
        
        # Configuration parameters
        self.near_threshold = self.config.get('spatial', {}).get('near_threshold', 0.5)  # meters
        self.stacking_threshold = self.config.get('spatial', {}).get('stacking_threshold', 0.1)  # meters
        self.alignment_angle_threshold = self.config.get('spatial', {}).get('alignment_threshold', 0.15)  # radians
        self.relation_memory_limit = self.config.get('spatial', {}).get('memory_limit', 200)
        
        # Cache for efficiency
        self.relation_cache = {}
    
    def add_relation(self, relation: SpatialRelation) -> None:
        """Add a new spatial relationship."""
        self.relations.append(relation)
        
        # Add reciprocal if applicable
        if relation.is_reciprocal:
            self.relations.append(relation.get_reciprocal())
            
        # Limit memory usage by removing oldest relations if needed
        if len(self.relations) > self.relation_memory_limit:
            self.relations = self.relations[-self.relation_memory_limit:]
            
        # Clear cache as it's now stale
        self.relation_cache = {}
    
    def infer_spatial_relations(self, source_id: str, source_pos: List[float], 
                               target_id: str, target_pos: List[float],
                               source_attrs: Optional[Dict] = None, 
                               target_attrs: Optional[Dict] = None) -> List[SpatialRelation]:
        """
        Infer spatial relationships between two objects based on their positions.
        
        Args:
            source_id: ID of the source object
            source_pos: 3D position of the source [x, y, z]
            target_id: ID of the target object
            target_pos: 3D position of the target [x, y, z]
            source_attrs: Additional attributes of the source object
            target_attrs: Additional attributes of the target object
            
        Returns:
            List of SpatialRelation objects
        """
        if not source_pos or not target_pos:
            return []
            
        # These will be the relationships we infer
        relations = []
        
        # Convert positions to numpy arrays for easier calculation
        source = np.array(source_pos)
        target = np.array(target_pos)
        
        # Calculate distance
        distance = np.linalg.norm(source - target)
        
        # Calculate vector from source to target
        direction = target - source
        
        # Calculate XZ (horizontal) distance for horizontal relations
        if len(source) >= 3 and len(target) >= 3:
            horizontal_distance = np.linalg.norm(np.array([direction[0], direction[2]]))
        else:
            horizontal_distance = np.linalg.norm(direction[:2])
        
        # Calculate vertical offset for vertical relations
        vertical_offset = direction[1] if len(direction) > 1 else 0
        
        # Calculate angle in the horizontal plane (for left/right/front/behind)
        # Assuming +X is "forward" and +Z is "right"
        if len(direction) >= 3:
            angle = math.atan2(direction[2], direction[0])
        else:
            angle = math.atan2(direction[1], direction[0])
            
        # Timestamp for all relations
        timestamp = time.time()
        
        # Distance-based relations
        if distance < self.near_threshold:
            relations.append(SpatialRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=RelationType.NEAR,
                distance=distance,
                angle=angle,
                confidence=1.0 - (distance / self.near_threshold),
                timestamp=timestamp
            ))
        else:
            relations.append(SpatialRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=RelationType.FAR,
                distance=distance,
                angle=angle,
                confidence=min(1.0, (distance - self.near_threshold) / 2.0),
                timestamp=timestamp
            ))
        
        # Vertical relations
        if vertical_offset > self.stacking_threshold:
            # Target is above source
            relations.append(SpatialRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=RelationType.ABOVE,
                distance=distance,
                angle=angle,
                confidence=min(1.0, vertical_offset / (2 * self.stacking_threshold)),
                timestamp=timestamp
            ))
            
            # Check if close enough horizontally to be considered stacking
            if horizontal_distance < self.stacking_threshold:
                relations.append(SpatialRelation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=RelationType.ON_TOP_OF,
                    distance=distance,
                    angle=angle,
                    confidence=1.0 - (horizontal_distance / self.stacking_threshold),
                    timestamp=timestamp
                ))
        elif vertical_offset < -self.stacking_threshold:
            # Target is below source
            relations.append(SpatialRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=RelationType.BELOW,
                distance=distance,
                angle=angle,
                confidence=min(1.0, abs(vertical_offset) / (2 * self.stacking_threshold)),
                timestamp=timestamp
            ))
            
            # Check if close enough horizontally to be considered stacking
            if horizontal_distance < self.stacking_threshold:
                relations.append(SpatialRelation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=RelationType.UNDERNEATH,
                    distance=distance,
                    angle=angle,
                    confidence=1.0 - (horizontal_distance / self.stacking_threshold),
                    timestamp=timestamp
                ))
        
        # Horizontal relations based on angle
        # Front/back relations (assuming +X axis is forward)
        if abs(angle) < math.pi/4 or abs(angle) > 3*math.pi/4:
            if abs(angle) < math.pi/4:  # Target is in front
                relations.append(SpatialRelation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=RelationType.IN_FRONT_OF,
                    distance=distance,
                    angle=angle,
                    confidence=1.0 - (abs(angle) / (math.pi/4)),
                    timestamp=timestamp
                ))
            else:  # Target is behind
                relations.append(SpatialRelation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=RelationType.BEHIND,
                    distance=distance,
                    angle=angle,
                    confidence=1.0 - ((math.pi - abs(angle)) / (math.pi/4)),
                    timestamp=timestamp
                ))
        
        # Left/right relations
        if math.pi/4 < angle < 3*math.pi/4:  # Target is to the right
            relations.append(SpatialRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=RelationType.RIGHT_OF,
                distance=distance,
                angle=angle,
                confidence=1.0 - (abs(angle - math.pi/2) / (math.pi/4)),
                timestamp=timestamp
            ))
        elif -3*math.pi/4 < angle < -math.pi/4:  # Target is to the left
            relations.append(SpatialRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=RelationType.LEFT_OF,
                distance=distance,
                angle=angle,
                confidence=1.0 - (abs(angle + math.pi/2) / (math.pi/4)),
                timestamp=timestamp
            ))
        
        # Alignment relation (books in a row)
        # Check if books are at similar height and lined up horizontally
        if abs(vertical_offset) < self.stacking_threshold:
            relations.append(SpatialRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=RelationType.ALIGNED_WITH,
                distance=distance,
                angle=angle,
                confidence=1.0 - (abs(vertical_offset) / self.stacking_threshold),
                timestamp=timestamp
            ))
        
        return relations
    
    def get_relations_for_object(self, object_id: str) -> List[SpatialRelation]:
        """Get all spatial relationships involving the specified object."""
        # Use cache if available
        if object_id in self.relation_cache:
            return self.relation_cache[object_id]
            
        # Find all relations where object is source or target
        object_relations = [r for r in self.relations 
                          if r.source_id == object_id or r.target_id == object_id]
        
        # Sort by timestamp, most recent first
        object_relations.sort(key=lambda r: r.timestamp, reverse=True)
        
        # Store in cache
        self.relation_cache[object_id] = object_relations
        
        return object_relations
    
    def get_objects_in_relation(self, object_id: str, relation_type: RelationType) -> List[str]:
        """Get all objects that have the specified relationship with the given object."""
        # Get relations where object is the source and relation matches
        relations = [r for r in self.relations 
                   if r.source_id == object_id and r.relation_type == relation_type]
        
        # Sort by confidence, highest first
        relations.sort(key=lambda r: r.confidence, reverse=True)
        
        # Return target IDs
        return [r.target_id for r in relations]
    
    def format_relation_for_llm(self, relation: SpatialRelation) -> str:
        """Format a spatial relationship for LLM consumption."""
        # Get verbalized distance
        distance_str = self._verbalize_distance(relation.distance)
        
        # Format the relationship string
        rel_str = f"is {relation.relation_type.value} ({distance_str})"
        
        # Add confidence if it's below 0.8
        if relation.confidence < 0.8:
            confidence_str = "possibly " if relation.confidence > 0.5 else "might be "
            rel_str = confidence_str + rel_str
            
        return rel_str
    
    def _verbalize_distance(self, distance: float) -> str:
        """Convert a distance value to a human-readable description."""
        if distance < 0.1:
            return "touching"
        elif distance < 0.3:
            return "very close"
        elif distance < 0.7:
            return "close"
        elif distance < 1.5:
            return "nearby"
        elif distance < 3.0:
            return "not far"
        else:
            return f"{distance:.1f}m away"
    
    def identify_book_arrangements(self) -> Dict[str, List[str]]:
        """Identify meaningful arrangements of books (shelves, stacks, etc.)."""
        arrangements = {
            "stacks": [],  # Books stacked vertically
            "rows": [],    # Books in horizontal rows (shelf-like)
            "groups": []   # Books clustered together
        }
        
        # Process all relations to find common patterns
        all_object_ids = set()
        for rel in self.relations:
            all_object_ids.add(rel.source_id)
            all_object_ids.add(rel.target_id)
        
        # Find stacked books
        stacks = []
        for obj_id in all_object_ids:
            # Get books that are on top of this one
            above_relations = [r for r in self.relations 
                             if r.source_id == obj_id and r.relation_type == RelationType.ON_TOP_OF]
            
            # Only consider high confidence stacking
            above_relations = [r for r in above_relations if r.confidence > 0.7]
            
            if above_relations:
                # This object has books stacked on it
                stack = [obj_id]
                stack.extend([r.target_id for r in above_relations])
                stacks.append(stack)
        
        # Find rows of books (aligned horizontally, e.g., on a shelf)
        rows = []
        aligned_pairs = {}
        
        for obj_id in all_object_ids:
            # Get books aligned with this one
            aligned_relations = [r for r in self.relations 
                               if r.source_id == obj_id and r.relation_type == RelationType.ALIGNED_WITH]
            
            # Only consider high confidence alignment
            aligned_relations = [r for r in aligned_relations if r.confidence > 0.8]
            
            for rel in aligned_relations:
                # Create a key for this alignment pair
                pair_key = tuple(sorted([obj_id, rel.target_id]))
                if pair_key not in aligned_pairs:
                    aligned_pairs[pair_key] = rel.confidence
        
        # Group aligned pairs into rows
        if aligned_pairs:
            # Sort pairs by confidence
            sorted_pairs = sorted(aligned_pairs.items(), key=lambda x: x[1], reverse=True)
            
            # Simple greedy algorithm to form rows
            remaining_pairs = dict(sorted_pairs)
            while remaining_pairs:
                # Start a new row with the highest confidence pair
                current_pair, _ = next(iter(remaining_pairs.items()))
                del remaining_pairs[current_pair]
                
                current_row = list(current_pair)
                
                # Keep adding connected pairs
                added = True
                while added:
                    added = False
                    pairs_to_remove = []
                    
                    for pair, _ in remaining_pairs.items():
                        # Check if this pair connects to our current row
                        if pair[0] in current_row or pair[1] in current_row:
                            # Add the new book to the row
                            new_book = pair[1] if pair[0] in current_row else pair[0]
                            if new_book not in current_row:
                                current_row.append(new_book)
                                added = True
                                pairs_to_remove.append(pair)
                    
                    # Remove processed pairs
                    for pair in pairs_to_remove:
                        del remaining_pairs[pair]
                
                # Add the completed row
                if len(current_row) >= 3:  # Only consider rows with at least 3 books
                    rows.append(current_row)
        
        # Find groups (clusters of nearby books)
        groups = []
        for obj_id in all_object_ids:
            # Get books near this one
            near_relations = [r for r in self.relations 
                            if r.source_id == obj_id and r.relation_type == RelationType.NEAR]
            
            # Only consider close proximity
            near_relations = [r for r in near_relations if r.distance < self.near_threshold]
            
            if len(near_relations) >= 2:  # At least 3 books in a group (source + 2 targets)
                group = [obj_id]
                group.extend([r.target_id for r in near_relations])
                groups.append(group)
        
        # Store the arrangements
        arrangements["stacks"] = stacks
        arrangements["rows"] = rows
        arrangements["groups"] = groups
        
        return arrangements