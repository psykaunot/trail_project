#!/usr/bin/env python3
"""
Book Experience Memory System
Implements a robust memory for tracking unique book instances in 3D space
"""

import numpy as np
import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import threading

@dataclass
class BookPhysicalProperties:
    """Physical properties of a book instance"""
    bbox: List[float]  # Bounding box in image coordinates
    world_position: Optional[np.ndarray] = None  # 3D position in world coordinates
    size_estimate: Optional[List[float]] = None  # Estimated dimensions [width, height, depth]
    color_dominant: Optional[str] = None  # Dominant color
    orientation: Optional[Dict[str, float]] = None  # Orientation info
    
    def to_dict(self):
        return {
            "bbox": self.bbox,
            "world_position": self.world_position.tolist() if self.world_position is not None else None,
            "size_estimate": self.size_estimate,
            "color_dominant": self.color_dominant,
            "orientation": self.orientation
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            bbox=data["bbox"],
            world_position=np.array(data["world_position"]) if data["world_position"] else None,
            size_estimate=data["size_estimate"],
            color_dominant=data["color_dominant"],
            orientation=data["orientation"]
        )

@dataclass
class BookObservation:
    """Single observation of a book"""
    timestamp: float
    camera_position: Dict[str, float]  # Pan/tilt at observation time
    confidence: float = 1.0
    perception_data: Dict = field(default_factory=dict)  # Raw perception output
    
    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "camera_position": self.camera_position,
            "confidence": self.confidence,
            "perception_data": self.perception_data
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            timestamp=data["timestamp"],
            camera_position=data["camera_position"],
            confidence=data["confidence"],
            perception_data=data["perception_data"]
        )

@dataclass
class BookExperience:
    """Complete experience of a unique book instance"""
    id: str  # Unique identifier for this book instance
    first_seen: float  # Timestamp of first observation
    last_seen: float  # Timestamp of last observation
    physical_properties: BookPhysicalProperties
    observations: List[BookObservation] = field(default_factory=list)
    vlm_captions: List[str] = field(default_factory=list)  # VLM descriptions
    spatial_context: Optional[Dict] = None  # Spatial relationship info
    
    def to_dict(self):
        return {
            "id": self.id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "physical_properties": self.physical_properties.to_dict(),
            "observations": [obs.to_dict() for obs in self.observations],
            "vlm_captions": self.vlm_captions,
            "spatial_context": self.spatial_context
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            first_seen=data["first_seen"],
            last_seen=data["last_seen"],
            physical_properties=BookPhysicalProperties.from_dict(data["physical_properties"]),
            observations=[BookObservation.from_dict(obs) for obs in data["observations"]],
            vlm_captions=data["vlm_captions"],
            spatial_context=data["spatial_context"]
        )


class BookExperienceMemory:
    """
    Memory system for storing and retrieving unique book experiences.
    Forms the foundation for the future Semantic Forest.
    """
    
    def __init__(self, save_path="book_experience_memory.json"):
        self.memories: Dict[str, BookExperience] = {}
        self.save_path = save_path
        self.lock = threading.Lock()
        
        # Matching thresholds
        self.position_threshold = 0.5  # meters
        self.bbox_iou_threshold = 0.7  # IoU for bounding box matching
        self.size_similarity_threshold = 0.8
        
        # Search termination parameters
        self.novelty_window = 30  # seconds
        self.novelty_count_threshold = 3  # min new books in window
        self.coverage_threshold = 0.9  # percentage of search space covered
        
        # Load existing memory if available
        self.load_memory()
        
        # Search state tracking
        self.last_novel_discovery = time.time()
        self.search_coverage_map = {}  # Track which areas have been searched
        self.recent_discoveries = []  # Track recent unique discoveries
    
    def save_memory(self):
        """Save memory to disk"""
        with self.lock:
            data = {
                "memories": {id: exp.to_dict() for id, exp in self.memories.items()},
                "metadata": {
                    "last_updated": time.time(),
                    "total_books": len(self.memories)
                }
            }
            
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def load_memory(self):
        """Load memory from disk"""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                
                with self.lock:
                    self.memories = {
                        id: BookExperience.from_dict(exp_data)
                        for id, exp_data in data.get("memories", {}).items()
                    }
                
                print(f"Loaded {len(self.memories)} book experiences from memory")
            except Exception as e:
                print(f"Error loading memory: {e}")
    
    def add_observation(self, book_data: Dict, camera_state: Dict) -> Tuple[str, bool]:
        """
        Add a book observation to memory, handling deduplication.
        
        Args:
            book_data: Detection data from perception system
            camera_state: Current camera pan/tilt position
            
        Returns:
            Tuple of (book_id, is_novel) where is_novel indicates if this is a new unique book
        """
        with self.lock:
            # Extract physical properties
            properties = self._extract_physical_properties(book_data)
            
            # Create observation
            observation = BookObservation(
                timestamp=time.time(),
                camera_position=camera_state,
                perception_data=book_data
            )
            
            # Check for matches with existing memories
            match_id = self._find_matching_book(properties)
            
            if match_id:
                # Update existing book experience
                book_exp = self.memories[match_id]
                book_exp.observations.append(observation)
                book_exp.last_seen = observation.timestamp
                
                # Update physical properties if confidence is higher
                if observation.confidence > 0.8:
                    self._update_physical_properties(book_exp, properties)
                
                return match_id, False
            else:
                # Create new book experience
                book_id = f"book_{int(time.time()*1000)}_{len(self.memories)}"
                
                new_experience = BookExperience(
                    id=book_id,
                    first_seen=observation.timestamp,
                    last_seen=observation.timestamp,
                    physical_properties=properties,
                    observations=[observation]
                )
                
                self.memories[book_id] = new_experience
                
                # Track novel discovery
                self.last_novel_discovery = time.time()
                self.recent_discoveries.append(book_id)
                
                # Save memory periodically
                if len(self.memories) % 5 == 0:
                    self.save_memory()
                
                return book_id, True
    
    def _extract_physical_properties(self, book_data: Dict) -> BookPhysicalProperties:
        """Extract physical properties from perception data"""
        properties = BookPhysicalProperties(
            bbox=book_data.get("bbox", [0, 0, 1, 1]),
            world_position=np.array(book_data.get("world_position")) if "world_position" in book_data else None
        )
        
        # Extract additional properties if available
        if "size_estimate" in book_data:
            properties.size_estimate = book_data["size_estimate"]
        
        if "dominant_color" in book_data:
            properties.color_dominant = book_data["dominant_color"]
        
        return properties
    
    def _find_matching_book(self, properties: BookPhysicalProperties) -> Optional[str]:
        """Find if a book with similar properties already exists in memory"""
        for book_id, book_exp in self.memories.items():
            if self._properties_match(properties, book_exp.physical_properties):
                return book_id
        return None
    
    def _properties_match(self, prop1: BookPhysicalProperties, prop2: BookPhysicalProperties) -> bool:
        """Determine if two sets of physical properties represent the same book"""
        # Check world position similarity
        if prop1.world_position is not None and prop2.world_position is not None:
            distance = np.linalg.norm(prop1.world_position - prop2.world_position)
            if distance < self.position_threshold:
                return True
            elif distance > self.position_threshold * 2:
                return False  # Too far apart to be the same book
        
        # Check bounding box overlap (IoU)
        if prop1.bbox and prop2.bbox:
            iou = self._calculate_iou(prop1.bbox, prop2.bbox)
            if iou > self.bbox_iou_threshold:
                return True
        
        # Check size similarity if available
        if prop1.size_estimate and prop2.size_estimate:
            size_sim = self._calculate_size_similarity(prop1.size_estimate, prop2.size_estimate)
            if size_sim > self.size_similarity_threshold:
                return True
        
        # If we have high confidence in color and it matches
        if prop1.color_dominant and prop2.color_dominant:
            if prop1.color_dominant == prop2.color_dominant:
                # Combine with other weak signals
                position_close = False
                if prop1.world_position is not None and prop2.world_position is not None:
                    distance = np.linalg.norm(prop1.world_position - prop2.world_position)
                    position_close = distance < self.position_threshold * 1.5
                
                if position_close:
                    return True
        
        return False
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        # Convert to absolute coordinates if needed
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_size_similarity(self, size1: List[float], size2: List[float]) -> float:
        """Calculate similarity between two size estimates"""
        size1_array = np.array(size1)
        size2_array = np.array(size2)
        
        # Normalize by the larger size
        max_size = np.maximum(size1_array, size2_array)
        diff = np.abs(size1_array - size2_array)
        
        # Avoid division by zero
        similarity = 1.0 - np.mean(diff / (max_size + 1e-6))
        return max(0.0, min(1.0, similarity))
    
    def _update_physical_properties(self, book_exp: BookExperience, new_props: BookPhysicalProperties):
        """Update physical properties with more accurate measurements"""
        old_props = book_exp.physical_properties
        
        # Update world position with weighted average if both available
        if old_props.world_position is not None and new_props.world_position is not None:
            # Weight by number of observations
            old_weight = len(book_exp.observations) - 1
            new_weight = 1
            total_weight = old_weight + new_weight
            
            old_props.world_position = (
                old_props.world_position * old_weight + new_props.world_position * new_weight
            ) / total_weight
        elif new_props.world_position is not None:
            old_props.world_position = new_props.world_position
        
        # Update other properties if they're more complete
        if new_props.size_estimate and not old_props.size_estimate:
            old_props.size_estimate = new_props.size_estimate
        
        if new_props.color_dominant and not old_props.color_dominant:
            old_props.color_dominant = new_props.color_dominant
    
    def should_terminate_search(self) -> Tuple[bool, str]:
        """
        Determine if the search should terminate based on memory state.
        
        Returns:
            Tuple of (should_terminate, reason)
        """
        current_time = time.time()
        
        # Check for novelty timeout
        time_since_last_novel = current_time - self.last_novel_discovery
        if time_since_last_novel > self.novelty_window:
            # Check how many new books found in recent window
            recent_window_start = current_time - self.novelty_window
            recent_count = sum(1 for book_id in self.recent_discoveries 
                             if self.memories[book_id].first_seen > recent_window_start)
            
            if recent_count < self.novelty_count_threshold:
                return True, f"Low novelty: Only {recent_count} new books in last {self.novelty_window}s"
        
        # Check coverage if we have position data
        if self._calculate_search_coverage() > self.coverage_threshold:
            return True, f"High coverage: {self._calculate_search_coverage():.1%} of search space covered"
        
        # Check total number of unique books
        if len(self.memories) > 50:  # Arbitrary threshold
            return True, f"Memory limit: {len(self.memories)} unique books found"
        
        return False, "Continue searching"
    
    def _calculate_search_coverage(self) -> float:
        """Calculate the percentage of search space covered"""
        # Simple implementation - can be enhanced with actual spatial coverage
        if not self.memories:
            return 0.0
        
        # Count unique camera positions observed
        unique_positions = set()
        for book_exp in self.memories.values():
            for obs in book_exp.observations:
                pan = round(obs.camera_position.get("pan", 0) * 10) / 10
                tilt = round(obs.camera_position.get("tilt", 0) * 10) / 10
                unique_positions.add((pan, tilt))
        
        # Estimate total possible positions (simplified)
        total_positions = 20  # Approximate number of scan positions
        coverage = len(unique_positions) / total_positions
        
        return min(1.0, coverage)
    
    def get_memory_stats(self) -> Dict:
        """Get statistics about the current memory state"""
        with self.lock:
            total_books = len(self.memories)
            total_observations = sum(len(exp.observations) for exp in self.memories.values())
            
            # Recent discovery rate
            current_time = time.time()
            recent_window = 60  # Last minute
            recent_books = sum(1 for exp in self.memories.values() 
                             if exp.first_seen > current_time - recent_window)
            
            return {
                "total_unique_books": total_books,
                "total_observations": total_observations,
                "recent_discoveries": recent_books,
                "time_since_last_novel": current_time - self.last_novel_discovery,
                "search_coverage": self._calculate_search_coverage()
            }
    
    def get_book_experience(self, book_id: str) -> Optional[BookExperience]:
        """Retrieve a specific book experience"""
        return self.memories.get(book_id)
    
    def get_all_experiences(self) -> List[BookExperience]:
        """Get all book experiences"""
        with self.lock:
            return list(self.memories.values())
    
    def add_vlm_caption(self, book_id: str, caption: str):
        """Add a VLM caption to a book experience"""
        with self.lock:
            if book_id in self.memories:
                self.memories[book_id].vlm_captions.append(caption)
    
    def update_spatial_context(self, book_id: str, context: Dict):
        """Update spatial context for a book"""
        with self.lock:
            if book_id in self.memories:
                self.memories[book_id].spatial_context = context