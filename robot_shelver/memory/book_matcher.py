"""
Enhanced book matching algorithm for improved duplicate detection.
Uses multiple matching criteria with configurable weights.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re
import time

class BookMatcher:
    """Advanced matching system for detecting duplicate book observations."""
    
    def __init__(self, config=None):
        """
        Initialize the BookMatcher with configurable parameters.
        
        Args:
            config: Configuration dictionary with matching parameters
        """
        self.config = config or {}
        
        # Set much stricter thresholds to improve deduplication
        self.spatial_threshold = self.config.get('matching', {}).get('spatial_threshold', 0.2)  # reduced from 0.3 meters
        self.visual_similarity_threshold = self.config.get('matching', {}).get('visual_threshold', 0.85)  # increased from 0.8
        self.temporal_threshold = self.config.get('matching', {}).get('temporal_threshold', 30)  # reduced from 60 seconds
        
        # Significantly increased spatial weight for better deduplication
        self.weights = {
            'spatial': self.config.get('matching', {}).get('weights', {}).get('spatial', 0.8),  # increased from 0.7
            'description': self.config.get('matching', {}).get('weights', {}).get('description', 0.15),  # reduced from 0.2
            'visual': self.config.get('matching', {}).get('weights', {}).get('visual', 0.05),  # reduced from 0.1
            'temporal': self.config.get('matching', {}).get('weights', {}).get('temporal', 0.0)
        }
        
        # Normalize weights to sum to 1.0
        weight_sum = sum(self.weights.values())
        if weight_sum > 0:
            for key in self.weights:
                self.weights[key] /= weight_sum
                
        # Confidence degradation factors
        self.confidence_decay_rate = self.config.get('matching', {}).get('confidence_decay', 0.95)
        
        # Debug mode
        self.debug = self.config.get('debug', False)
    
    def is_same_book(self, book1: Any, book2: Any, threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Determine if two book observations likely represent the same book.
        
        Args:
            book1: First book node/data
            book2: Second book node/data
            threshold: Similarity threshold (0-1) for considering books as the same
            
        Returns:
            Tuple of (is_match: bool, similarity_score: float)
        """
        # Get book attributes safely
        book1_attrs = getattr(book1, 'attributes', book1) or {}
        book2_attrs = getattr(book2, 'attributes', book2) or {}
        
        # Get descriptions - handling different object types
        if hasattr(book1, 'description'):
            book1_desc = book1.description
        elif isinstance(book1, dict) and 'description' in book1:
            book1_desc = book1['description']
        else:
            book1_desc = ''
            
        if hasattr(book2, 'description'):
            book2_desc = book2.description
        elif isinstance(book2, dict) and 'description' in book2:
            book2_desc = book2['description']
        else:
            book2_desc = ''
        
        # Calculate similarity scores for different factors
        spatial_sim = self._spatial_similarity(book1_attrs, book2_attrs)
        description_sim = self._description_similarity(book1_desc, book2_desc)
        visual_sim = self._visual_similarity(book1_attrs, book2_attrs)
        temporal_sim = self._temporal_similarity(book1_attrs, book2_attrs)
        
        # Combine scores using weights
        combined_score = (
            self.weights['spatial'] * spatial_sim +
            self.weights['description'] * description_sim + 
            self.weights['visual'] * visual_sim +
            self.weights['temporal'] * temporal_sim
        )
        
        if self.debug:
            print(f"Matching scores: spatial={spatial_sim:.2f}, description={description_sim:.2f}, "
                  f"visual={visual_sim:.2f}, temporal={temporal_sim:.2f}, combined={combined_score:.2f}")
            
        # Return match decision and score
        return combined_score >= threshold, combined_score
    
    def _spatial_similarity(self, attrs1: Dict, attrs2: Dict) -> float:
        """Calculate similarity based on physical proximity with enhanced accuracy."""
        pos1 = attrs1.get('world_position')
        pos2 = attrs2.get('world_position')
        
        # If either position is missing, can't use spatial matching
        if not pos1 or not pos2:
            return 0.0
            
        try:
            # Calculate Euclidean distance
            distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
            
            # Much stricter distance matching
            # Extremely close (within spatial_threshold * 0.5) - nearly guaranteed to be the same book
            if distance < self.spatial_threshold * 0.5:  # Under 10cm with default threshold
                # Scale from 0.95 to 1.0 for very close books
                similarity = 0.95 + min(0.05, 1.0 - distance / (self.spatial_threshold * 0.5))
                print(f"Very close spatial match at {distance:.3f}m (similarity: {similarity:.2f})")
                return float(similarity)
                
            # Fairly close (within spatial_threshold) - very likely the same book
            elif distance < self.spatial_threshold:  # Under 20cm with default threshold
                # Scale from 0.8 to 0.95
                similarity = 0.8 + 0.15 * (1 - distance / self.spatial_threshold)
                return float(similarity)
                
            # Somewhat close (within 1.5x threshold) - possibly the same book but requires other evidence
            elif distance < self.spatial_threshold * 1.5:  # Under 30cm with default threshold
                # Scale from 0.5 to 0.8
                similarity = 0.5 + 0.3 * (1 - (distance - self.spatial_threshold) / (self.spatial_threshold * 0.5))
                return float(similarity)
            
            # Further away - exponential decay for distances beyond 1.5x threshold
            else:
                # Convert distance to similarity score using sharper exponential decay
                similarity = np.exp(-distance / (self.spatial_threshold * 0.8))  # Faster decay
                return float(min(similarity, 0.5))  # Cap at 0.5 for distant books
                
        except Exception as e:
            print(f"Error calculating spatial similarity: {e}")
            return 0.0
    
    def _description_similarity(self, desc1: str, desc2: str) -> float:
        """Calculate similarity based on textual descriptions."""
        if not desc1 or not desc2:
            return 0.0
            
        # Normalize descriptions
        desc1 = desc1.lower().strip()
        desc2 = desc2.lower().strip()
        
        # Exact match
        if desc1 == desc2:
            return 1.0
            
        # Substring match (one contains the other)
        if desc1 in desc2 or desc2 in desc1:
            return 0.8
        
        # Extract key properties using regular expressions
        type1 = self._extract_book_type(desc1)
        type2 = self._extract_book_type(desc2)
        
        color1 = self._extract_color(desc1)
        color2 = self._extract_color(desc2)
        
        # Compare extracted properties
        type_match = type1 and type2 and (type1 == type2 or type1 in type2 or type2 in type1)
        color_match = color1 and color2 and (color1 == color2 or color1 in color2 or color2 in color1)
        
        # Calculate similarity based on property matches
        if type_match and color_match:
            return 0.9
        elif type_match:
            return 0.7
        elif color_match:
            return 0.6
        
        # Word overlap ratio as fallback
        words1 = set(re.findall(r'\b\w+\b', desc1))
        words2 = set(re.findall(r'\b\w+\b', desc2))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        return jaccard_similarity * 0.6  # Scale down the pure word-based similarity
    
    def _visual_similarity(self, attrs1: Dict, attrs2: Dict) -> float:
        """Calculate similarity based on visual properties."""
        # Compare bounding box dimensions if available
        bbox1 = attrs1.get('bbox')
        bbox2 = attrs2.get('bbox')
        
        if not bbox1 or not bbox2 or len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0
            
        # Calculate width/height ratio
        width1 = bbox1[2] - bbox1[0]
        height1 = bbox1[3] - bbox1[1]
        ratio1 = width1 / height1 if height1 > 0 else 0
        
        width2 = bbox2[2] - bbox2[0]
        height2 = bbox2[3] - bbox2[1]
        ratio2 = width2 / height2 if height2 > 0 else 0
        
        # Compare ratios (closer = more similar)
        if ratio1 <= 0 or ratio2 <= 0:
            return 0.0
            
        ratio_diff = abs(ratio1 - ratio2)
        ratio_similarity = max(0, 1.0 - ratio_diff / max(ratio1, ratio2))
        
        # If we had more visual features (color histograms, etc.), we could add them here
        
        return ratio_similarity
    
    def _temporal_similarity(self, attrs1: Dict, attrs2: Dict) -> float:
        """Calculate similarity based on observation time proximity."""
        time1 = attrs1.get('timestamp')
        time2 = attrs2.get('timestamp')
        
        if not time1 or not time2:
            return 0.0
            
        # Calculate time difference in seconds
        time_diff = abs(time1 - time2)
        
        # Convert to similarity score (more recent = higher similarity)
        # Use exponential decay: e^(-time_diff/threshold)
        similarity = np.exp(-time_diff / self.temporal_threshold)
        
        return float(similarity)
    
    def _extract_book_type(self, text: str) -> str:
        """Extract book type from description."""
        book_types = [
            'textbook', 'novel', 'dictionary', 'encyclopedia', 
            'manual', 'guide', 'book', 'notebook', 'journal',
            'publication', 'magazine'
        ]
        
        for book_type in book_types:
            if book_type in text:
                return book_type
                
        return ""
    
    def _extract_color(self, text: str) -> str:
        """Extract color information from description."""
        colors = [
            'red', 'green', 'blue', 'yellow', 'orange', 'purple',
            'pink', 'brown', 'black', 'white', 'gray', 'grey'
        ]
        
        for color in colors:
            if color in text:
                return color
                
        return ""
    
    def update_match_confidence(self, match_score: float, existing_confidence: float) -> float:
        """Update confidence value for a matched book."""
        # Confidence starts to degrade when match score is below 0.9
        confidence_scale = 1.0 if match_score >= 0.9 else self.confidence_decay_rate
        
        # Update with weighted average between new and existing confidence
        updated_confidence = existing_confidence * confidence_scale
        
        # Ensure confidence doesn't drop below a minimum threshold
        min_confidence = 0.3
        return max(min_confidence, updated_confidence)
    
    def get_best_matching_node(self, new_node: Any, existing_nodes: List[Any], 
                              min_threshold: float = 0.7) -> Tuple[Optional[Any], float]:
        """
        Find the best matching existing node for a new observation with greatly enhanced deduplication logic.
        
        Args:
            new_node: New book observation
            existing_nodes: List of existing book nodes
            min_threshold: Minimum similarity threshold for a match
            
        Returns:
            Tuple of (best_match_node, similarity_score)
        """
        best_match = None
        best_score = 0.0
        all_candidates = []
        
        # VERY STRICT SPATIAL MATCHING - this is now the primary deduplication strategy
        # Get new node position
        new_attrs = getattr(new_node, 'attributes', {}) or {}
        new_pos = new_attrs.get('world_position')
        
        spatial_matches = []
        if new_pos:
            # Find nodes that are very close in space (strict spatial matching)
            for node in existing_nodes:
                node_attrs = getattr(node, 'attributes', {}) or {}
                node_pos = node_attrs.get('world_position')
                
                if node_pos:
                    try:
                        # Calculate distance
                        distance = np.linalg.norm(np.array(new_pos) - np.array(node_pos))
                        
                        # MUCH STRICTER THRESHOLDS FOR SPATIAL MATCHING
                        # If extremely close (within 20cm), this is certainly the same book
                        if distance < 0.2:
                            # Basically guarantee a match with 0.99 score
                            spatial_matches.append((node, 0.99, distance))
                            print(f"EXACT MATCH: Book at distance {distance:.3f}m")
                        # If very close (within 30cm), this is almost certainly the same book
                        elif distance < 0.3:
                            spatial_matches.append((node, 0.95 + min(0.05, 1.0 - distance), distance))
                            print(f"VERY CLOSE MATCH: Book at distance {distance:.3f}m")
                        # If close (within 40cm), it's likely the same book
                        elif distance < 0.4:
                            spatial_matches.append((node, 0.90 + min(0.05, 1.0 - distance), distance))
                            print(f"CLOSE MATCH: Book at distance {distance:.3f}m")
                    except Exception as e:
                        if self.debug:
                            print(f"Error computing distance: {e}")
        
        # If we found close spatial matches, always prioritize them over description matching
        if spatial_matches:
            # Sort by distance (lower is better)
            spatial_matches.sort(key=lambda x: x[2])
            best_spatial_node, best_spatial_score, distance = spatial_matches[0]
            
            # Always log spatial matches regardless of debug mode
            print(f"Found close spatial match at {distance:.3f}m with score {best_spatial_score:.2f}")
            
            # For very close matches (within 20cm), skip additional checks
            if distance < 0.2:
                return best_spatial_node, best_spatial_score
                
            # For other spatial matches, also check if descriptions reasonably match
            # to avoid mistakenly matching different books that are close to each other
            new_desc = getattr(new_node, 'description', '')
            existing_desc = getattr(best_spatial_node, 'description', '')
            
            # Very basic description similarity check
            desc_similarity = self._description_similarity(new_desc, existing_desc)
            print(f"Description similarity: {desc_similarity:.2f}")
            
            # If descriptions are completely different but books are close,
            # we still trust position more than description (books don't move by themselves)
            if desc_similarity < 0.3 and distance < 0.3:
                print(f"Warning: Descriptions differ significantly but positions match closely")
                # Still use spatial match with slightly reduced confidence
                return best_spatial_node, best_spatial_score * 0.95
                
            # Normal case - good spatial and description match
            return best_spatial_node, best_spatial_score
            
        # If no good spatial match, do full matching with all criteria as a fallback
        for node in existing_nodes:
            is_match, score = self.is_same_book(new_node, node, threshold=min_threshold)
            
            # Store all decent matches for analysis
            if score > min_threshold * 0.8:  # Track any reasonably close match
                all_candidates.append((node, score))
            
            if is_match and score > best_score:
                best_match = node
                best_score = score
        
        # Enhanced logging for debugging matches
        if len(all_candidates) > 1:
            print(f"Multiple potential matches found ({len(all_candidates)}):")
            for idx, (node, score) in enumerate(sorted(all_candidates, key=lambda x: x[1], reverse=True)[:3]):
                node_desc = getattr(node, 'description', str(node))
                node_pos = getattr(node, 'attributes', {}).get('world_position', 'Unknown')
                print(f"  {idx+1}. Score: {score:.2f}, Description: {node_desc[:30]}..., Pos: {node_pos}")
        
        return best_match, best_score