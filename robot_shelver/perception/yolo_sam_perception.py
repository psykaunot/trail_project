import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import sys
import os
import cv2
import json
import time
from datetime import datetime
from pathlib import Path
import habitat_sim
import magnum as mn
# Add Embodied_RAG to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../Embodied_RAG'))

from embodied_rag_stubs import SpatialRelationshipExtractor, OllamaLLM

class YoloSamPerception:
    def __init__(self, use_yolo_sam: bool = True, confidence_threshold: float = 0.5):
        # Create directories for logging
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(current_file_dir, "perception_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create LLM interface for SpatialRelationshipExtractor
        self.llm_interface = OllamaLLM(model="qwen3:8b")
        self.spatial_extractor = SpatialRelationshipExtractor(llm_interface=self.llm_interface)
        
        # YOLO-SAM specific parameters
        self.use_yolo_sam = use_yolo_sam
        self.yolo_detector = None
        self.sam_predictor = None
        self.confidence_threshold = confidence_threshold
        self.camera_params = None
        self.sim = None
        
        # Cache frequently used objects
        self.object_classes = [
            'book', 'books', 'textbook', 'notebook', 'novel', 'journal', 'magazine',
            'dictionary', 'encyclopedia', 'manual', 'document', 'folder', 'binder',
            'thesis', 'paper', 'publication', 'guide', 'handbook', 'almanac'
        ]
        
        # Objects that could visually resemble books
        self.book_like_objects = [
            'box', 'suitcase', 'handbag', 'backpack', 'briefcase', 'luggage',
            'laptop', 'rectangle', 'object', 'package', 'container', 'cell phone',
            'remote', 'tray', 'tv', 'monitor', 'keyboard', 'bowl', 'cup', 'chair', 
            'umbrella', 'vase', 'potted plant', 'clock', 'bottle', 'bench'
        ]
        
        print(f"YOLO-SAM perception initialized (use_yolo_sam={use_yolo_sam})")
        
    def find_books_in_image(self, image: np.ndarray, confidence_threshold: float = None) -> List[Dict]:
        """
        Find books using YOLO-SAM.
        
        Args:
            image: Input image as numpy array
            confidence_threshold: Optional override for the confidence threshold
            
        Returns:
            List of dictionaries with book detection information
        """
        if image is None or image.size == 0:
            print("Warning: Invalid image provided to find_books_in_image")
            return []
            
        try:
            # Use provided confidence threshold or fall back to default
            threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
            print(f"Using confidence threshold: {threshold}")
            
            # Preprocess image (color correction)
            processed_image = self._preprocess_image(image)
            
            # Initialize YOLO if needed
            if self.use_yolo_sam:
                if self.yolo_detector is None:
                    print("YOLO detector not initialized, initializing now...")
                    self._initialize_yolo_detector()
                    
                if self.yolo_detector is None:
                    print("Failed to initialize YOLO detector")
                    return []
                
                # Detect books using YOLO
                return self._yolo_detect_books(processed_image)
            else:
                print("YOLO-SAM is disabled in configuration")
                return []
                
        except Exception as e:
            print(f"Error in YOLO book detection: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error case image
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(self.log_dir, f"yolo_error_{timestamp}.png")
                cv2.imwrite(img_path, image)
                print(f"Error in detection, saved image: {img_path}")
            except:
                print("Could not save error image")
                
            return []
        
    def _initialize_yolo_detector(self):
        """Initialize YOLO detector with proper error handling."""
        try:
            # Try to import required packages
            from ultralytics import YOLO
            
            # Try to load a custom model first, fallback to standard model
            custom_model_paths = [
                "yolov8n-books.pt",     # Custom books-focused model if available
                "yolov8n-custom.pt",    # Custom-named model if available
                "yolov8n.pt"            # Standard YOLO model as fallback
            ]
            
            # Check various paths where the model might be
            search_dirs = [
                os.getcwd(),
                os.path.dirname(os.getcwd()),
                os.path.dirname(os.path.dirname(os.getcwd())),
                os.path.join(os.path.dirname(__file__), '../..'),  # Robot shelver root directory
                os.path.join(os.path.dirname(__file__), '..'),     # Robot shelver parent directory
                os.path.dirname(__file__),                         # Current directory
                os.path.expanduser("~/.cache/ultralytics"),        # Default YOLO cache dir
                "/home/ubuntu/trail_project/habitat-lab/robot_shelver"  # Absolute path to project
            ]
            
            # Create cache directory if it doesn't exist
            cache_dir = os.path.expanduser("~/.cache/ultralytics")
            os.makedirs(cache_dir, exist_ok=True)
            
            model_path = None
            
            # Find first existing model
            for custom_model in custom_model_paths:
                for search_dir in search_dirs:
                    candidate_path = os.path.join(search_dir, custom_model)
                    if os.path.exists(candidate_path):
                        model_path = candidate_path
                        print(f"Found YOLO model at: {model_path}")
                        break
                if model_path is not None:
                    break
            
            # If no model found in search paths, check if default model can be used
            if model_path is None:
                model_path = "yolov8n.pt"
                print(f"No existing model found, will use/download: {model_path}")
            
            # Initialize YOLO model
            self.yolo_detector = YOLO(model_path)
            print(f"YOLO detector initialized with model: {model_path}")
            
            # Try to initialize SAM if available, but make it optional
            self.sam_predictor = None  # Default to None
            
            try:
                # First check if segment_anything module is available
                try:
                    import torch
                    import segment_anything
                    has_sam_module = True
                    print("segment_anything module is available")
                except ImportError:
                    has_sam_module = False
                    print("segment_anything module is not installed - SAM segmentation will not be available")
                
                # Only try to load SAM if the module is available
                if has_sam_module:
                    from segment_anything import SamPredictor, sam_model_registry
                    sam_checkpoint = "sam_vit_b_01ec64.pth" 
                    model_type = "vit_b"
                    
                    # Check multiple locations for the SAM model
                    sam_path = None
                    for search_dir in search_dirs:
                        candidate_path = os.path.join(search_dir, sam_checkpoint)
                        if os.path.exists(candidate_path):
                            sam_path = candidate_path
                            print(f"Found SAM model at: {sam_path}")
                            break
                    
                    # Also check the cache directory
                    if sam_path is None:
                        candidate_path = os.path.join(cache_dir, sam_checkpoint)
                        if os.path.exists(candidate_path):
                            sam_path = candidate_path
                            print(f"Found SAM model at: {sam_path}")
                    
                    if sam_path is None:
                        print("SAM model file not found, SAM segmentation will not be available")
                    else:
                        try:
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            sam = sam_model_registry[model_type](checkpoint=sam_path)
                            sam.to(device=device)
                            self.sam_predictor = SamPredictor(sam)
                            print(f"SAM initialized successfully on {device}")
                        except Exception as e:
                            print(f"SAM initialization failed during model loading: {e}")
            except Exception as e:
                print(f"SAM initialization skipped: {e}")
                print("Object detection will continue without segmentation masks")
                
        except Exception as e:
            print(f"Error initializing YOLO detector: {e}")
            import traceback
            traceback.print_exc()
            self.yolo_detector = None
        
    def _preprocess_image(self, image, save_debug=True):
        """Handle image format conversion while preserving original colors."""
        if image is None or image.size == 0:
            return image

        try:
            # Save original image for debugging before any processing
            original_image = image.copy() if save_debug else None
            
            # Verify actual image format instead of assuming RGBA
            if image.ndim == 3:
                if image.shape[2] == 4:
                    # Only convert if truly needed - check if alpha channel is used
                    alpha_channel = image[:,:,3]
                    if not np.all(alpha_channel == 255):  # If alpha has non-255 values
                        print("Converting RGBA image to RGB (with alpha channel)...")
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    else:
                        # Alpha channel is all 255, just slice RGB channels
                        image = image[:,:,:3]
            
            # Skip processing if not RGB
            if image.ndim != 3 or image.shape[2] != 3:
                print(f"Image format not suitable for preprocessing: shape={image.shape}")
                return image
                
            # Instead of applying color correction, keep original colors
            # This preserves the natural appearance in the saved images
            processed = image  # No color correction - use original image directly

            # Removed debug image saving for color correction to reduce clutter

            return processed
        except Exception as e:
            print(f"Image preprocessing error: {e}")
            return image

    def _yolo_detect_books(self, image: np.ndarray) -> List[Dict]:
        """Detect books using YOLO with optional SAM refinement and additional fallback."""
        if self.yolo_detector is None:
            print("YOLO detector not initialized")
            return []
            
        books = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Use extremely low confidence threshold for books to catch ANY potential books
            book_confidence_threshold = max(0.15, self.confidence_threshold - 0.35)  # Minimal threshold for books
            print(f"Using minimal confidence threshold for books: {book_confidence_threshold:.2f}")
            
            # Get predictions from YOLO
            # Use higher IoU threshold for detecting stacked books
            results = self.yolo_detector(image, conf=book_confidence_threshold, iou=0.3)
            
            # Process detection results
            for result in results:
                # Extract all detected objects
                for i, (box, cls, conf) in enumerate(zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy()
                )):
                    # Get class name
                    class_name = result.names[int(cls)]
                    
                    # Check if it's a book-related class or has book in name
                    is_book = class_name.lower() in self.object_classes or 'book' in class_name.lower()
                    
                    # Also detect objects that might be books (we'll be more lenient)
                    might_be_book = class_name.lower() in self.book_like_objects
                    
                    if (is_book or might_be_book) and conf >= book_confidence_threshold:
                        # Format bounding box as normalized [x_min, y_min, x_max, y_max]
                        x1, y1, x2, y2 = box
                        h, w = image.shape[:2]
                        norm_bbox = [float(x1/w), float(y1/h), float(x2/w), float(y2/h)]
                        
                        # Check if detection is likely a robot part (lower portion of image)
                        y_center = (norm_bbox[1] + norm_bbox[3]) / 2
                        is_in_robot_zone = y_center > 0.6  # Bottom 40% of image
                        is_common_false_positive = class_name in ['airplane', 'boat', 'bench']
                        
                        # Calculate physical dimensions based on bounding box and depth
                        # First get real-world width and height in meters (if we can)
                        physical_width_m = None
                        physical_height_m = None
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        if self.sim and self.camera_params:
                            try:
                                # Get world position for center point
                                sensor_obj = self.camera_params["sensor_obj"]
                                render_camera = sensor_obj.render_camera
                                ray = render_camera.unproject(mn.Vector2i(center_x, center_y))
                                raycast_results = self.sim.cast_ray(ray)
                                
                                if raycast_results.has_hits():
                                    # Get world position
                                    center_point = raycast_results.hits[0].point
                                    
                                    # Try to get world positions for corners to estimate size
                                    corner1_x = int(x1)
                                    corner1_y = int(y1)
                                    corner2_x = int(x2) 
                                    corner2_y = int(y2)
                                    
                                    ray1 = render_camera.unproject(mn.Vector2i(corner1_x, corner1_y))
                                    ray2 = render_camera.unproject(mn.Vector2i(corner2_x, corner2_y)) 
                                    
                                    raycast1 = self.sim.cast_ray(ray1)
                                    raycast2 = self.sim.cast_ray(ray2)
                                    
                                    if raycast1.has_hits() and raycast2.has_hits():
                                        corner1_point = raycast1.hits[0].point
                                        corner2_point = raycast2.hits[0].point
                                        
                                        # Calculate physical dimensions
                                        physical_width_m = np.linalg.norm(
                                            np.array([corner2_point[0], corner2_point[2]]) - 
                                            np.array([corner1_point[0], corner1_point[2]])
                                        )
                                        physical_height_m = np.linalg.norm(
                                            np.array([corner2_point[1]]) - 
                                            np.array([corner1_point[1]])
                                        )
                            except Exception as e:
                                print(f"Error calculating physical dimensions: {e}")
                        
                        # Skip if physical dimensions are available and unreasonable for a book
                        # Typical book dimensions: width 15-30cm, height 20-35cm
                        MAX_BOOK_WIDTH_M = 0.30   # 30cm max width
                        MAX_BOOK_HEIGHT_M = 0.35  # 35cm max height
                        
                        if physical_width_m and physical_height_m:
                            if physical_width_m > MAX_BOOK_WIDTH_M or physical_height_m > MAX_BOOK_HEIGHT_M:
                                print(f"Ignoring oversized object ({physical_width_m:.2f}m × {physical_height_m:.2f}m)")
                                continue
                        
                        # Skip robot parts
                        if is_in_robot_zone and is_common_false_positive:
                            print(f"Ignoring likely robot part detected as {class_name} in bottom of frame")
                            continue
                        
                        # Create book entry
                        book_name = "book" if might_be_book else class_name
                        detection_quality = "high" if conf > 0.5 else "medium" if conf > 0.35 else "low"
                        
                        # More detailed description based on detection quality
                        if might_be_book:
                            description = f"A possible book (detected as {class_name}) with {conf:.2f} confidence ({detection_quality})"
                        else:
                            description = f"A {book_name} detected by YOLO with {conf:.2f} confidence ({detection_quality})"
                            
                        book = {
                            "name": book_name,
                            "description": description,
                            "bbox": norm_bbox,
                            "confidence": float(conf),
                            "detection_method": "yolo",
                            "detection_quality": detection_quality,
                            "original_class": class_name,
                            "might_be_book": might_be_book,  # Track objects that may be books
                            "y_center": y_center  # Store position for filtering
                        }
                        
                        # Apply SAM segmentation if available (but don't require it)
                        mask = None
                        if self.sam_predictor is not None:
                            try:
                                mask = self._apply_sam_segmentation(image, box)
                                if mask is not None:
                                    book["segmentation_mask"] = mask
                                    book["detection_method"] = "yolo_sam"
                            except Exception as e:
                                print(f"SAM segmentation failed for this detection: {e}")
                                # Continue without segmentation mask
                        else:
                            # SAM not available - add a note to the detection
                            book["segmentation_available"] = False
                                
                        books.append(book)
            
            # Try specialized detection if no books found with regular YOLO
            if not books:
                # Attempt specialized book detection for difficult scenes
                specialized_books = self._specialized_book_detection(image)
                if specialized_books:
                    print(f"Found {len(specialized_books)} books with specialized detection")
                    books.extend(specialized_books)
                
            # Post-process all detected books
            books = self._post_process_book_detections(books)
            
            # Save visualization for debugging
            if books:
                # Make sure image is RGB for drawing but avoid unnecessary conversions
                if image.shape[2] == 4:  # RGBA
                    # Check if alpha channel is actually used
                    alpha_channel = image[:,:,3]
                    if not np.all(alpha_channel == 255):  # If alpha has non-255 values
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    else:
                        # Alpha is all 255, just take RGB channels
                        rgb_image = image[:,:,:3].copy()
                else:
                    rgb_image = image.copy()
                    
                annotated_img = self._draw_detections(rgb_image, books)
                debug_path = os.path.join(self.log_dir, f"yolo_books_{timestamp}.jpg")
                cv2.imwrite(debug_path, annotated_img)
                print(f"Saved YOLO detection visualization to {debug_path}")
                
            return books
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all objects in an image using YOLO.
        
        Args:
            image: Image array to perform detection on
            
        Returns:
            List of dictionaries with object data
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            print("Warning: Invalid image provided to detect_objects")
            return []
            
        try:
            # Process image
            processed_image = self._preprocess_image(image)
            
            # Initialize YOLO if needed
            if self.yolo_detector is None:
                self._initialize_yolo_detector()
                if self.yolo_detector is None:
                    print("Failed to initialize YOLO detector")
                    return []
            
            # Run YOLO detection with increased IoU threshold and lower confidence
            # A higher IoU threshold helps with overlapping books on shelves
            results = self.yolo_detector(processed_image, conf=self.confidence_threshold, iou=0.3)
            
            objects = []
            for result in results:
                for i, (box, cls, conf) in enumerate(zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy(),
                    result.boxes.conf.cpu().numpy()
                )):
                    # Get class name
                    class_name = result.names[int(cls)]
                    
                    # Format bounding box as normalized coordinates
                    x1, y1, x2, y2 = box
                    h, w = image.shape[:2]
                    norm_bbox = [float(x1/w), float(y1/h), float(x2/w), float(y2/h)]
                    
                    # Create object entry
                    obj = {
                        "name": class_name,
                        "description": f"A {class_name}",
                        "bbox": norm_bbox,
                        "confidence": float(conf),
                        "detection_method": "yolo"
                    }
                    objects.append(obj)
            
            # Save result visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Make sure image is RGB for drawing but avoid unnecessary conversions
            if image.shape[2] == 4:  # RGBA
                # Check if alpha channel is actually used
                alpha_channel = image[:,:,3]
                if not np.all(alpha_channel == 255):  # If alpha has non-255 values
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    # Alpha is all 255, just take RGB channels
                    rgb_image = image[:,:,:3].copy()
            else:
                rgb_image = image.copy()
                
            annotated_img = self._draw_detections(rgb_image, objects)
            debug_path = os.path.join(self.log_dir, f"yolo_objects_{timestamp}.jpg")
            cv2.imwrite(debug_path, annotated_img)
            
            return objects
                
        except Exception as e:
            print(f"Error in YOLO object detection: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    def _apply_sam_segmentation(self, image: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """Apply SAM segmentation to refine object boundaries."""
        # Safety check - if SAM isn't available, don't try to use it
        if self.sam_predictor is None:
            return None
            
        try:
            # Validate inputs
            if image is None or not isinstance(image, np.ndarray):
                print("Invalid image provided to SAM segmentation")
                return None
                
            if box is None or not isinstance(box, np.ndarray):
                print("Invalid bounding box provided to SAM segmentation")
                return None
                
            # Check image shape
            if len(image.shape) < 3 or image.shape[2] < 3:
                print(f"Image shape not suitable for SAM: {image.shape}")
                return None
            
            # Handle various image formats carefully
            if image.shape[2] == 4:  # RGBA
                # Check if alpha channel is actually used
                alpha_channel = image[:,:,3]
                if not np.all(alpha_channel == 255):  # If alpha has non-255 values
                    # True RGBA image with transparency
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    # Alpha is all 255, just take RGB channels
                    image_rgb = image[:,:,:3].copy()
            elif image.shape[2] == 3:  # BGR/RGB - assume RGB for SAM
                image_rgb = image.copy()
            else:
                print(f"Unexpected image channels: {image.shape[2]}")
                return None
                
            # Set image in predictor with timeout protection
            try:
                # This can sometimes hang if the model is not loaded correctly
                self.sam_predictor.set_image(image_rgb)
            except Exception as e:
                print(f"SAM set_image failed: {e}")
                return None
            
            # Get xyxy coordinates with bounds checking
            try:
                h, w = image.shape[:2]
                x1, y1, x2, y2 = box.astype(int)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w-1))
                y2 = max(0, min(y2, h-1))
                
                # Ensure box has some area
                if x1 >= x2 or y1 >= y2:
                    print("Invalid box dimensions for SAM")
                    return None
            except Exception as e:
                print(f"SAM box processing error: {e}")
                return None
            
            # Get masks from SAM
            try:
                masks, scores, _ = self.sam_predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=True
                )
                
                # Return highest-scoring mask if any masks were found
                if len(scores) > 0:
                    best_mask_idx = np.argmax(scores)
                    if scores[best_mask_idx] > 0.5:  # Only use mask if score is reasonable
                        return masks[best_mask_idx]
                    else:
                        print(f"SAM mask score too low: {scores[best_mask_idx]:.2f}")
                else:
                    print("SAM returned no masks")
            except Exception as e:
                print(f"SAM predict failed: {e}")
            
            return None
            
        except Exception as e:
            print(f"SAM segmentation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _post_process_book_detections(self, books: List[Dict]) -> List[Dict]:
        """
        Post-process book detections to improve accuracy:
        1. Remove duplicate detections (overlapping boxes)
        2. Boost confidence for books on shelves or in clusters
        3. Filter out likely false positives
        4. Filter out robot parts (typically in bottom portion of image)
        """
        if not books:
            return []
            
        try:
            # Sort by confidence (highest first)
            books = sorted(books, key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Filter out likely robot parts (usually in lower part of image)
            # Robot parts are often in the bottom 40% of the image
            non_robot_books = []
            for book in books:
                bbox = book.get('bbox')
                if bbox:
                    # Check if object is primarily in the lower part of the frame
                    # which is likely to be the robot itself
                    y_center = (bbox[1] + bbox[3]) / 2
                    
                    # Skip if the center of the bounding box is in the bottom 40% of the image
                    # AND it's detected as an airplane or other common false positive
                    is_in_robot_zone = y_center > 0.6  # Bottom 40% of image
                    is_common_false_positive = book.get('original_class') in ['airplane', 'boat', 'bench']
                    
                    if is_in_robot_zone and is_common_false_positive:
                        print(f"Filtering out likely robot part detected as {book.get('original_class')}")
                        continue
                    
                non_robot_books.append(book)
            
            # Remove overlapping detections (with high IoU)
            filtered_books = []
            for book in non_robot_books:
                # Check if this book overlaps significantly with any higher-confidence book
                bbox1 = book.get('bbox')
                is_duplicate = False
                
                for filtered_book in filtered_books:
                    bbox2 = filtered_book.get('bbox')
                    if bbox1 and bbox2:
                        iou = self._calculate_iou(bbox1, bbox2)
                        # If IoU is high, consider it a duplicate
                        if iou > 0.65:  # Higher threshold for considering duplicates
                            is_duplicate = True
                            break
                
                if not is_duplicate:
                    filtered_books.append(book)
            
            # Boost confidence for books that are aligned (likely on shelves)
            boosted_books = filtered_books.copy()
            
            # Find horizontal alignment patterns (books on shelves)
            y_values = [book['bbox'][1] for book in boosted_books if 'bbox' in book]  # y coordinates
            for i, book in enumerate(boosted_books):
                if 'bbox' not in book:
                    continue
                    
                # Check if there are other books at similar height (shelf alignment)
                y_coord = book['bbox'][1]
                aligned_count = sum(1 for y in y_values if abs(y - y_coord) < 0.05)
                
                # If multiple books are aligned horizontally, likely a shelf
                if aligned_count >= 3:
                    # Boost confidence for books on detected shelves
                    boosted_books[i]['confidence'] = min(1.0, book['confidence'] * 1.2)
                    boosted_books[i]['alignment_boost'] = True
            
            # Add additional insights for each book
            for book in boosted_books:
                # Add dimensions (width/height ratio helps identify books)
                if 'bbox' in book:
                    bbox = book['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    book['width_height_ratio'] = width / height if height > 0 else 0
                    
                    # Most books have width/height ratio around 0.7-0.8
                    is_book_shape = 0.4 <= book['width_height_ratio'] <= 1.2
                    book['is_book_shape'] = is_book_shape
                    
                    # Adjust confidence based on shape
                    if 'might_be_book' in book and book['might_be_book'] and is_book_shape:
                        book['confidence'] = min(1.0, book['confidence'] * 1.15)
            
            return boosted_books
                
        except Exception as e:
            print(f"Error in post-processing book detections: {e}")
            import traceback
            traceback.print_exc()
            return books
    
    def _specialized_book_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Specialized book detection for difficult scenes.
        Uses edge detection, contour analysis, and geometric heuristics
        to find book-like rectangular objects.
        """
        if image is None or image.size == 0 or image.ndim != 3:
            return []
            
        try:
            # Convert to grayscale for edge detection
            if image.shape[2] >= 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
                
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Apply Canny edge detection
            edges = cv2.Canny(filtered, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            potential_books = []
            
            # Check each contour
            for contour in contours:
                # Calculate perimeter and approximate the contour
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
                
                # Check if it's a rectangle (4 points) and large enough
                if len(approx) >= 4 and len(approx) <= 6:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(approx)
                    
                    # Skip if too small
                    if w * h < 1000:  # Minimum area in pixels
                        continue
                        
                    # Calculate aspect ratio (width/height)
                    aspect_ratio = float(w) / h
                    
                    # Most books have aspect ratios between 0.5 and 1.2
                    if 0.4 <= aspect_ratio <= 2.0:
                        # Convert to normalized coordinates
                        h_img, w_img = image.shape[:2]
                        norm_bbox = [
                            float(x) / w_img,
                            float(y) / h_img,
                            float(x + w) / w_img,
                            float(y + h) / h_img
                        ]
                        
                        # Skip if detection is likely a robot part (lower portion of image)
                        y_center = (norm_bbox[1] + norm_bbox[3]) / 2
                        if y_center > 0.6:  # Bottom 40% of image
                            print(f"Skipping potential book in robot zone (y_center = {y_center:.2f})")
                            continue
                            
                        # Calculate physical dimensions if possible
                        physical_width_m = None
                        physical_height_m = None
                        
                        # Get center of bounding box in pixels
                        center_x = int((x + w/2))
                        center_y = int((y + h/2))
                        
                        if self.sim and self.camera_params:
                            try:
                                sensor_obj = self.camera_params["sensor_obj"]
                                render_camera = sensor_obj.render_camera
                                
                                # Try to get world positions for corners to estimate size
                                corner1_x = x
                                corner1_y = y
                                corner2_x = x + w
                                corner2_y = y + h
                                
                                ray1 = render_camera.unproject(mn.Vector2i(corner1_x, corner1_y))
                                ray2 = render_camera.unproject(mn.Vector2i(corner2_x, corner2_y))
                                
                                raycast1 = self.sim.cast_ray(ray1)
                                raycast2 = self.sim.cast_ray(ray2)
                                
                                if raycast1.has_hits() and raycast2.has_hits():
                                    corner1_point = raycast1.hits[0].point
                                    corner2_point = raycast2.hits[0].point
                                    
                                    # Calculate physical width and height
                                    physical_width_m = np.linalg.norm(
                                        np.array([corner2_point[0], corner2_point[2]]) - 
                                        np.array([corner1_point[0], corner1_point[2]])
                                    )
                                    physical_height_m = np.linalg.norm(
                                        np.array([corner2_point[1]]) - 
                                        np.array([corner1_point[1]])
                                    )
                            except Exception as e:
                                print(f"Error calculating specialized detection size: {e}")
                        
                        # Skip if physical dimensions are unreasonable for a book
                        # Typical book dimensions: width 15-30cm, height 20-35cm
                        MAX_BOOK_WIDTH_M = 0.30   # 30cm max width
                        MAX_BOOK_HEIGHT_M = 0.35  # 35cm max height
                        
                        if physical_width_m and physical_height_m:
                            if physical_width_m > MAX_BOOK_WIDTH_M or physical_height_m > MAX_BOOK_HEIGHT_M:
                                print(f"Skipping oversized specialized detection ({physical_width_m:.2f}m × {physical_height_m:.2f}m)")
                                continue
                        
                        # Create a book entry
                        confidence = min(0.8, 0.4 + (1.0 / (abs(aspect_ratio - 0.7) + 0.1)))
                        book = {
                            "name": "book",
                            "description": f"A potential book detected with specialized detection (confidence: {confidence:.2f})",
                            "bbox": norm_bbox,
                            "confidence": confidence,
                            "detection_method": "specialized",
                            "might_be_book": True,
                            "detection_quality": "medium",
                            "original_class": "rectangle" 
                        }
                        potential_books.append(book)
            
            # Save visualization for debugging
            if potential_books:
                debug_img = image.copy()
                for book in potential_books:
                    bbox = book["bbox"]
                    x_min = int(bbox[0] * image.shape[1])
                    y_min = int(bbox[1] * image.shape[0])
                    x_max = int(bbox[2] * image.shape[1])
                    y_max = int(bbox[3] * image.shape[0])
                    
                    # Draw rectangle
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), color, 2)
                    cv2.putText(debug_img, f"Book {book['confidence']:.2f}", 
                               (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Save debug image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Removed specialized_books debug image saving to reduce clutter
                
                # Removed edge detection debug image saving
            
            return potential_books
            
        except Exception as e:
            print(f"Error in specialized book detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU (Intersection over Union) of two bounding boxes"""
        # Extract coordinates
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection area
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        if union_area <= 0:
            return 0.0
            
        return intersection_area / union_area

    def _extract_spatial_context(self, target_book: Dict, all_books: List[Dict]) -> Dict:
        """Extract spatial relationships between books"""
        context = {}
        try:
            if "world_position" in target_book:
                context["nearby_books"] = []
                for other_book in all_books:
                    if other_book != target_book and "world_position" in other_book:
                        relation = self.spatial_extractor.compute_relation(
                            target_book["world_position"],
                            other_book["world_position"]
                        )
                        context["nearby_books"].append({
                            "book_id": other_book.get("id"),
                            "relation": relation
                        })
        except Exception as e:
            print(f"Error extracting spatial context: {e}")
            
        return context
    
    def set_camera_params(self, sim, camera_uuid: str):
        """Set camera parameters for coordinate conversion."""
        self.sim = sim
        sensor = sim._sensors[camera_uuid]
        self.camera_params = {
            "uuid": camera_uuid,
            "resolution": sensor._spec.resolution,
            "sensor_obj": sensor._sensor_object
        }
            
    def get_book_position(self, book_bbox: List[float]) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """
        Convert book detection to 3D coordinates using raycasting.
        
        Args:
            book_bbox: Normalized bounding box coordinates [x_min, y_min, x_max, y_max]
            
        Returns:
            A tuple of (position, object_reference) where:
            - position: 3D world position as np.ndarray
            - object_reference: Book object reference if available, None otherwise
        """
        if not book_bbox or len(book_bbox) != 4 or self.camera_params is None or self.sim is None:
            print("Invalid input parameters for get_book_position")
            return None, None
        
        try:
            # Get center of bounding box
            height, width = self.camera_params["resolution"]
            center_x = int((book_bbox[0] + book_bbox[2]) / 2 * width)
            center_y = int((book_bbox[1] + book_bbox[3]) / 2 * height)
            
            print(f"Computing position for book at pixel ({center_x}, {center_y})")
            
            # Get sensor object and render camera
            sensor_obj = self.camera_params["sensor_obj"]
            render_camera = sensor_obj.render_camera
            
            # Create ray from pixel
            ray = render_camera.unproject(mn.Vector2i(center_x, center_y))
            
            # Cast ray to find world position
            raycast_results = self.sim.cast_ray(ray)
            
            # Default object reference to None
            obj_reference = None
            
            if raycast_results.has_hits():
                # Get the hit point
                hit_point = raycast_results.hits[0].point
                
                print(f"Raycast hit at position: {hit_point}")
                
                # Find the closest book in the scene to use as reference
                try:
                    # Get all objects from rigid object manager
                    rigid_obj_mgr = self.sim.get_rigid_object_manager()
                    all_handles = rigid_obj_mgr.get_object_handles()
                    
                    print(f"Available object handles: {all_handles}")
                    
                    # Look for any books in the scene
                    closest_book = None
                    min_distance = float('inf')
                    
                    for obj_handle in all_handles:
                        if "book" in obj_handle.lower():
                            try:
                                obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
                                obj_pos = obj.translation
                                
                                # Calculate distance to hit point
                                dist = np.linalg.norm(np.array([
                                    hit_point[0] - obj_pos[0],
                                    hit_point[1] - obj_pos[1],
                                    hit_point[2] - obj_pos[2]
                                ]))
                                
                                print(f"Book {obj_handle} is {dist:.3f}m from hit point")
                                
                                if dist < min_distance:
                                    min_distance = dist
                                    closest_book = obj
                            except Exception as obj_error:
                                print(f"Error accessing object {obj_handle}: {obj_error}")
                    
                    # Use the closest book if it's within a reasonable distance
                    if closest_book is not None:
                        if min_distance < 0.5:  # 50cm threshold
                            obj_reference = closest_book
                            print(f"Using closest book at distance {min_distance:.3f}m")
                        else:
                            print(f"Closest book is too far: {min_distance:.3f}m > 0.5m threshold")
                            
                except Exception as e:
                    print(f"Error finding closest book: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Always return a tuple of (position, object_reference)
                # The object_reference may be None if no matching object was found
                return hit_point, obj_reference
            else:
                print("Raycast had no hits, trying depth image fallback")
                
                # Try depth image if available
                try:
                    depth_uuid = "depth_" + self.camera_params["uuid"].split("_")[0]
                    if depth_uuid in self.sim._sensors:
                        depth_obs = self.sim.get_sensor_observations()[depth_uuid]
                        depth_value = depth_obs[center_y, center_x]
                        
                        if depth_value > 0:
                            # Calculate position using depth
                            world_pos = ray.origin + ray.direction * depth_value
                            print(f"Got position from depth: {world_pos}")
                            
                            # Find the closest book to this position
                            try:
                                rigid_obj_mgr = self.sim.get_rigid_object_manager()
                                all_handles = rigid_obj_mgr.get_object_handles()
                                
                                closest_book = None
                                min_distance = float('inf')
                                
                                for obj_handle in all_handles:
                                    if "book" in obj_handle.lower():
                                        obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
                                        obj_pos = obj.translation
                                        
                                        # Calculate distance to depth position
                                        dist = np.linalg.norm(np.array([
                                            world_pos[0] - obj_pos[0],
                                            world_pos[1] - obj_pos[1],
                                            world_pos[2] - obj_pos[2]
                                        ]))
                                        
                                        if dist < min_distance:
                                            min_distance = dist
                                            closest_book = obj
                                
                                if closest_book is not None and min_distance < 0.5:
                                    obj_reference = closest_book
                                    print(f"Found book near depth position at distance: {min_distance:.3f}m")
                            except Exception as e:
                                print(f"Error finding book near depth position: {e}")
                            
                            # No object information from depth, but maintaining consistent return format
                            return world_pos, obj_reference
                    else:
                        print(f"Depth sensor {depth_uuid} not found")
                except Exception as e:
                    print(f"Depth-based positioning failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
            # No position found at all
            print("No position could be determined for book")
            return None, None
            
        except Exception as e:
            print(f"Error calculating book position: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on the image."""
        # Make sure we're working with a copy to avoid modifying the original
        image = image.copy()
        
        # Keep original colors for visualization - no color correction
        # This ensures the visualization matches what the camera actually sees
            
        for obj in detections:
            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
                
            # Convert normalized coordinates to pixel values
            h, w = image.shape[:2]
            x_min = int(bbox[0] * w)
            y_min = int(bbox[1] * h)
            x_max = int(bbox[2] * w)
            y_max = int(bbox[3] * h)
            
            # Select color based on confidence
            conf = obj.get("confidence", 0.0)
            color = (0, int(255 * min(conf, 1.0)), 0)  # Green, brightness based on confidence
            
            # Draw bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Prepare label text
            label = f"{obj.get('name', 'unknown')} {conf:.2f}"
            
            # Draw label background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x_min, y_min - text_size[1] - 5), 
                         (x_min + text_size[0], y_min), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (x_min, y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                       
            # Draw world position if available
            if "world_position" in obj:
                pos = obj["world_position"]
                pos_text = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
                cv2.putText(image, pos_text, (x_min, y_max + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # For saved images we need to convert back to BGR for OpenCV
        image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image
        return image_to_save
        
    def draw_robot_coordinates(self, image: np.ndarray, robot_position, robot_rotation):
        """Draw robot position and orientation information on the image."""
        # Make a copy to avoid modifying the original
        img = image.copy()
        height, width = img.shape[:2]

        # Create info overlay
        padding = 10
        line_height = 25
        num_lines = 4
        overlay_height = num_lines * line_height + 2 * padding
        overlay = np.zeros((overlay_height, width, 3), dtype=np.uint8)

        # Draw semi-transparent background
        cv2.rectangle(overlay, (0, 0), (width, overlay_height), (30, 30, 30), -1)

        # Add position
        pos_text = f"Position: ({robot_position[0]:.2f}, {robot_position[1]:.2f}, {robot_position[2]:.2f})"
        cv2.putText(overlay, pos_text, (padding, padding + line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add yaw/pitch/roll (extract from quaternion)
        try:
            rot_matrix = robot_rotation.to_matrix()
            euler = mn.Math.euler_angles(rot_matrix)
            yaw, pitch, roll = euler.x, euler.y, euler.z
        except:
            # Simplified quaternion to Euler angles
            q0, q1, q2, q3 = robot_rotation.scalar, robot_rotation.vector.x, robot_rotation.vector.y, robot_rotation.vector.z
            roll = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1*q1 + q2*q2))
            pitch = np.arcsin(2*(q0*q2 - q3*q1))
            yaw = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2*q2 + q3*q3))

        rot_text = f"Rotation (rad): Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}"
        cv2.putText(overlay, rot_text, (padding, padding + 2*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add time
        time_text = f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(overlay, time_text, (width - 300, padding + 3*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Combine overlay with image
        result = np.vstack([overlay, img])

        return result