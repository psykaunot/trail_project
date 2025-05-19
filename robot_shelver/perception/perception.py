import base64
import requests
import json
import cv2
import numpy as np
import os
from datetime import datetime
import re
import habitat_sim
import magnum as mn

class Perception:
    def __init__(self, model_name="yaniserrol/vlm-r1:latest", api_url="http://localhost:11434/api/chat"):
        """Initialize perception system with vision-language model integration."""
        self.model_name = model_name
        print(f"Using VLM model: {self.model_name}")
        self.api_url = api_url
        
        # Create logs directory for perception records
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(current_file_dir, "perception_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Camera parameters (will be updated when simulator is available)
        self.camera_params = None
        self.sim = None
    
    def set_camera_params(self, sim, camera_uuid="robot_rgb"):
        """Set camera parameters from the simulator for coordinate conversion."""
        self.sim = sim
        sensor = sim._sensors[camera_uuid]
        self.camera_params = {
            "uuid": camera_uuid,
            "resolution": sensor._spec.resolution,
            "sensor_obj": sensor._sensor_object
        }
    
    def encode_image(self, cv2_image):
        """Convert a CV2 image to a base64 encoded string."""
        success, encoded_image = cv2.imencode('.png', cv2_image)
        if not success:
            raise ValueError("Could not encode image")
        
        base64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        return base64_image
    
    def process_image(self, image, prompt="What do you see in this image? Describe it in detail."):
        """Process image with VLM and return description."""
        base64_image = self.encode_image(image)
        
        # Prepare API request for Ollama
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a perception system for a robot. Provide detailed, accurate descriptions of what you see."
                },
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image]
                }
            ],
            "stream": False
        }
        
        # Make API request
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the description from the response
            description = result.get("message", {}).get("content", "")
            
            # Log the result
            self.log_image(image, description)
            
            return description
        except Exception as e:
            print(f"Error processing image: {e}")
            return f"Error: {str(e)}"
        
    def _preprocess_image(self, image, save_debug=False):
        """Apply color correction to image."""
        if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
            return image
            
        # Apply color correction
        r, g, b = cv2.split(image)
        
        # Stronger correction factors
        gain_r = 1.5   # Increase red
        gain_g = 1.3   # Increase green
        gain_b = 0.7   # Decrease blue
        
        r = np.clip(r * gain_r, 0, 255).astype(np.uint8)
        g = np.clip(g * gain_g, 0, 255).astype(np.uint8)
        b = np.clip(b * gain_b, 0, 255).astype(np.uint8)
        
        processed = cv2.merge([r, g, b])
        
        if save_debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            debug_dir = os.path.join(self.log_dir, "color_correction")
            os.makedirs(debug_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(debug_dir, f"original_{timestamp}.png"), image)
            cv2.imwrite(os.path.join(debug_dir, f"corrected_{timestamp}.png"), processed)
            
        return processed
    
    def log_image(self, image, description):
        """Log the image and its description for future analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the image
        img_path = os.path.join(self.log_dir, f"image_{timestamp}.png")
        image_to_save = image.copy()
        if image_to_save.shape[2] == 4: # Check if it has an alpha channel (RGBA)
            image_to_save = cv2.cvtColor(image_to_save, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(img_path, image)
        
        # Save the description
        text_path = os.path.join(self.log_dir, f"image_{timestamp}.txt")
        with open(text_path, "w") as f:
            f.write(description)
        
        # Save metadata
        meta_path = os.path.join(self.log_dir, f"image_{timestamp}.json")
        with open(meta_path, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "model": self.model_name,
                "image_path": os.path.basename(img_path),
                "description_path": os.path.basename(text_path)
            }, f, indent=2)
    
    def detect_objects(self, image, prompt=None):
        """Detect objects in the image using the vision-language model."""
        if prompt is None:
            prompt = """Identify all objects in this image and provide their exact locations.
            
            For each object, provide:
            1. Object name
            2. Bounding box coordinates in format [x_min, y_min, x_max, y_max] as float values between 0 and 1
            3. A brief description of the object
            
            Format your response as a JSON array with fields: 
            "name", "bbox", "description"
            
            Be precise and thorough in your analysis.
            """
        
        description = self.process_image(image, prompt)
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response (might be embedded in text)
            json_match = re.search(r'\[\s*\{.*\}\s*\]', description, re.DOTALL)
            if json_match:
                potential_json = json_match.group(0)
                objects = json.loads(potential_json)
                return objects
            else:
                # If no JSON array is found, try to find a single JSON object
                json_match = re.search(r'\{\s*".*"\s*:.*\}', description, re.DOTALL)
                if json_match:
                    potential_json = json_match.group(0)
                    objects = json.loads(potential_json)
                    return [objects]
            
            return []
        except Exception as e:
            print(f"Error parsing object detection results: {e}")
            return []
    
    def visualize_detections(self, image, objects):
        """Draw bounding boxes and information on an image for detected objects."""
        vis_image = image.copy()
        height, width = image.shape[:2]

        for i, obj in enumerate(objects):
            bbox = obj.get("bbox", [])

            # Skip if bbox is empty or invalid
            if not bbox or len(bbox) != 4:
                # Skip invalid bounding boxes
                continue

            # Ensure values are valid
            try:
                x_min = max(0, min(int(bbox[0] * width), width-1))
                y_min = max(0, min(int(bbox[1] * height), height-1))
                x_max = max(0, min(int(bbox[2] * width), width-1))
                y_max = max(0, min(int(bbox[3] * height), height-1))

                # Skip if box has no area
                if x_max <= x_min or y_max <= y_min:
                    continue

                # Skip very large bboxes (likely the robot)
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                if bbox_width > width * 0.5 or bbox_height > height * 0.5:
                    # Skip large bounding boxes silently
                    continue

            except (ValueError, TypeError) as e:
                # Skip invalid bounding boxes silently
                continue

            # Draw bounding box
            color = (0, 255, 0)  # Green color for visibility
            cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color, 2)

            # Add label
            label = obj.get("name", "Book")
            cv2.putText(vis_image, label, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Calculate center of bounding box for world coordinate estimation
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2

            # Try to calculate world coordinates if camera parameters are available
            if self.camera_params is not None:
                world_pos = self.pixel_to_world_coordinates(center_x, center_y)
                if world_pos is not None:
                    # Add world coordinate info
                    coord_text = f"Pos: ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})"
                    cv2.putText(vis_image, coord_text, (x_min, y_max + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return vis_image
    
    def analyze_scene(self, image):
        """Analyze the scene to identify objects, relationships, and spatial structure."""
        prompt = """Provide a structured analysis of this scene with the following information in JSON format:
        1. "scene_type": The type of environment (e.g., kitchen, office, living room)
        2. "objects": A list of visible objects, each with:
           - "name": Object name
           - "position": Relative position description (e.g., "center", "top-left", "on the table")
           - "attributes": List of attributes (color, size, material, etc.)
        3. "spatial_relationships": List of relationships between objects
        4. "navigation_cues": Important landmarks or paths visible in the scene
        
        Format as a single JSON object.
        """
        
        result = self.process_image(image, prompt)
        
        try:
            # Try to extract JSON from the text response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                scene_data = json.loads(json_match.group(0))
                return scene_data
            return {"error": "Could not parse scene analysis", "raw_text": result}
        except Exception as e:
            return {"error": str(e), "raw_text": result}
    
    def pixel_to_world_coordinates(self, pixel_x, pixel_y):
        """Convert pixel coordinates to world coordinates using raycasting."""
        if self.camera_params is None or self.sim is None:
            return None
        
        try:
            # Get the sensor object and its render camera
            sensor_obj = self.camera_params["sensor_obj"]
            render_camera = sensor_obj.render_camera
            
            # Create a ray using unproject
            ray = render_camera.unproject(
                mn.Vector2i(int(pixel_x), int(pixel_y))
            )
            
            # Cast ray into scene
            raycast_results = self.sim.cast_ray(ray)
            
            if raycast_results.has_hits():
                # Return the world position of the first hit
                hit_point = raycast_results.hits[0].point
                return hit_point
            else:
                # Try to use depth image if available
                try:
                    depth_uuid = "depth_" + self.camera_params["uuid"].split("_")[0]
                    if depth_uuid in self.sim._sensors:
                        depth_obs = self.sim.get_sensor_observations()[depth_uuid]
                        depth_value = depth_obs[pixel_y, pixel_x]
                        
                        if depth_value > 0:
                            # Calculate world position using depth
                            world_pos = ray.origin + ray.direction * depth_value
                            return world_pos
                except:
                    pass
                    
                return None  # No hit found
                
        except Exception as e:
            print(f"Error converting pixel to world coordinates: {e}")
            return None
    
    def draw_robot_coordinates(self, image, robot_position, robot_rotation):
        """Add robot position and orientation information to the image."""
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
    
    def find_books_in_image(self, image):
        """Find books in the given image with enhanced color correction and robust detection."""
        if image is None or image.size == 0:
            print("Warning: Invalid image provided to find_books_in_image")
            return []

        try:
            # Apply color correction to fix blueish tint
            processed_image = self._preprocess_image(image, save_debug=False)

            # Enhanced prompt with more specific guidance
            prompt = """
            Carefully inspect this image and find ALL books. Books may be:
            - On the floor or any horizontal surface
            - On shelves or vertical surfaces
            - Green, brown, or other colors typical for books
            - Different sizes (small to large)
            - Partially visible or at odd angles
            - May have visible pages, spine, or cover

            IMPORTANT: IGNORE the robot arm/gripper if visible.

            For EACH BOOK you find, provide:
            1. A tight bounding box: [x_min, y_min, x_max, y_max] (all values 0-1)
            2. A brief description of the book
            3. Color if visible
            4. Location (e.g., "on floor," "on shelf")

            Return results as JSON array: [{"name": "book", "bbox": [0.3, 0.1, 0.5, 0.3], "description": "Green textbook on floor", "color": "green"}]

            If NO books are visible, return an empty array: []
            """

            # First attempt with VLM-based object detection
            print("Attempting primary book detection...")
            books = self.detect_objects(processed_image, prompt)

            # Check for valid responses
            if not books:
                print("No books found in primary detection")

            # Verify and validate detected books
            valid_books = []
            for i, book in enumerate(books):
                bbox = book.get("bbox")
                if not bbox or len(bbox) != 4:
                    print(f"Warning: Book {i} has invalid bbox: {bbox}")
                    continue

                # Validate bbox coordinates are within range
                if all(0 <= coord <= 1 for coord in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    valid_books.append(book)
                else:
                    print(f"Warning: Book {i} has out-of-range bbox: {bbox}")

            # If we found valid books, log and visualize them
            if valid_books:
                print(f"Found {len(valid_books)} valid books")

                # Visualize the detections
                annotated_image = self.visualize_detections(processed_image, valid_books)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(self.log_dir, f"book_detection_{timestamp}.png")
                cv2.imwrite(img_path, annotated_image)
                print(f"Saved book detection image: {img_path}")
                return valid_books

            # If no valid books found, try fallback detection with a simpler prompt
            print("No valid books found, trying fallback detection...")
            fallback_prompt = """
            Focus ONLY on finding books in this image.
            A book typically has a rectangular shape with visible spine or pages.

            For each book, provide the bounding box coordinates as [x_min, y_min, x_max, y_max] 
            where all values are between 0 and 1.

            Return results as: [{"name": "book", "bbox": [0.3, 0.1, 0.5, 0.3]}]

            If no books are visible, return []
            """

            fallback_books = self.detect_objects(processed_image, fallback_prompt)
            # Check fallback results
            if fallback_books:
                print(f"Fallback detection found {len(fallback_books)} potential books")

            # Validate fallback results
            valid_fallback = []
            for i, book in enumerate(fallback_books):
                bbox = book.get("bbox")
                if bbox and len(bbox) == 4 and all(0 <= coord <= 1 for coord in bbox) and bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    valid_fallback.append(book)

            if valid_fallback:
                print(f"Fallback detection found {len(valid_fallback)} books")
                annotated_image = self.visualize_detections(processed_image, valid_fallback)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(self.log_dir, f"book_fallback_{timestamp}.png")
                cv2.imwrite(img_path, annotated_image)
                return valid_fallback

            # No books found in the image
            print("No books found in image")

            return []

        except Exception as e:
            print(f"Error in book detection: {e}")
            return []
    
    def get_book_position(self, book_bbox):
        """Convert book detection to 3D coordinates."""
        if not book_bbox or len(book_bbox) != 4:
            return None
        
        # Get center of book bounding box
        height, width = self.camera_params["resolution"]
        center_x = int((book_bbox[0] + book_bbox[2]) / 2 * width)
        center_y = int((book_bbox[1] + book_bbox[3]) / 2 * height)
        
        # Use depth information to get world position
        world_pos = self.pixel_to_world_coordinates(center_x, center_y)
        return world_pos