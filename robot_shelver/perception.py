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
    def __init__(self, model_name="llava-phi3", api_url="http://localhost:11434/api/chat"):
        """Initialize the image system with Ollama API integration."""
        self.model_name = model_name
        self.api_url = api_url
        
        # Create logs directory in robot_shelver folder
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_dir = os.path.join(current_file_dir, "image_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"Perception logs will be saved to: {self.log_dir}")
        
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
        print(f"Camera parameters set for {camera_uuid}: {self.camera_params['resolution']}")
    
    def encode_image(self, cv2_image):
        """Convert a CV2 image to a base64 encoded string."""
        success, encoded_image = cv2.imencode('.png', cv2_image)
        if not success:
            raise ValueError("Could not encode image")
        
        base64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
        return base64_image
    
    def process_image(self, image, prompt="What do you see in this image? Describe it in detail."):
        """Process an image with the VLM and return the description."""
        base64_image = self.encode_image(image)
        
        # Prepare API request for Ollama
        payload = {
            "model": self.model_name,
            "messages": [
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
            print(f"Error processing image with VLM: {e}")
            return f"Error: {str(e)}"
    
    def log_image(self, image, description):
        """Log the image and its description for future use."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the image
        img_path = os.path.join(self.log_dir, f"image_{timestamp}.png")
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
        """Detect objects in the image and return their bounding boxes."""
        if prompt is None:
            prompt = """Identify the main objects in this image. 
            For each object, provide:
            1. Object name
            2. Bounding box coordinates in format [x_min, y_min, x_max, y_max] as float values between 0 and 1
            3. A brief description of the object
            
            Format your response as a JSON array of objects with fields: 
            "name", "bbox", "description"
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
                
            print("Could not extract JSON from VLM response")
            print(f"Raw response: {description}")
            return []
        except Exception as e:
            print(f"Error parsing object detection results: {e}")
            print(f"Raw response: {description}")
            return []
    
    def visualize_detections(self, image, objects):
        """Draw bounding boxes on the image for detected objects with world coordinates."""
        vis_image = image.copy()
        height, width = image.shape[:2]
        
        for obj in objects:
            if "bbox" in obj and len(obj["bbox"]) == 4:
                # Convert normalized coordinates to pixel values
                x_min = int(obj["bbox"][0] * width)
                y_min = int(obj["bbox"][1] * height)
                x_max = int(obj["bbox"][2] * width)
                y_max = int(obj["bbox"][3] * height)
                
                # Calculate center of bounding box for world coordinate estimation
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Add label
                label = obj.get("name", "Object")
                cv2.putText(vis_image, label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Try to calculate world coordinates if camera parameters are available
                if self.camera_params is not None:
                    world_pos = self.pixel_to_world_coordinates(center_x, center_y)
                    if world_pos is not None:
                        # Add world coordinate info
                        coord_text = f"Pos: ({world_pos[0]:.2f}, {world_pos[1]:.2f}, {world_pos[2]:.2f})"
                        cv2.putText(vis_image, coord_text, (x_min, y_max + 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        return vis_image
    
    def analyze_scene(self, image):
        """Analyze the scene to identify objects and their spatial relationships."""
        prompt = """Provide a structured analysis of this scene with the following information in JSON format:
        1. "scene_type": The type of environment (e.g., kitchen, office, living room)
        2. "objects": A list of visible objects, each with:
           - "name": Object name
           - "position": Relative position description (e.g., "center", "top-left", "on the table")
           - "attributes": List of attributes (color, size, material, etc.)
        3. "spatial_relationships": List of relationships between objects (e.g., "cup on table", "chair next to desk")
        4. "navigation_cues": Important landmarks or paths visible in the scene
        
        Format as a single JSON object."""
        
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
            print("Camera parameters not set. Cannot convert coordinates.")
            return None
        
        try:
            # Get the sensor object and its render camera
            sensor_obj = self.camera_params["sensor_obj"]
            render_camera = sensor_obj.render_camera  # Access the render_camera property
            
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
                # Alternative: try to use depth image if available
                try:
                    # Check if we have a depth sensor with matching name pattern
                    depth_uuid = "depth_" + self.camera_params["uuid"].split("_")[0]
                    if depth_uuid in self.sim._sensors:
                        depth_obs = self.sim.get_sensor_observations()[depth_uuid]
                        depth_value = depth_obs[pixel_y, pixel_x]
                        
                        if depth_value > 0:
                            # Calculate world position using depth
                            world_pos = ray.origin + ray.direction * depth_value
                            return world_pos
                except Exception as e:
                    print(f"Error using depth for coordinate conversion: {e}")
                    
                return None  # No hit found
                
        except Exception as e:
            print(f"Error converting pixel to world coordinates: {e}")
            return None
    
    def draw_robot_coordinates(self, image, robot_position, robot_rotation):
        """Draw robot coordinates on the image."""
        # Make a copy to avoid modifying the original
        img = image.copy()
        height, width = img.shape[:2]

        # Ensure img has 3 channels (RGB)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

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
        # Convert quaternion to Euler angles
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

        # Add quaternion for reference
        quat_text = f"Quaternion: [{robot_rotation.scalar:.2f}, {robot_rotation.vector.x:.2f}, {robot_rotation.vector.y:.2f}, {robot_rotation.vector.z:.2f}]"
        cv2.putText(overlay, quat_text, (padding, padding + 3*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Add current time
        time_text = f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(overlay, time_text, (width - 300, padding + 3*line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Make sure overlay and image have the same number of channels
        if overlay.shape[2] != img.shape[2]:
            if img.shape[2] == 4:  # If image has alpha channel
                # Add alpha channel to overlay
                alpha = np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=np.uint8) * 255
                overlay = np.concatenate([overlay, alpha], axis=2)
            else:
                # Convert image to 3 channels
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Combine overlay with image
        result = np.vstack([overlay, img])

        return result