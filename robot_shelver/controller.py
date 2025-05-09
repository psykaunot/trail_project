#!/usr/bin/env python3
import os
import time
import threading
import queue
import json
import random
import numpy as np
import magnum as mn
import requests
import traceback
from enum import Enum
import habitat_sim


class ControlMode(Enum):
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    NAVIGATE = "navigate"
    GRASP = "grasp"
    PLACE = "place"
    EXPLORE = "explore"
    EXECUTING = "executing"

class Controller:
    """LLM-powered robot controller for autonomous operation."""
    
    def __init__(self, llm_model="qwen3:8b", api_url="http://localhost:11434/api/chat", debug=True):
        """Initialize the controller with the specified LLM model."""
        self.llm_model = llm_model
        self.api_url = api_url
        self.mode = ControlMode.MANUAL
        self.target_object = None
        self.target_position = None
        self.task_description = None
        self.progress = "Not started"
        self.debug = debug
        
        # Queues for communication between threads
        self.command_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
        # Robot state information
        self.robot_position = None
        self.robot_rotation = None
        self.locobot = None
        self.motor_ids = None
        self.motor_settings = None
        self.dof_map = None
        self.drive_speed = 1.0
        self.turn_speed = 0.5
        self.arm_speed = 0.7
        self.grip_speed = 0.7
        
        # Perception information
        self.current_image = None
        self.detected_objects = []
        self.scene_analysis = None
        
        # Thread control
        self.running = False
        self.thread = None
        
        print(f"Controller initialized with LLM model: {llm_model}")
    
    def set_robot_controls(self, locobot, motor_ids, motor_settings, dof_map, 
                          drive_speed=1.0, turn_speed=0.5, arm_speed=0.7, grip_speed=0.7, sim=None):
        """Set the robot control interfaces for direct motor control."""
        self.locobot = locobot
        self.motor_ids = motor_ids
        self.motor_settings = motor_settings
        self.dof_map = dof_map
        self.sim = sim
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed
        self.arm_speed = arm_speed
        self.grip_speed = grip_speed
        
        if self.debug:
            print(f"Robot controls set: drive_speed={drive_speed}, turn_speed={turn_speed}")
    
    def start(self):
        """Start the controller thread."""
        if self.thread is not None and self.thread.is_alive():
            print("Controller thread is already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"Started controller thread with LLM model: {self.llm_model}")
        return True
    
    def stop(self):
        """Stop the controller thread."""
        self.running = False
        if self.thread is not None:
            try:
                self.thread.join(timeout=2.0)
            except Exception as e:
                print(f"Error stopping controller thread: {e}")
            self.thread = None
        print("Stopped controller thread")
    
    def update_robot_state(self, position, rotation):
        """Update the current robot position and orientation."""
        self.robot_position = position
        self.robot_rotation = rotation
    
    def update_perception(self, image=None, objects=None, scene_analysis=None):
        """Update perception data from vision systems."""
        if image is not None:
            self.current_image = image
        if objects is not None:
            self.detected_objects = objects
            if self.debug:
                print(f"Updated detected objects: {len(objects)} objects")
        if scene_analysis is not None:
            self.scene_analysis = scene_analysis
    
    def get_status(self):
        """Get the current controller status."""
        return {
            "mode": self.mode.value,
            "target_object": self.target_object,
            "task_description": self.task_description,
            "progress": self.progress,
            "position": str(self.robot_position) if self.robot_position else "Unknown",
        }
    
    def navigate_to_object(self, object_name):
        """Start navigation to reach a named object."""
        if self.debug:
            print(f"Starting navigation to object: {object_name}")
        
        self.mode = ControlMode.NAVIGATE
        self.target_object = object_name
        self.progress = "Planning path"
        self.task_description = f"Navigate to {object_name}"
        
        # Find object in detected objects list
        target_obj = None
        for obj in self.detected_objects:
            if obj.get("name", "").lower() == object_name.lower():
                target_obj = obj
                break
        
        if target_obj and "bbox" in target_obj:
            # Start a separate thread for navigation planning
            threading.Thread(
                target=self._execute_navigation,
                args=(target_obj,),
                daemon=True
            ).start()
            return True
        else:
            self.progress = "Failed - Object not found"
            print(f"Navigation failed: Object '{object_name}' not found")
            return False
    
    def grasp_object(self, object_name):
        """Start grasping sequence for a named object."""
        if self.debug:
            print(f"Starting grasp sequence for object: {object_name}")
        
        self.mode = ControlMode.GRASP
        self.target_object = object_name
        self.progress = "Planning grasp"
        self.task_description = f"Grasp {object_name}"
        
        # Find object in detected objects list
        target_obj = None
        for obj in self.detected_objects:
            if obj.get("name", "").lower() == object_name.lower():
                target_obj = obj
                break
        
        if target_obj:
            # Start a separate thread for grasp planning
            threading.Thread(
                target=self._execute_grasp,
                args=(target_obj,),
                daemon=True
            ).start()
            return True
        else:
            self.progress = "Failed - Object not found"
            print(f"Grasp failed: Object '{object_name}' not found")
            return False
    
    def start_exploration(self):
        """Start autonomous exploration of the environment."""
        if self.debug:
            print("Starting autonomous exploration")
        
        self.mode = ControlMode.EXPLORE
        self.target_object = None
        self.progress = "Starting exploration"
        self.task_description = "Autonomous exploration"
        
        # Start exploration in a separate thread
        threading.Thread(
            target=self._execute_exploration,
            daemon=True
        ).start()
        
        return True
    
    def stop_autonomous_control(self):
        """Stop any autonomous behavior and return to manual control."""
        if self.debug:
            print("Stopping autonomous control, returning to manual mode")
        
        self.mode = ControlMode.MANUAL
        self.target_object = None
        self.progress = "Stopped"
        self.task_description = None
        
        # Put a stop command in the queue
        try:
            self.command_queue.put(0)  # 0 = stop command
        except Exception as e:
            print(f"Error adding stop command to queue: {e}")
        
        return True
    
    def _run(self):
        """Main controller thread loop."""
        print("Controller thread started")
        try:
            while self.running:
                # Check if we're in autonomous mode and need to plan next actions
                if self.mode == ControlMode.EXPLORE:
                    # Add exploration planning logic here
                    pass
                
                # Sleep to avoid consuming too much CPU
                time.sleep(0.1)
        except Exception as e:
            print(f"Error in controller thread: {e}")
            traceback.print_exc()
        print("Controller thread stopped")
    
    def _execute_navigation(self, target_obj):
        """Execute navigation to a target object."""
        try:
            self.mode = ControlMode.EXECUTING
            self.progress = "Moving to object"
            
            # Extract target position from object bbox (center-bottom point)
            if "bbox" in target_obj:
                bbox = target_obj["bbox"]
                # Use depth info if available to get world position
                # For now, just navigate forward as example
                for i in range(3):
                    self.progress = f"Moving forward ({i+1}/3)"
                    self.command_queue.put(65362)  # Forward key
                    time.sleep(0.5)
                
                # Now turn to face the object
                self.progress = "Turning to face object"
                self.command_queue.put(65361)  # Left key
                time.sleep(0.5)
            
            self.progress = "Reached target"
            time.sleep(1.0)
            self.mode = ControlMode.MANUAL
            self.progress = "Navigation complete"
        except Exception as e:
            print(f"Error during navigation: {e}")
            traceback.print_exc()
            self.progress = f"Navigation failed: {str(e)}"
            self.mode = ControlMode.MANUAL
    
    def _execute_grasp(self, target_obj):
        """Execute grasping sequence for a target object."""
        try:
            self.mode = ControlMode.EXECUTING
            self.progress = "Positioning arm"
            
            # Move arm to pre-grasp position
            for i in range(2):
                self.command_queue.put(ord('i'))  # Shoulder up
                time.sleep(0.5)  # Longer pause between commands
            
            for i in range(2):
                self.command_queue.put(ord('o'))  # Elbow up
                time.sleep(0.5)
            
            # Open gripper
            self.progress = "Opening gripper"
            self.command_queue.put(ord('g'))
            time.sleep(1.0)  # Longer pause for gripper to complete
            
            # Move arm down to object
            self.progress = "Approaching object"
            for i in range(3):
                self.command_queue.put(ord('k'))  # Shoulder down
                time.sleep(0.5)
            
            # Close gripper
            self.progress = "Closing gripper"
            self.command_queue.put(ord('h'))
            time.sleep(1.0)  # Longer pause for gripper to complete
            
            # Lift object
            self.progress = "Lifting object"
            for i in range(2):
                self.command_queue.put(ord('i'))  # Shoulder up
                time.sleep(0.5)
            
            self.mode = ControlMode.MANUAL
            self.progress = "Grasp complete"
        except Exception as e:
            print(f"Error during grasp execution: {e}")
            traceback.print_exc()
            self.progress = f"Grasp failed: {str(e)}"
            self.mode = ControlMode.MANUAL
    

    def _execute_exploration(self):
        """Execute exploration with visible movements."""
        try:
            self.mode = ControlMode.EXECUTING
            self.progress = "Exploring environment"
    
            import environment
            environment.set_exploration_active(True)
    
            # Get initial position
            initial_pos = self.locobot.translation
            current_pos = [initial_pos[0], initial_pos[1], initial_pos[2]]
    
            for i in range(10):  # 10 movement steps
                # Generate small random movement
                dx = random.uniform(-0.15, 0.15)
                dz = random.uniform(-0.15, 0.15)
    
                # Calculate new position
                new_pos = [
                    current_pos[0] + dx,
                    0.0,  # Keep y at floor level
                    current_pos[2] + dz
                ]
    
                # Apply position directly
                state = self.locobot.rigid_state
                state.translation = mn.Vector3(new_pos[0], new_pos[1], new_pos[2])
                self.locobot.rigid_state = state
    
                # Zero all velocities
                self.locobot.root_linear_velocity = mn.Vector3(0, 0, 0)
                self.locobot.root_angular_velocity = mn.Vector3(0, 0, 0)
    
                # CRITICAL: Update reference position in environment to prevent snapping back
                environment._reference_position = mn.Vector3(new_pos[0], new_pos[1], new_pos[2])
    
                # Update current position for next step
                current_pos = new_pos
    
                # Wait between moves
                time.sleep(0.5)
    
            self.progress = "Exploration complete"
        finally:
            environment.set_exploration_active(False)

    def _plan_exploration_action(self):
        """Plan the next exploration action based on current state."""
        try:
            # If we have detected objects, try to navigate to one
            if self.detected_objects and random.random() < 0.3:
                obj = random.choice(self.detected_objects)
                self.target_object = obj.get("name", "unknown object")
                return 65362  # Forward key - move toward objects
            
            # Otherwise, choose a random movement
            action_choices = [
                {"type": "FORWARD", "value": self.drive_speed * 0.8},  # Move forward (slower for safety)
                {"type": "TURN_LEFT", "value": self.turn_speed * 0.8},  # Turn left (slower for safety)
                {"type": "TURN_RIGHT", "value": self.turn_speed * 0.8},  # Turn right (slower for safety)
                65362,  # Forward key
                65361,  # Left key
                65363,  # Right key
            ]
            action = random.choice(action_choices)
            return action
        except Exception as e:
            print(f"Error planning exploration action: {e}")
            return None
    
    def _query_llm(self, prompt):
        """Query the LLM for planning or decision making with reduced verbosity."""
        try:
            # Prepare the API request payload
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": "You are a robot control assistant."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            # Make the API request
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # Extract the response content
            response_text = result.get("message", {}).get("content", "")
            
            # Clean up verbose responses
            import re
            # Remove "think" sections if present
            cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            
            # If response is too long, extract just the decision
            if len(cleaned_response) > 100:
                # For camera mission, extract PAN and TILT values
                pan_match = re.search(r"PAN:\s*([-+]?\d*\.\d+|\d+)", cleaned_response)
                tilt_match = re.search(r"TILT:\s*([-+]?\d*\.\d+|\d+)", cleaned_response)
                if pan_match and tilt_match:
                    return f"PAN: {pan_match.group(1)} TILT: {tilt_match.group(1)}"
                
                # For other missions, extract just the action
                for action in ["TURN_LEFT", "TURN_RIGHT", "MOVE_FORWARD", "APPROACH_BOOK", "GRAB_BOOK"]:
                    if action in cleaned_response:
                        return action
            
            return cleaned_response
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return "Error: Could not get response from language model"
        
    def _scan_room_for_books(self):
        """Scan the room for books using the camera."""
        self.progress = "Scanning room for books"
        books_found = []

        # Initialize camera controller if needed
        if not hasattr(self, 'camera_controller') and self.locobot:
            from camera_controller import CameraController
            self.camera_controller = CameraController(
                self.locobot, self.motor_ids, self.motor_settings, self.dof_map
            )

        # Scan positions (left, center, right)
        scan_positions = [
            (-1.0, -0.2),  # Left side
            (0.0, -0.2),   # Center 
            (1.0, -0.2),   # Right side
        ]

        for pan, tilt in scan_positions:
            self.progress = f"Scanning position: pan={pan:.1f}, tilt={tilt:.1f}"

            # Move camera to scanning position
            if hasattr(self, 'camera_controller'):
                self.camera_controller.move_camera(
                    pan_delta=pan - self.camera_controller.pan_target, 
                    tilt_delta=tilt - self.camera_controller.tilt_target
                )

            # Pause to let camera settle
            time.sleep(1.0)

            # Get current image
            if self.sim:
                obs = self.sim.get_sensor_observations()
                front_img = obs['robot_rgb'].copy()

                # Use perception module to find books
                if hasattr(self, 'perception'):
                    books = self.perception.find_books_in_image(front_img)
                    books_found.extend(books)
                    print(f"Found {len(books)} books at position {pan:.1f}, {tilt:.1f}")
                else:
                    from perception import Perception
                    self.perception = Perception()
                    self.perception.set_camera_params(self.sim, "robot_rgb")

        print(f"Total books found: {len(books_found)}")
        return books_found

    def _approach_book(self, book):
        """Navigate to the detected book."""
        if 'bbox' not in book:
            print("Book detection missing bounding box information")
            return False

        # Get 3D world position from the 2D bounding box
        book_pos = self.perception.get_book_position(book['bbox'])
        if book_pos is None:
            print("Could not determine book position in 3D space")
            return False

        print(f"Book position: {book_pos}")
        self.progress = f"Moving to book at {book_pos}"

        # Use NavMesh navigator to move to the book (stopping short for grasping)
        from navmesh_navigator import NavMeshNavigator
        navigator = NavMeshNavigator(
            self.sim, self.locobot, self.motor_ids, self.motor_settings, self.dof_map
        )

        # Calculate approach position (slightly back from book for better grasping)
        approach_distance = 0.5  # meters
        direction = (book_pos - self.locobot.translation).normalized()
        approach_pos = book_pos - direction * approach_distance
        approach_pos.y = 0.0  # Ensure we're at floor level

        # Navigate to approach position
        success = navigator.navigate_to(approach_pos)
        return success

    def _grab_book(self):
        """Execute the book grasping sequence."""
        self.progress = "Grasping book"

        # Create pick & place task
        from pick_place_demo import PickAndPlaceTask
        pick_task = PickAndPlaceTask(
            self.sim, self.locobot, self.motor_ids, self.motor_settings, self.dof_map
        )

        # Execute the pick sequence with camera guidance
        pick_task.camera_guided_grasp()
        self.progress = "Book grabbed successfully"
        return True

    def find_book_mission(self):
        """Start the book finding mission."""
        print("MISSION STARTED: Looking for books...")
        self.mode = ControlMode.EXECUTING
        self.task_description = "Find and pick book"

        # Call our scanning function
        books = self._scan_room_for_books()
        if not books:
            print("No books found!")
            self.mode = ControlMode.MANUAL
            self.progress = "Mission failed: No books found"
            return False

        # Move to the book
        self.progress = "Moving to book"
        target_book = books[0]  # Pick first book found
        success = self._approach_book(target_book)

        # Pick up the book
        if success:
            success = self._grab_book()
            if success:
                self.progress = "Mission complete: Book found and grabbed"
            else:
                self.progress = "Mission failed: Could not grab book"
        else:
            self.progress = "Mission failed: Could not reach book"

        self.mode = ControlMode.MANUAL
        return success
    

    def llm_guided_book_mission(self):
        """Use LLM to guide a step-by-step book finding mission"""
        print("🤖 Starting LLM-guided book mission")
        self.mode = ControlMode.EXECUTING
        self.task_description = "Find and grab book using LLM"
        self.progress = "Analyzing scene"

        # Get current view
        obs = self.sim.get_sensor_observations()
        front_img = obs['robot_rgb'].copy()

        # Run object detection first
        from perception import Perception
        perception = Perception()
        detected_objects = perception.detect_objects(front_img)
        books = [obj for obj in detected_objects if obj.get('name', '').lower() == 'book']

        # Prepare context for LLM with detection results
        book_context = ""
        if books:
            book_context = f"I detected {len(books)} books in the current view."
        else:
            book_context = "I don't see any books in the current view."

        # Ask LLM for next step
        prompt = f"""You are controlling a robot to find and grab a book. {book_context}

    Current robot position: {self.locobot.translation}
    Current view: Front camera

    What should the robot do next? Choose ONE action:
    1. TURN_LEFT - Turn the entire robot left to search
    2. TURN_RIGHT - Turn the entire robot right to search
    3. MOVE_FORWARD - Move robot forward
    4. APPROACH_BOOK - Move closer to visible book
    5. GRAB_BOOK - Position arm and grab visible book

    Reply with ONLY the action name and a brief explanation."""

        # Send to LLM
        response = self._query_llm(prompt)
        print(f"LLM guidance: {response}")

        # Extract action based on LLM response
        action = None
        if "TURN_LEFT" in response:
            self._execute_turn_left()
            action = "Turning left to search"
        elif "TURN_RIGHT" in response:
            self._execute_turn_right()
            action = "Turning right to search"
        elif "MOVE_FORWARD" in response:
            self._execute_move_forward()
            action = "Moving forward"
        elif "APPROACH_BOOK" in response and books:
            self._execute_approach_book(books[0])
            action = "Approaching book"
        elif "GRAB_BOOK" in response and books:
            self._execute_grab_book()
            action = "Grabbing book"
        else:
            action = "No action taken - continuing search"
            self._execute_turn_right()  # Default action

        self.progress = action
        self.mode = ControlMode.MANUAL
        return True

    def _execute_turn_left(self):
        """Execute a left turn using wheel motors"""
        if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
            left_id = self.dof_map["wheel_left_joint"]
            right_id = self.dof_map["wheel_right_joint"]

            # Set wheel velocities for turning
            self.motor_settings[left_id].velocity_target = -self.turn_speed
            self.motor_settings[right_id].velocity_target = self.turn_speed

            # Apply settings
            self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
            self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

            # Step physics for a second
            import time
            t_end = time.time() + 1.0
            while time.time() < t_end:
                self.sim.step_physics(1/60.0)

    def _execute_turn_right(self):
        """Execute a right turn using wheel motors"""
        if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
            left_id = self.dof_map["wheel_left_joint"]
            right_id = self.dof_map["wheel_right_joint"]

            # Set wheel velocities for turning
            self.motor_settings[left_id].velocity_target = self.turn_speed
            self.motor_settings[right_id].velocity_target = -self.turn_speed

            # Apply settings
            self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
            self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

            # Step physics for a second
            import time
            t_end = time.time() + 1.0
            while time.time() < t_end:
                self.sim.step_physics(1/60.0)

    def _execute_move_forward(self):
        """Execute a forward movement using wheel motors"""
        if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
            left_id = self.dof_map["wheel_left_joint"]
            right_id = self.dof_map["wheel_right_joint"]

            # Set wheel velocities for moving forward
            self.motor_settings[left_id].velocity_target = self.drive_speed
            self.motor_settings[right_id].velocity_target = self.drive_speed

            # Apply settings
            self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
            self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

            # Step physics for a second
            import time
            t_end = time.time() + 1.0
            while time.time() < t_end:
                self.sim.step_physics(1/60.0)

    def camera_guided_book_search(self):
        """Use camera pan/tilt to scan for books with LLM guidance"""
        print("📷 Starting camera-guided book search")
        self.mode = ControlMode.EXECUTING
        self.task_description = "Find book using camera"
        self.progress = "Preparing camera"

        # First, ensure we have access to camera joints
        if "pan" not in self.dof_map or "tilt" not in self.dof_map:
            print("ERROR: Camera pan/tilt joints not found")
            return False

        # Get joint IDs
        pan_id = self.dof_map["pan"]
        tilt_id = self.dof_map["tilt"]

        # Function to directly move camera joints
        def move_camera(pan_pos, tilt_pos):
            print(f"Moving camera to pan={pan_pos:.2f}, tilt={tilt_pos:.2f}")

            # Override camera position using joint positions
            joint_positions = self.locobot.joint_positions
            pan_offset = self.locobot.get_link_joint_pos_offset(pan_id)
            tilt_offset = self.locobot.get_link_joint_pos_offset(tilt_id)

            if pan_offset >= 0 and tilt_offset >= 0:
                # Apply new positions
                joint_positions[pan_offset] = pan_pos
                joint_positions[tilt_offset] = tilt_pos
                self.locobot.joint_positions = joint_positions

                # Set them repeatedly over time to prevent stabilization
                import time
                for _ in range(10):
                    # Re-apply positions several times
                    self.locobot.joint_positions = joint_positions
                    # Step physics briefly
                    self.sim.step_physics(1/60.0)
                    time.sleep(0.05)

                print(f"Camera moved to: pan={pan_pos:.2f}, tilt={tilt_pos:.2f}")
                return True
            return False

        # Get current view
        obs = self.sim.get_sensor_observations()
        front_img = obs['robot_rgb'].copy()

        # Run object detection
        from perception import Perception
        perception = Perception()
        detected_objects = perception.detect_objects(front_img)
        books = [obj for obj in detected_objects if obj.get('name', '').lower() == 'book']

        # Prepare context for LLM
        book_context = ""
        if books:
            book_context = f"I detected {len(books)} books in the current view."
        else:
            book_context = "I don't see any books in the current view."

        # Ask LLM for camera movement
        prompt = f"""You are controlling a robot's camera to find books. {book_context}

    You control the camera's pan (left/right) and tilt (up/down) motors.
    - Pan range: -1.0 (far left) to 1.0 (far right)
    - Tilt range: -0.5 (down) to 0.1 (up)

    Current camera orientation:
    - Pan: 0.0 (center)
    - Tilt: -0.26 (slightly down)

    What camera position should I move to next? Respond with ONLY:
    PAN: [value] TILT: [value]
    """

        # Send to LLM
        response = self._query_llm(prompt)
        print(f"LLM camera guidance: {response}")

        # Extract pan/tilt values from response
        import re
        pan_match = re.search(r"PAN:\s*([-+]?\d*\.\d+|\d+)", response)
        tilt_match = re.search(r"TILT:\s*([-+]?\d*\.\d+|\d+)", response)

        # Apply camera movement if values found
        if pan_match and tilt_match:
            try:
                pan_value = float(pan_match.group(1))
                tilt_value = float(tilt_match.group(1))

                # Clamp to valid ranges
                pan_value = max(-1.0, min(1.0, pan_value))
                tilt_value = max(-0.5, min(0.1, tilt_value))

                # Move camera
                success = move_camera(pan_value, tilt_value)
                if success:
                    self.progress = f"Moved camera to pan={pan_value:.2f}, tilt={tilt_value:.2f}"
                else:
                    self.progress = "Failed to move camera"
            except ValueError:
                self.progress = "Invalid camera values from LLM"
        else:
            self.progress = "Could not extract camera values from LLM response"

        self.mode = ControlMode.MANUAL
        return True