#!/usr/bin/env python3
import os
import sys
import time
import threading
import queue
import json
import random
import numpy as np
import magnum as mn
import requests
import traceback
from enum import Enum, auto 
import yaml
import re
from typing import Dict, Tuple, Optional, Any

from tools import get_tool_descriptions
from memory.memory import BookExperienceMemory

sys.path.append(os.path.join(os.path.dirname(__file__), 'Embodied_RAG'))
from memory.semantic_memory import SemanticMemory
# Import stubs instead to avoid OpenAI dependency
from embodied_rag_stubs import EmbodiedRetriever

class MissionState(Enum):
    SEARCH = auto()
    VALIDATE = auto()
    REACH_CHECK = auto()
    PICK = auto()
    COMPLETE = auto()


class ControlMode(Enum):
    AUTONOMOUS = "autonomous"
    CAMERA_SCAN = "camera_scan"
    EXPLORE = "explore"
    EXECUTING = "executing"
    BOOK_SEARCH = "book_search"

class MissionStateMachine:
    """State machine for book search and pick mission."""
    
    def __init__(self, controller):
        self.controller = controller
        self.current_state = MissionState.SEARCH
        self.target_book = None
        self.pick_attempts = 0
        self.max_pick_attempts = 3
        
    def update(self):
        """Update mission state based on current conditions."""
        if self.current_state == MissionState.SEARCH:
            # Check if we found any books
            with self.controller.book_search_agent.found_books_lock:
                num_books = len(self.controller.book_search_agent.found_books)
                
            if num_books > 0:
                print(f"Found {num_books} books, transitioning to VALIDATE phase")
                self.current_state = MissionState.VALIDATE
                
        elif self.current_state == MissionState.VALIDATE:
            # Select the closest reachable book
            with self.controller.book_search_agent.found_books_lock:
                books = self.controller.book_search_agent.found_books.copy()
            
            if books:
                # Find closest book with world position
                robot_pos = self.controller.robot_position
                closest_book = None
                min_distance = float('inf')
                
                for book in books:
                    if "world_position" in book and robot_pos is not None:
                        book_pos = np.array(book["world_position"])
                        robot_base = np.array(robot_pos)
                        distance = np.linalg.norm(book_pos - robot_base)
                        
                        if distance < min_distance:
                            min_distance = distance
                            closest_book = book
                
                if closest_book:
                    self.target_book = closest_book
                    self.current_state = MissionState.REACH_CHECK
                    print(f"Selected book at distance {min_distance:.2f}m")
                    
        elif self.current_state == MissionState.REACH_CHECK:
            # Check if book is within arm reach (0.6m)
            if self.target_book and "world_position" in self.target_book:
                book_pos = np.array(self.target_book["world_position"])
                robot_pos = np.array(self.controller.robot_position)
                distance = np.linalg.norm(book_pos - robot_pos)
                
                if distance < 0.6:  # Arm reach threshold
                    print(f"Book is within reach ({distance:.2f}m), attempting pick")
                    self.current_state = MissionState.PICK
                else:
                    print(f"Book too far ({distance:.2f}m), need to move closer")
                    # TODO: Add navigation to get closer
                    self.current_state = MissionState.SEARCH
                    
        elif self.current_state == MissionState.PICK:
            # Attempt to pick the book
            if self.pick_attempts < self.max_pick_attempts:
                success = self._attempt_pick()
                if success:
                    self.current_state = MissionState.COMPLETE
                    print("Pick successful! Mission complete.")
                else:
                    self.pick_attempts += 1
                    print(f"Pick failed, attempt {self.pick_attempts}/{self.max_pick_attempts}")
                    if self.pick_attempts >= self.max_pick_attempts:
                        print("Max pick attempts reached, returning to search")
                        self.current_state = MissionState.SEARCH
                        self.pick_attempts = 0

                        
    def _attempt_pick(self):
        """Attempt to pick the target book."""
        if not self.target_book:
            print("Error: No target book specified for pick operation")
            return False
            
        # Validate target book has position
        if not self.target_book.get("world_position"):
            print("Error: Target book has no world position")
            return False
            
        # Send pick command through the command queue
        try:
            self.controller.command_queue.put({
                'tool': '_execute_pick',
                'params': {
                    'book_position': self.target_book["world_position"],
                    'book_object': self.target_book.get("object_ref")
                }
            })
            
            # Wait for result with proper timeout and exception handling
            try:
                result = self.controller.result_queue.get(timeout=60.0)  # Significantly increased timeout for pick operation
                success = result.get('success', False)
                print(f"Pick operation {'succeeded' if success else 'failed'}: {result.get('message', '')}")
                return success
            except queue.Empty:
                print("Pick operation timed out waiting for result")
                return False
        except Exception as e:
            print(f"Error executing pick operation: {e}")
            return False


class Controller:
    """LLM-powered robot controller for fully autonomous operations."""
    
    def __init__(self, llm_model="qwen3:8b", api_url="http://localhost:11434/api/chat", debug=True):
        """Initialize the controller with the specified LLM model."""
        self.llm_model = llm_model
        self.api_url = api_url
        self.mode = ControlMode.AUTONOMOUS
        self.target_object = None
        self.target_position = None
        self.task_description = None
        self.progress = "Ready for autonomous operations"
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
        self.sim = None
        
        # Camera control access
        self.camera_controller = None
        
        # Perception information
        self.current_image = None
        self.detected_objects = []
        self.scene_analysis = None
        
        # Thread control
        self.running = False
        self.thread = None
        
        # Tracking for camera scanning
        self.current_scan_pattern = None
        self.scan_start_time = None
        self.scan_results = []
        
        # LLM interaction context
        self.conversation_history = []
        
        print(f"Autonomous controller initialized with LLM model: {llm_model}")
    
    def set_robot_controls(self, locobot, motor_ids, motor_settings, dof_map, 
                          drive_speed=1.0, turn_speed=0.5, arm_speed=0.7, grip_speed=0.7, sim=None):
        """Set the robot control interfaces for direct motor control."""
        self.locobot = locobot
        self.motor_ids = motor_ids
        self.motor_settings = motor_settings
        self.dof_map = dof_map
        self.sim = sim
        
        # Initialize camera controller
        from camera.camera_controller import CameraController
        self.camera_controller = CameraController(
            locobot=self.locobot,
            motor_ids=self.motor_ids,
            motor_settings=self.motor_settings,
            dof_map=self.dof_map,
            debug=self.debug
        )
        
        if self.debug:
            print("Robot controls set and camera controller initialized")
    
    def start(self):
        """Start the controller thread."""
        if self.thread is not None and self.thread.is_alive():
            print("Controller thread is already running")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(f"Started autonomous controller thread with LLM model: {self.llm_model}")
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
        
        # Make sure to stop any ongoing camera scanning
        if self.camera_controller:
            self.camera_controller.stop_scanning()
            
        print("Stopped controller thread")
    
    def update_robot_state(self, position, rotation):
        """Update the current robot position and orientation."""
        self.robot_position = position
        self.robot_rotation = rotation
        
        # Update robot position in visualization tool if available
        from tools import visualize_semantic_forest
        if hasattr(visualize_semantic_forest, 'robot_position'):
            visualize_semantic_forest.robot_position = position
            
        # Update robot position in memory system if available
        if hasattr(self, 'book_search_agent') and hasattr(self.book_search_agent, 'book_memory'):
            memory = self.book_search_agent.book_memory
            if hasattr(memory, 'update_robot_position') and callable(memory.update_robot_position):
                memory.update_robot_position(position)
    
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
            "camera_scanning": self.camera_controller.is_scanning if self.camera_controller else False
        }
    
    def start_camera_scan(self, pattern=None, granularity=None):
        """Start automated camera scanning with specified pattern."""
        if not self.camera_controller:
            self.progress = "Failed - Camera controller not initialized"
            return False
            
        self.mode = ControlMode.CAMERA_SCAN
        self.task_description = "Automated environment scanning"
        self.progress = "Starting scan"
        
        # Use LLM to determine scan pattern if not specified
        if not pattern and not granularity:
            llm_scan_command = self._generate_scan_command()
            pattern = llm_scan_command.get("pattern", "detailed")
        
        # Start scanning with proper parameters
        pause_time = 3.0  # Default pause time
        if isinstance(pattern, dict) and "pause_time" in pattern:
            pause_time = pattern.get("pause_time", 3.0)
        elif hasattr(self, '_generate_scan_command') and isinstance(pattern, str):
            scan_command = self._generate_scan_command()
            pause_time = scan_command.get("pause_time", 3.0)
            
        # Start scanning with pattern and pause time
        success = self.camera_controller.start_scanning(
            pattern_name=pattern if isinstance(pattern, str) else pattern.get("pattern", "detailed"),
            pause_time=pause_time
        )
        
        if success:
            self.progress = f"Scanning with pattern: {pattern}"
            self.scan_start_time = time.time()
            self.scan_results = []
            
            # Start a thread to monitor and process scanning results
            threading.Thread(
                target=self._process_scan_results,
                daemon=True
            ).start()
        else:
            self.progress = "Failed to start scanning"
            self.mode = ControlMode.AUTONOMOUS
            
        return success
    
    def _generate_scan_command(self):
        """Use LLM to generate an optimal scanning command."""
        prompt = """As an AI assistant controlling a robot's camera, determine the best scanning pattern 
        for the current environment. Think step by step:
        
        1. Consider the available scanning patterns:
           - "wide": 5 positions scanning left to right
           - "detailed": 10 positions in a grid pattern 
           - "vertical": 4 positions scanning up to down
           - "horizontal": 7 positions scanning left to right
           - "object_search": 6 positions focused on areas where objects typically are
           - "continuous": Dense pattern covering the entire field of view
        
        2. Analyze what would be most effective for general scene understanding
        
        3. Choose one pattern and explain why it's optimal
        
        Respond with a JSON object containing:
        {
          "reasoning": "Your step-by-step reasoning",
          "pattern": "selected_pattern",
          "pause_time": seconds_to_pause_at_each_position
        }
        """
        
        response = self._query_llm(prompt)
        
        try:
            # Extract JSON from response
            json_pattern = re.search(r'\{.*\}', response, re.DOTALL)
            if json_pattern:
                command = json.loads(json_pattern.group(0))
                return command
        except Exception as e:
            print(f"Error parsing LLM scan command: {e}")
        
        # Default if parsing fails
        return {"pattern": "detailed", "pause_time": 3.0}
    
    def stop_camera_scan(self):
        """Stop the current camera scanning operation."""
        if not self.camera_controller:
            return False
            
        success = self.camera_controller.stop_scanning()
        
        if success:
            self.progress = "Scan stopped"
            self.mode = ControlMode.AUTONOMOUS
            
            # Finalize scanning results
            scan_duration = time.time() - self.scan_start_time if self.scan_start_time else 0
            print(f"Scan completed in {scan_duration:.1f} seconds")
            print(f"Detected {len(self.scan_results)} objects during scan")
            
        return success
    
    def _process_scan_results(self):
        """Process the results from automated scanning."""
        last_processed_time = time.time()
        
        # Continue processing until scan is complete
        while self.camera_controller and self.camera_controller.is_scanning:
            # Check if it's time to process the current view
            current_time = time.time()
            
            # Process roughly every 2 seconds
            if current_time - last_processed_time > 2.0:
                last_processed_time = current_time
                
                try:
                    # Get current camera view
                    if self.sim:
                        obs = self.sim.get_sensor_observations()
                        front_img = obs['robot_rgb'].copy()
                        
                        # Analyze the image with perception
                        if hasattr(self, 'perception'):
                            # Object detection
                            detected_objects = self.perception.detect_objects(front_img)
                            self.scan_results.extend(detected_objects)
                            
                            # Get camera state for reference
                            camera_state = self.camera_controller.get_current_state()
                            
                            # Log the detected objects with camera position
                            for obj in detected_objects:
                                obj["camera_position"] = {
                                    "pan": camera_state["pan"],
                                    "tilt": camera_state["tilt"]
                                }
                                
                            self.progress = f"Scanning: {len(self.scan_results)} objects found"
                except Exception as e:
                    print(f"Error processing scan results: {e}")
            
            # Small sleep to avoid consuming CPU
            time.sleep(0.1)
        
        # Final processing when scan completes
        try:
            if self.scan_start_time:
                scan_duration = time.time() - self.scan_start_time
                self.progress = f"Scan complete: {len(self.scan_results)} objects in {scan_duration:.1f}s"
                
                # Filter duplicates by object name and position
                unique_objects = {}
                for obj in self.scan_results:
                    if "name" in obj and "bbox" in obj:
                        key = f"{obj['name']}_{obj['bbox'][0]:.1f}_{obj['bbox'][1]:.1f}"
                        unique_objects[key] = obj
                
                # Update results with unique objects
                self.scan_results = list(unique_objects.values())
                
                # Update detected objects
                self.detected_objects = self.scan_results
                
                print(f"Scan processing complete: {len(self.scan_results)} unique objects found")
                
                # Process results with LLM
                self._analyze_scan_results()
                
                # Return to autonomous mode
                self.mode = ControlMode.AUTONOMOUS
        except Exception as e:
            print(f"Error in final scan processing: {e}")
            self.progress = f"Scan processing error: {str(e)}"
    
    def _analyze_scan_results(self):
        """Use LLM to analyze scan results and determine next actions."""
        if not self.scan_results:
            return
            
        # Create a summary of found objects
        object_summary = "\n".join([
            f"- {obj.get('name', 'Unknown object')}: {obj.get('description', 'No description')}"
            for obj in self.scan_results[:15]  # Limit to first 15 objects
        ])
        
        prompt = f"""You are an AI assistant controlling a robot. The robot has completed a scan 
        of the environment and found {len(self.scan_results)} objects.
        
        Here are the first objects detected:
        {object_summary}
        
        Think step by step:
        1. What are the most interesting objects detected?
        2. Are there any objects that warrant further investigation?
        3. What should be the next action for the robot?
        
        Respond with a JSON object containing:
        {{
          "reasoning": "Your step-by-step reasoning",
          "summary": "A brief summary of what was found",
          "next_action": "The next recommended action (scan, explore, investigate, etc.)",
          "target_object": "Any specific object to focus on",
          "action_parameters": {{}} // Any parameters for the next action
        }}
        """
        
        response = self._query_llm(prompt)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group(0))
                
                # Update state based on analysis
                if "summary" in analysis:
                    print("\n===== SCAN ANALYSIS =====")
                    print(analysis["summary"])
                    print("=========================\n")
                
                if "target_object" in analysis and analysis["target_object"]:
                    self.target_object = analysis["target_object"]
                
                if "next_action" in analysis:
                    next_action = analysis["next_action"].lower()
                    if next_action == "scan":
                        # Start another scan with different pattern
                        pattern = analysis.get("action_parameters", {}).get("pattern", "detailed")
                        self.start_camera_scan(pattern=pattern)
                    elif next_action == "investigate" and self.target_object:
                        # Focus on specific object
                        self.investigate_object(self.target_object)
        except Exception as e:
            print(f"Error processing scan analysis: {e}")
    
    def execute_camera_command(self, command):
        """Execute a camera command from LLM reasoning process."""
        if not self.camera_controller:
            print("Camera controller not initialized")
            return {"success": False, "error": "Camera controller not available"}
            
        try:
            result = self.camera_controller.execute_llm_command(command)
            
            # Log command execution
            if result["success"]:
                print(f"Executed camera command: {command.get('action')} - {result.get('observation', '')}")
            else:
                print(f"Failed to execute camera command: {command.get('action')} - {result.get('error', '')}")
                
            return result
        except Exception as e:
            print(f"Error executing camera command: {e}")
            return {"success": False, "error": str(e)}
    
    def investigate_object(self, object_name):
        """Focus camera on a specific object of interest."""
        # Find object in detected objects
        target_obj = None
        for obj in self.detected_objects:
            if obj.get("name", "").lower() == object_name.lower():
                target_obj = obj
                break
                
        if not target_obj:
            print(f"Object '{object_name}' not found in detected objects")
            return False
            
        # Get camera position where object was detected
        if "camera_position" in target_obj:
            cam_pos = target_obj["camera_position"]
            pan = cam_pos.get("pan")
            tilt = cam_pos.get("tilt")
            
            if pan is not None and tilt is not None:
                # Move camera to position where object was detected
                command = {
                    "action": "move",
                    "reasoning": f"Moving to position where {object_name} was detected",
                    "parameters": {
                        "absolute": True,
                        "pan": pan,
                        "tilt": tilt
                    }
                }
                
                result = self.execute_camera_command(command)
                if result["success"]:
                    self.progress = f"Investigating {object_name}"
                    self.target_object = object_name
                    return True
                    
        return False
    
    def _run(self):
        """Main controller thread loop."""
        print("Autonomous controller thread started")
        try:
            
            while self.running:
                # Check controller mode and update accordingly
                if self.mode == ControlMode.CAMERA_SCAN:
                    # Monitor camera scanning
                    if self.camera_controller and not self.camera_controller.is_scanning:
                        # Scanning finished, ask LLM for next action
                        self._request_next_action()
                
                elif self.mode == ControlMode.AUTONOMOUS:
                    # Regular autonomous mode - check if we should initiate an action
                    if not hasattr(self, 'last_autonomous_action'):
                        self.last_autonomous_action = 0
                    
                    # Every 10 seconds, ask LLM what to do next
                    current_time = time.time()
                    if current_time - self.last_autonomous_action > 10.0:
                        self.last_autonomous_action = current_time
                        self._request_next_action()
                
                # Sleep to avoid consuming too much CPU
                time.sleep(0.1)
        except Exception as e:
            print(f"Error in controller thread: {e}")
            traceback.print_exc()
        print("Controller thread stopped")
    
    def _request_next_action(self):
        """Ask the LLM for the next action to take."""
        # Get environment context
        camera_state = self.camera_controller.get_current_state() if self.camera_controller else {}
        num_objects = len(self.detected_objects)
        
        prompt = f"""You are an AI assistant controlling a robot with a camera. 
        Current state:
        - Mode: {self.mode.value}
        - Robot position: {self.robot_position}
        - Camera pan: {camera_state.get('pan', 'unknown')}, tilt: {camera_state.get('tilt', 'unknown')}
        - Objects detected: {num_objects}
        - Current progress: {self.progress}
        
        Think step by step:
        1. What is the most appropriate next action for the robot?
        2. How should the camera be positioned to gather useful information?
        3. What specific command parameters would be most effective?
        
        Available actions:
        - "scan": Scan the environment (patterns: "wide", "detailed", "vertical", "horizontal", "object_search")
        - "look_at": Look in a specific direction (directions: "center", "left", "right", "up", "down")
        - "sweep": Perform a gradual pan sweep at a specific tilt
        - "investigate": Examine a specific detected object more closely
        
        Respond with a JSON object containing:
        {{
          "reasoning": "Your step-by-step reasoning",
          "action": "The selected action",
          "parameters": {{}} // Parameters specific to the selected action
        }}
        """
        
        response = self._query_llm(prompt)
        
        try:
            # Extract JSON from response
            import re
            # More robust pattern to extract well-formed JSON
            json_match = re.search(r'\{(?:[^{}]|"(?:\\.|[^"\\])*"|\{(?:[^{}]|"(?:\\.|[^"\\])*")*\})*\}', response, re.DOTALL)
            
            if json_match:
                try:
                    json_str = json_match.group(0)
                    # Clean up common JSON issues
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                    command = json.loads(json_str)
                    
                    # Execute camera command
                    action = command.get("action", "").lower()
                    
                    if action in ["scan", "look_at", "sweep"]:
                        self.execute_camera_command(command)
                    elif action == "investigate":
                        object_name = command.get("parameters", {}).get("object")
                        if object_name:
                            self.investigate_object(object_name)
                    
                    # Update conversation history
                    self.conversation_history.append({"role": "assistant", "content": response})
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e} - Trying to fix format")
                    # Try to clean up JSON
                    try:
                        # Remove problematic characters and try again
                        clean_json = json_str.replace('\n', ' ').replace('\r', '')
                        # Remove any trailing commas in objects/arrays
                        clean_json = re.sub(r',\s*}', '}', clean_json)
                        clean_json = re.sub(r',\s*]', ']', clean_json)
                        command = json.loads(clean_json)
                        if "action" in command:
                            self.execute_camera_command(command)
                    except Exception:
                        print(f"Failed to fix JSON format: {json_str}")
            else:
                print("No valid JSON found in LLM response")
        except Exception as e:
            print(f"Error processing LLM action: {e}")
    
    def _query_llm(self, prompt):
        """Query the LLM for planning or decision making."""
        try:
            # Create system prompt for LLM
            system_prompt = """You are a robot control assistant specialized in autonomous camera control.
            Your task is to direct the robot's camera for optimal environment perception.
            
            Always structure your thinking step by step, reasoning clearly toward a decision.
            Provide your responses in JSON format whenever requested.
            """
            
            # Prepare the API request payload
            payload = {
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            
            # Make the API request with increased timeout to prevent timeouts
            # LLMs can take time to generate complex responses
            response = requests.post(self.api_url, json=payload, timeout=120.0)  # 2 minute timeout
            response.raise_for_status()
            result = response.json()
            
            # Extract the response content
            response_text = result.get("message", {}).get("content", "")
            
            # Keep conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return "Error: Could not get response from language model"
            

    def _is_within_reach(self, world_pos: Tuple[float,float,float]) -> bool:
        # Simple spherical check (0.5 m reach)
        if self.robot_position is None:
            return False
        base = np.array(self.robot_position)
        return np.linalg.norm(base - np.array(world_pos)) < 0.5

    
    def _plan_book_actions(self, books):
        """Use LLM to plan actions for found books."""
        if not books:
            return
            
        # Create a summary of found books
        book_summary = "\n".join([
            f"- Book at position: {book.get('world_position', 'unknown')}, "
            f"description: {book.get('description', 'No description')}"
            for book in books[:5]  # Limit to first 5 books
        ])
        
        prompt = f"""You are an AI assistant controlling a robot. The robot has found {len(books)} books 
        during its scan.
        
        Book details:
        {book_summary}
        
        Think step by step:
        1. What's the most interesting book found?
        2. Should the robot move closer to examine any book?
        3. What camera angles would provide the best view of the books?
        
        Respond with a JSON object containing:
        {{
          "reasoning": "Your step-by-step reasoning",
          "recommended_action": "examine", // or "navigate", "grasp", etc.
          "target_book": 0, // index of most interesting book (0-based)
          "camera_command": {{
            "action": "move", // camera action
            "parameters": {{}} // parameters for the camera action
          }}
        }}
        """
        
        response = self._query_llm(prompt)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(0))
                
                # Execute camera command if provided
                if "camera_command" in plan:
                    self.execute_camera_command(plan["camera_command"])
                
                # Set target book
                if "target_book" in plan and 0 <= plan["target_book"] < len(books):
                    target_idx = plan["target_book"]
                    target_book = books[target_idx]
                    self.target_object = f"book_{target_idx}"
                    
                    # Show interest in this book
                    if "world_position" in target_book:
                        self.target_position = target_book["world_position"]
                        print(f"Target book position: {self.target_position}")
                        if self._is_within_reach(self.target_position):
                            from manipulation.pick_place_demo import PickAndPlaceTask
                            task = PickAndPlaceTask(
                                sim=self.sim, locobot=self.locobot,
                                motor_ids=self.motor_ids, motor_settings=self.motor_settings,
                                dof_map=self.dof_map,
                                book_object=self._get_book_handle(self.target_object)
                            )
                            task.execute_pick_and_place()
                            self.mission_active = False
                            print(" Mission complete — book picked!")

                            self.state = MissionState.COMPLETE

                
                # Update progress with recommendation
                if "recommended_action" in plan:
                    self.progress = f"Recommended: {plan['recommended_action']} book"
        except Exception as e:
            print(f"Error processing book action plan: {e}")




class AutonomousBookSearchAgent:
    """Fully autonomous agent for book searching using camera control."""
    
    def __init__(self, camera_controller, perception, sim, 
                 llm_model="qwen3:8b", prompts_file="prompt.yaml", 
                 tool_descriptions=None, semantic_memory=None, config=None, debug=False):
        """Initialize the autonomous agent."""
        self.llm_model = llm_model
        self.api_url = "http://localhost:11434/api/chat"
        self.debug = debug
        
        # Set components
        self.camera_controller = camera_controller
        self.perception = perception
        self.sim = sim
        self.config = config or {}
        
        # Load YAML prompts
        self.prompt_templates = self._load_prompts(prompts_file)
        self.tool_descriptions = tool_descriptions or "No tool descriptions available"
        
        # Thread-safe command queue
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Mission state
        self.found_books = []
        self.found_books_lock = threading.Lock()
        self.scanned_positions = []
        from enum import auto
        self.state = MissionState.SEARCH
        self.mission_active = False
        self.conversation_history = []
        
        # Initialize memory FIRST
        # Use provided semantic memory or create default
        if semantic_memory:
            self.book_memory = semantic_memory
        else:
            # Add memory system - adapt to use SemanticMemory if available
            try:
                from memory.semantic_memory import SemanticMemory
                self.book_memory = SemanticMemory(self.config)
                print("Using enhanced Semantic Forest memory system")
            except ImportError:
                from memory.memory import BookExperienceMemory
                self.book_memory = BookExperienceMemory(save_path="book_search_memory.json")
                print("Using basic book experience memory system")
        
        # Use ONLY our stub implementation to avoid OpenAI dependency
        print("Initializing Embodied RAG with Ollama-only stub implementation")
        
        # Import our stub implementation
        from embodied_rag_stubs import EmbodiedRAG
        self.embodied_rag = EmbodiedRAG(
            llm_model=llm_model,
            config=config,
            semantic_memory=self.book_memory
        )
        
        # Connect the semantic memory
        if hasattr(self.embodied_rag, 'semantic_memory') and not self.embodied_rag.semantic_memory:
            self.embodied_rag.semantic_memory = self.book_memory
        
        # Register visualization tools
        from tools import visualize_semantic_forest
        visualize_semantic_forest.forest = self.book_memory.semantic_forest
        visualize_semantic_forest.robot_position = [0, 0, 0]  # Will be updated during mission
        
        # Add RAG capability to the agent
        self._add_rag_capabilities()
        
        # Add search statistics
        self.search_stats = {
            'total_detections': 0,
            'unique_books': 0,
            'duplicate_rejections': 0
        }
        
        print("Autonomous Book Search Agent initialized")

    
    def _add_rag_capabilities(self):
        """Add Embodied RAG capabilities to the agent."""
        # Add RAG-specific tools to the tool registry
        from tools import TOOL_REGISTRY

        # Make sure we have the RAG query and analyze methods
        if hasattr(self.embodied_rag, 'query') and hasattr(self.embodied_rag, 'analyze_spatial'):
            # Add RAG tools to the agent's available tools
            rag_tools = {
                "query_books": self.embodied_rag.query,
                "analyze_spatial_relationship": self.embodied_rag.analyze_spatial
            }
            
            for name, tool in rag_tools.items():
                TOOL_REGISTRY[name] = tool
                print(f"Registered RAG tool: {name}")
            
            # Update tool descriptions
            self.tool_descriptions += "\n\n== RAG Tools ==\n"
            self.tool_descriptions += "query_books: Search for books using natural language\n"
            self.tool_descriptions += "analyze_spatial_relationship: Analyze spatial relationships between books\n"
        else:
            print("WARNING: Embodied RAG does not have required methods - some tools will be unavailable")
    
    def start_mission(self):
        """Start the autonomous book search mission."""
        self.mission_active = True
        
        # Get initial camera state
        camera_state = self.camera_controller.get_current_state()
        
        # Format initial prompt
        initial_prompt = self.prompt_templates['initial_prompt'].format(
            current_pan=camera_state['pan'],
            current_tilt=camera_state['tilt']
        )
        
        print("\n===== Starting Autonomous Book Search Mission =====")
        
        # Execute ReAct loop
        self._execute_react_loop(initial_prompt)
        
        # Handle different memory classes
        if hasattr(self.book_memory, 'save_memory'):
            self.book_memory.save_memory()
            print(f"\n===== Mission Complete =====")
            print(f"Total unique books found: {len(self.found_books)}")
            
            # Get save path depending on memory type
            if hasattr(self.book_memory, 'save_path'):
                print(f"Memory saved to: {self.book_memory.save_path}")
            else:
                memory_save_path = self.config.get('paths', {}).get('memory_save', 'semantic_forest_save.json')
                print(f"Memory saved to: {memory_save_path}")
            
            print("===========================\n")
        else:
            print(f"\n===== Mission Complete =====")
            print(f"Total unique books found: {len(self.found_books)}")
            print("===========================\n")
        
        return self.found_books
    
    def _execute_action(self, action: Dict) -> str:
        """Execute action via main thread only."""
        tool_name = action.get('tool')
        params = action.get('parameters', {})

        # Send to main thread
        self.command_queue.put({
            'tool': tool_name,
            'params': params
        })

        # Wait for result
        try:
            result = self.result_queue.get(timeout=60.0)  # Significantly increased timeout for complex actions

            # Get camera state from result, not directly
            camera_state = result.get('camera_state', {'pan': 0, 'tilt': 0})

            # Format observation
            return self.prompt_templates['observation_template'].format(
                action_type=tool_name,
                status=result.get('status', 'unknown'),
                current_pan=camera_state['pan'],
                current_tilt=camera_state['tilt'],
                num_objects=result.get('num_objects', 0),
                num_books=result.get('num_books', 0),
                additional_info=json.dumps(result)
            )
        except queue.Empty:
            return self.prompt_templates['action_error_template'].format(
                error_message="Action timeout"
            )
    
    def _check_for_books(self) -> list:
        """Check for books using memory deduplication."""
        # Send command to main thread
        self.command_queue.put({
            'tool': '_check_books',
            'params': {}
        })

        try:
            result = self.result_queue.get(timeout=30.0)  # Significantly increased timeout for book checking
            books = result.get('books', [])

            # Get camera state
            camera_state = self.camera_controller.get_current_state()

            # Process through memory for deduplication
            unique_books = []
            self.search_stats['total_detections'] += len(books)

            for book in books:
                # Add to memory and check uniqueness
                node_id, is_novel = self.book_memory.add_observation(book, camera_state)

                if is_novel:
                    unique_books.append(book)
                    self.search_stats['unique_books'] += 1
                    print(f"New unique book found: {node_id}")
                else:
                    self.search_stats['duplicate_rejections'] += 1

            return unique_books

        except queue.Empty:
            return []
        
    def _load_prompts(self, prompts_file: str) -> Dict:
        """Load prompts from YAML file."""
        prompts_path = os.path.join(os.path.dirname(__file__), prompts_file)
        try:
            with open(prompts_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading prompts file: {e}")
            return {}
    def stop_mission(self):
        """Stop the mission."""
        self.mission_active = False
        if self.camera_controller:
            self.camera_controller.stop_scanning()
        print("Mission stopped by user")


    def _execute_react_loop(self, initial_prompt: str):
        """Execute the ReAct loop until mission complete."""
        prompt = initial_prompt
        max_iterations = 20
        iteration = 0   

        # Initialize action tracking
        stats = {
            'successful_actions': 0,
            'failed_actions': 0,
            'books_found': 0,
            'last_action_type': None
        }   

        while self.mission_active and iteration < max_iterations:
            iteration += 1  

            # Get memory stats for display only
            memory_stats = self.book_memory.get_memory_stats()  

            print(f"\n{'='*50}")
            print(f"Iteration {iteration}/{max_iterations}")
            print(f"Books found: {memory_stats['total_unique_books']}")
            print(f"Observations: {memory_stats['total_observations']}")
            print(f"Recent discoveries: {memory_stats['recent_discoveries']}")
            print(f"Time since last novel: {memory_stats['time_since_last_novel']:.1f}s")
            print(f"Coverage: {memory_stats['search_coverage']:.1%}")
            print(f"{'='*50}")  

            response = self._get_llm_response(prompt)
            
            # Check if response contains an error message indicating LLM failure
            if isinstance(response, str) and response.startswith("ERROR:"):
                print("Detected LLM failure. Using fallback scanning action instead.")
                # Create a fallback scanning action
                action = {
                    "tool": "start_scan",
                    "parameters": {
                        "pattern": "book_search",
                        "pause_time": 3.0
                    }
                }
                thought = "Using fallback scanning pattern since LLM is unavailable"
                print(f"\nFallback Thought: {thought}")
            else:
                # Normal LLM response processing
                thought, action = self._parse_response(response)    
                if thought:
                    print(f"\nThought: {thought}")  

            if action:
                print(f"Action: {json.dumps(action)}")
                stats['last_action_type'] = action.get('tool', 'unknown')   

                try:
                    observation = self._execute_action(action)
                    stats['successful_actions'] += 1
                    print(f"Observation: {observation}")    

                    # Track book discoveries
                    prev_book_count = len(self.found_books)
                    self._update_book_tracking(observation)
                    new_book_count = len(self.found_books)  

                    if new_book_count > prev_book_count:
                        stats['books_found'] += (new_book_count - prev_book_count)
                        print(f"🔍 NEW BOOK DISCOVERED! Total: {new_book_count}")   

                    if self._should_complete_mission():
                        self.mission_active = False
                        break
                    
                    prompt = observation    

                except Exception as e:
                    stats['failed_actions'] += 1
                    error_msg = f"Action execution failed: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    prompt = self.prompt_templates['action_error_template'].format(
                        error_message=error_msg
                    )
            else:
                prompt = self.prompt_templates['action_error_template'].format(
                    error_message="No valid action provided."
                )
                print("WARNING: No valid action parsed from LLM response")  

            # Adaptive delay
            if stats['last_action_type'] == 'move_camera':
                time.sleep(1.0)
            else:
                time.sleep(0.5) 

        # Final report combines memory stats and action stats
        print(f"\n{'='*50}")
        print("MISSION SUMMARY")
        print(f"{'='*50}")
        print(f"Iterations completed: {iteration}")
        print(f"Successful actions: {stats['successful_actions']}")
        print(f"Failed actions: {stats['failed_actions']}")
        print(f"Books found: {stats['books_found']}")

        # Memory stats
        final_memory = self.book_memory.get_memory_stats()
        print(f"Final unique books: {final_memory['total_unique_books']}")
        print(f"Total observations: {final_memory['total_observations']}")
        print(f"Search coverage: {final_memory['search_coverage']:.1%}")
        print(f"{'='*50}\n")

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from LLM."""
        try:
            # Check if we have camera access for state information
            camera_state = None
            if hasattr(self, 'camera_controller') and self.camera_controller:
                # Request camera state through the command queue if in a thread
                self.command_queue.put({
                    'tool': '_get_camera_state',
                    'params': {}
                })

                try:
                    camera_state = self.result_queue.get(timeout=15.0)  # Increased timeout for camera state
                except queue.Empty:
                    camera_state = {'pan': 0, 'tilt': 0}
            else:
                # Default camera state if controller not available
                camera_state = {'pan': 0, 'tilt': -0.26}

            # Get tool descriptions
            tool_desc = get_tool_descriptions()

            # Format the system prompt with camera state and tool descriptions
            system_prompt = self.prompt_templates['system_prompt'].format(
                tool_descriptions=tool_desc,
                current_pan=camera_state.get('pan', 0),
                current_tilt=camera_state.get('tilt', -0.26)
            )
            # Build the message history for the LLM
            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history,  # Include previous conversation
                {"role": "user", "content": prompt}
            ]
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 2000  # Ensure we have enough space for response
            }
            # Make the actual API request with timeout
            try:
                # Use significantly increased timeout to prevent timeouts with complex prompts
                # The default 30s is often not enough for complex reasoning tasks
                response = requests.post(self.api_url, json=payload, timeout=120.0)  # 2 minute timeout
                
                # Check for HTTP errors
                response.raise_for_status()
            except requests.exceptions.Timeout:
                print("ERROR: LLM request timed out after 120 seconds. Is Ollama running?")
                print("TIP: Start Ollama with 'ollama serve' in a separate terminal or try a smaller model")
                return "ERROR: LLM request timed out. Using fallback behavior."
            except requests.exceptions.ConnectionError:
                print("ERROR: Connection to Ollama failed. Is the server running at", self.api_url)
                print("TIP: Start Ollama with 'ollama serve' in a separate terminal")
                return "ERROR: Connection to LLM failed. Using fallback behavior."

            # Parse the JSON response
            response_json = response.json()
            llm_response = response_json.get("message", {}).get("content", "")

            if not llm_response:
                print("WARNING: Empty response from LLM")
                return ""

            # Clean up the response to prevent JSON parsing issues
            llm_response = self._clean_llm_response(llm_response)

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": llm_response})

            # Keep conversation history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

            return llm_response

        except requests.exceptions.RequestException as e:
            print(f"Network error in LLM request: {e}")
            return ""
        except KeyError as e:
            print(f"Unexpected response format from LLM: {e}")
            print(f"Response content: {response_json if 'response_json' in locals() else 'No response'}")
            return ""
        except Exception as e:
            print(f"Unexpected error in LLM request: {e}")
            import traceback
            traceback.print_exc()
            return ""
        

    def _clean_llm_response(self, response: str) -> str:
        """Clean up LLM response for JSON parsing."""
        if not response:
            return response

        try:
            # Find JSON object start and track nesting level
            start = response.find('{')
            if start >= 0:
                brace_count = 0
                in_string = False
                escape_next = False

                for i in range(start, len(response)):
                    if escape_next:
                        escape_next = False
                        continue

                    char = response[i]
                    if char == '\\':
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found complete JSON object
                                potential_json = response[start:i+1]
                                try:
                                    # Verify it's valid JSON
                                    parsed = json.loads(potential_json)
                                    # Check if it has expected fields
                                    if 'tool' in parsed or 'action' in parsed:
                                        return potential_json
                                except Exception:
                                    pass
        except Exception as e:
            print(f"Error in JSON cleaning: {e}")

        return response
        
    def _parse_response(self, response: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Parse thought and action from LLM response."""
        thought = None
        action = None


        # First, let's try to clean the response
        response = response.strip()

        # Extract thought - look for various patterns
        thought_patterns = [
            r'Thought:\s*(.*?)(?=Action:|{|$)',
            r'thinking:\s*(.*?)(?=Action:|{|$)',
            r'^\s*([^{]*?)(?={|Action:|$)'  # Any text before JSON or Action
        ]

        for pattern in thought_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                thought = match.group(1).strip()
                if thought:  # Only use if we found meaningful content
                    break

        # Now let's find the action JSON with multiple strategies

        # Strategy 1: Find JSON after "Action:" label
        action_match = re.search(r'Action:\s*({.*?})', response, re.DOTALL)
        if action_match:
            json_str = action_match.group(1)
            try:
                # First, try to parse as-is
                action = json.loads(json_str)
            except json.JSONDecodeError:
                # Try to fix common issues
                try:
                    # Remove comments
                    json_str = re.sub(r'//.*?\n', '', json_str)
                    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
                    # Fix trailing commas
                    json_str = re.sub(r',\s*}', '}', json_str)
                    json_str = re.sub(r',\s*]', ']', json_str)
                    # Remove potential line breaks in string values
                    json_str = re.sub(r':\s*"(.*?)"', lambda m: ':"' + m.group(1).replace('\n', ' ') + '"', json_str)
                    # Clean up any other non-standard formatting
                    json_str = json_str.replace('\n', ' ').replace('\r', '')
                    # Try parsing again
                    action = json.loads(json_str)
                except json.JSONDecodeError as e:
                    if self.debug:
                        print(f"JSON parse error: {e}")

        # Strategy 2: Find the first complete JSON object anywhere in the response
        if not action:
            # Use a more sophisticated approach to find complete JSON objects
            json_start = -1
            json_end = -1
            brace_count = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(response):
                # Handle escape characters
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                # Handle string literals
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                # Count braces only outside of strings
                if not in_string:
                    if char == '{':
                        if brace_count == 0:
                            json_start = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and json_start != -1:
                            json_end = i + 1
                            # Try to parse this JSON object
                            json_str = response[json_start:json_end]
                            try:
                                parsed = json.loads(json_str)
                                # Check if it's an action (has 'tool' field)
                                if 'tool' in parsed:
                                    action = parsed
                                    break
                            except json.JSONDecodeError:
                                # Try the next JSON object
                                json_start = -1

        # Strategy 3: Look for common action patterns in the text
        if not action and thought:
            # Try to infer action from the thought
            thought_lower = thought.lower()

            if "scan" in thought_lower:
                if "wide" in thought_lower:
                    action = {"tool": "start_scan", "parameters": {"pattern": "wide"}}
                elif "detailed" in thought_lower:
                    action = {"tool": "start_scan", "parameters": {"pattern": "detailed"}}
                elif "book" in thought_lower:
                    action = {"tool": "start_scan", "parameters": {"pattern": "book_search"}}
                else:
                    action = {"tool": "start_scan", "parameters": {"pattern": "wide"}}

            elif "move" in thought_lower and "camera" in thought_lower:
                # Try to extract pan/tilt values
                pan_match = re.search(r'pan[:\s]*([-\d.]+)', thought, re.IGNORECASE)
                tilt_match = re.search(r'tilt[:\s]*([-\d.]+)', thought, re.IGNORECASE)

                if pan_match or tilt_match:
                    params = {"absolute": True}
                    if pan_match:
                        params["pan"] = float(pan_match.group(1))
                    if tilt_match:
                        params["tilt"] = float(tilt_match.group(1))
                    action = {"tool": "move_camera", "parameters": params}

            elif "examine" in thought_lower or "analyze" in thought_lower:
                action = {"tool": "examine_area", "parameters": {"duration": 5.0}}

        # Debug output if we couldn't parse an action
        if not action and self.debug:
            print(f"WARNING: Could not parse action from response")

        return thought, action

    def _should_complete_mission(self) -> bool:
        """Check if mission should be completed."""
        # Original conditions
        if len(self.found_books) > 0 and len(self.scanned_positions) > 10:
            return True

        # Memory-based termination
        should_terminate, reason = self.book_memory.should_terminate_search()
        if should_terminate:
            print(f"Terminating search: {reason}")
            return True

        if len(self.scanned_positions) > 30:
            return True

        return False
    
    def get_scan_statistics(self):
        """Get current scanning statistics."""
        if not hasattr(self, '_scan_stats'):
            self._scan_stats = {
                'total_positions': len(self.scan_positions),
                'current_index': 0,
                'successful_scans': 0,
                'failed_attempts': 0
            }
        return self._scan_stats.copy()

    def _deduplicate_books(self, new_books):
        """Remove duplicate books based on spatial proximity."""
        DEDUP_THRESHOLD = 0.5  # 0.5m distance threshold

        unique_books = []
        for new_book in new_books:
            if "world_position" not in new_book:
                continue

            new_pos = np.array(new_book["world_position"])
            is_duplicate = False

            # Check against existing found books
            for existing_book in self.found_books:
                if "world_position" in existing_book:
                    existing_pos = np.array(existing_book["world_position"])
                    distance = np.linalg.norm(new_pos - existing_pos)

                    if distance < DEDUP_THRESHOLD:
                        is_duplicate = True
                        print(f"Duplicate book detected at distance {distance:.3f}m")
                        break
                    
            if not is_duplicate:
                unique_books.append(new_book)

        return unique_books

    def _update_book_tracking(self, observation: str):
        """Update book tracking and trigger picking."""
        match = re.search(r'Books found: (\d+)', observation)
        if match and int(match.group(1)) > 0:
            unique_new_books = self._check_for_books()
    
            self.search_stats['total_detections'] += len(unique_new_books)
            self.search_stats['unique_books'] += len(unique_new_books)
    
            if unique_new_books:
                print(f"Found {len(unique_new_books)} new unique books!")
                
                # Trigger book picking when books are found
                if hasattr(self, 'controller') and self.controller:
                    self.controller._plan_book_actions(unique_new_books)
        
