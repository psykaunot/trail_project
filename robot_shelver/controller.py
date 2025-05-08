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
    
    def __init__(self, llm_model="qwen2.5:7b", api_url="http://localhost:11434/api/chat", debug=True):
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
        """Query the LLM for planning or decision making."""
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
            return response_text
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return "Error: Could not get response from language model"