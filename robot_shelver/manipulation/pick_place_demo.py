#!/usr/bin/env python3
import os
import sys
import numpy as np
import magnum as mn
import time
import math
import habitat_sim
import threading

# Import our environment setup and robot control functions
from environment import run_simulator_step

class PickAndPlaceTask:
    def __init__(self, sim, locobot, motor_ids, motor_settings, dof_map, book_object=None):
        self.sim = sim
        self.locobot = locobot
        self.motor_ids = motor_ids
        self.motor_settings = motor_settings
        self.dof_map = dof_map
        
        # Define arm joint names for easier reference
        self.arm_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
        self.gripper_joints = ["left_finger", "right_finger"]
        
        # Define comfortable speeds for arm movement
        self.arm_speed = 0.3
        self.grip_speed = 0.5
        
        # State tracking
        self.is_executing = False
        self.current_step = 0
        self.task_thread = None
        self.step_complete = threading.Event()
        
        # Task sequence
        self.task_sequence = [
            self.move_to_pre_grasp_position,
            self.move_to_grasp_position,
            self.grasp_object,
            self.lift_object,
            self.move_to_place_position,
            self.release_object,
            self.return_to_rest_position
        ]
        
        # Use provided book object or find it in the scene
        self.book_object = book_object
        if self.book_object is None:
            # Try to find book in the scene
            print("No book object provided, searching in scene...")
            self.book_object = self._find_book_object()
        else:
            print(f"Using provided book object at position: {self.book_object.translation}")
            
        print("Pick and Place Task initialized")
    
    def _find_book_object(self):
        """Try to find a book object that's already in the scene."""
        try:
            # Get all objects in the scene
            rigid_obj_mgr = self.sim.get_rigid_object_manager()
            all_objects = rigid_obj_mgr.get_object_handles()
            
            # Look for a book object
            book_objects = [obj for obj in all_objects if "book" in obj.lower()]
            
            if book_objects:
                # Found a book object, get it
                book_handle = book_objects[0]
                book_obj = rigid_obj_mgr.get_object_by_handle(book_handle)
                print(f"Found existing book object: {book_handle}")
                return book_obj
            
            return None
            
        except Exception as e:
            print(f"Error finding book object: {e}")
            return None
    
    def move_arm_joint(self, joint_name, angle, wait=True, max_iterations=30):
        """Move a specific arm joint to a target angle with better handling."""
        if joint_name not in self.dof_map:
            print(f"WARNING: Joint '{joint_name}' not found in dof_map")
            return False

        joint_id = self.dof_map[joint_name]

        # Get current position and joint limits
        pos_offset = self.locobot.get_link_joint_pos_offset(joint_id)
        if pos_offset < 0:
            print(f"ERROR: No valid position offset for joint {joint_name}")
            return False

        current_pos = self.locobot.joint_positions[pos_offset]

        # Get joint limits
        lower, upper = self.locobot.joint_position_limits
        lower_limit = lower[pos_offset]
        upper_limit = upper[pos_offset]

        # Clamp target angle to joint limits
        angle = max(lower_limit, min(angle, upper_limit))

        print(f"Moving {joint_name} from {current_pos:.2f} to {angle:.2f} (limits: {lower_limit:.2f} to {upper_limit:.2f})")

        # Configure joint motor for position control - much slower now
        self.motor_settings[joint_id] = habitat_sim.physics.JointMotorSettings(
            position_target=angle,
            position_gain=2.0,          # Very low gain for slower movement
            velocity_target=0.0,
            velocity_gain=0.8,          # Lower damping
            max_impulse=50.0            # Much lower force for gentler, slower movement
        )

        # Apply motor settings
        if joint_id in self.motor_ids:
            self.locobot.update_joint_motor(self.motor_ids[joint_id], self.motor_settings[joint_id])
        else:
            print(f"WARNING: No motor ID for joint {joint_name}")
            return False

        # Wait for movement to complete with UI updates
        if wait:
            iterations = 0
            while iterations < max_iterations:
                # Step physics with smaller steps for smoother motion
                self.sim.step_physics(1/200.0)
                
                # Check current position
                current_pos = self.locobot.joint_positions[pos_offset]
                if abs(current_pos - angle) < 0.1:  # Wider tolerance
                    print(f"Joint {joint_name} reached {current_pos:.2f} (target: {angle:.2f})")
                    return True
                
                iterations += 1
                # Allow main loop to update UI with more time
                time.sleep(0.05)  # 50ms pause for slower motion
            
            print(f"WARNING: Timeout moving joint {joint_name} to {angle:.2f}, got to {current_pos:.2f}")

        return True
    
    def open_gripper(self):
        """Open the gripper."""
        try:
            # Define safe open positions
            LEFT_OPEN = 0.037   # Left finger fully open position
            RIGHT_OPEN = -0.037  # Right finger fully open position
            
            # Set gripper motor settings
            left_id = self.dof_map["left_finger"]
            right_id = self.dof_map["right_finger"]
            
            self.motor_settings[left_id] = habitat_sim.physics.JointMotorSettings(
                position_target=LEFT_OPEN,
                position_gain=500.0,
                velocity_target=0.0,
                velocity_gain=100.0,
                max_impulse=1000.0
            )
            
            self.motor_settings[right_id] = habitat_sim.physics.JointMotorSettings(
                position_target=RIGHT_OPEN,
                position_gain=500.0,
                velocity_target=0.0,
                velocity_gain=100.0,
                max_impulse=1000.0
            )
            
            # Apply motor settings
            self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
            self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])
            
            # Step physics to allow gripper to open
            for _ in range(15):
                self.sim.step_physics(1/60.0)
                time.sleep(0.01)  # Allow UI updates
            
            print("Gripper opened")
            return True
            
        except Exception as e:
            print(f"Error opening gripper: {e}")
            return False
    
    def close_gripper(self):
        """Close the gripper."""
        try:
            # Define safe closed positions
            LEFT_CLOSED = 0.015    # Left finger fully closed position
            RIGHT_CLOSED = -0.015  # Right finger fully closed position
            
            # Set gripper motor settings
            left_id = self.dof_map["left_finger"]
            right_id = self.dof_map["right_finger"]
            
            self.motor_settings[left_id] = habitat_sim.physics.JointMotorSettings(
                position_target=LEFT_CLOSED,
                position_gain=500.0,
                velocity_target=0.0,
                velocity_gain=100.0,
                max_impulse=1000.0
            )
            
            self.motor_settings[right_id] = habitat_sim.physics.JointMotorSettings(
                position_target=RIGHT_CLOSED,
                position_gain=500.0,
                velocity_target=0.0,
                velocity_gain=100.0,
                max_impulse=1000.0
            )
            
            # Apply motor settings
            self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
            self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])
            
            # Step physics to allow gripper to close
            for _ in range(15):
                self.sim.step_physics(1/60.0)
                time.sleep(0.01)  # Allow UI updates
            
            print("Gripper closed")
            return True
            
        except Exception as e:
            print(f"Error closing gripper: {e}")
            return False
    
    def move_to_pre_grasp_position(self):
        """Move arm to pre-grasp position with smaller, incremental movements."""
        print("Moving to pre-grasp position...")
        
        # Move joints incrementally with pauses
        self.move_arm_joint("waist", 0.0)  # Center waist
        self.sim.step_physics(0.05)
        
        # Try for smaller shoulder movement first
        self.move_arm_joint("shoulder", 0.3)  # Raise shoulder a bit
        self.sim.step_physics(0.05)
        
        # Then elbow
        self.move_arm_joint("elbow", 0.7)  # Bend elbow slightly
        self.sim.step_physics(0.05)
        
        # Then more shoulder movement
        self.move_arm_joint("shoulder", 0.5)  # Raise shoulder more
        self.sim.step_physics(0.05)
        
        # Complete arm movement
        self.move_arm_joint("forearm_roll", 0.0)
        self.sim.step_physics(0.05)
        self.move_arm_joint("wrist_angle", 0.0)
        self.sim.step_physics(0.05)
        
        # Open gripper
        self.open_gripper()
        
        return True
    
    def move_to_grasp_position(self):
        """Lower arm to grasp the book."""
        print("Moving to grasp position...")
        
        # Lower arm to book
        self.move_arm_joint("shoulder", 0.2)  # Lower shoulder
        self.move_arm_joint("elbow", 0.8)  # Extend elbow
        self.move_arm_joint("wrist_angle", 0.5)  # Tilt wrist down
        
        return True
    
    def grasp_object(self):
        """Grasp the book."""
        print("Grasping object...")
        
        # Close gripper
        self.close_gripper()
        
        # If the book is a rigid object, attach it to the gripper
        if self.book_object is not None:
            try:
                # Set book as kinematic (optional, depends on desired behavior)
                self.book_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                
                # Get the end effector position
                wrist_link_id = self.dof_map.get("wrist_angle", -1)
                if wrist_link_id != -1:
                    # Attach book to wrist by updating its position in subsequent frames
                    self.attached_object = self.book_object
            except Exception as e:
                print(f"Error attaching object: {e}")
        
        return True
    
    def lift_object(self):
        """Lift the grasped book."""
        print("Lifting object...")
        
        # Lift arm
        self.move_arm_joint("shoulder", 0.7)  # Raise shoulder
        self.move_arm_joint("elbow", 1.2)  # Bend elbow
        
        # Update attached object position if needed
        if hasattr(self, 'attached_object') and self.attached_object is not None:
            try:
                # Get gripper position
                wrist_link_id = self.dof_map.get("wrist_angle", -1)
                if wrist_link_id != -1:
                    wrist_node = self.locobot.get_link_scene_node(wrist_link_id)
                    # Update book position based on wrist position
                    book_state = self.attached_object.rigid_state
                    book_state.translation = wrist_node.translation - mn.Vector3(0, 0.05, 0)
                    book_state.rotation = wrist_node.rotation
                    self.attached_object.rigid_state = book_state
            except Exception as e:
                print(f"Error updating attached object: {e}")
        
        return True
    
    def move_to_place_position(self):
        """Move to position for placing the book."""
        print("Moving to place position...")
        
        # Rotate waist to new position (e.g., 90 degrees to the right)
        self.move_arm_joint("waist", 1.0)  # Rotate waist
        
        # Extend arm
        self.move_arm_joint("shoulder", 0.3)  # Lower shoulder
        self.move_arm_joint("elbow", 0.9)  # Extend elbow
        
        # Update attached object position if needed
        if hasattr(self, 'attached_object') and self.attached_object is not None:
            try:
                # Get gripper position
                wrist_link_id = self.dof_map.get("wrist_angle", -1)
                if wrist_link_id != -1:
                    wrist_node = self.locobot.get_link_scene_node(wrist_link_id)
                    # Update book position based on wrist position
                    book_state = self.attached_object.rigid_state
                    book_state.translation = wrist_node.translation - mn.Vector3(0, 0.05, 0)
                    book_state.rotation = wrist_node.rotation
                    self.attached_object.rigid_state = book_state
            except Exception as e:
                print(f"Error updating attached object: {e}")
        
        return True
    
    def release_object(self):
        """Release the book."""
        print("Releasing object...")
        
        # Open gripper
        self.open_gripper()
        
        # If the book is a rigid object, detach it and make it dynamic
        if hasattr(self, 'attached_object') and self.attached_object is not None:
            try:
                self.attached_object.motion_type = habitat_sim.physics.MotionType.DYNAMIC
                self.attached_object = None
            except Exception as e:
                print(f"Error releasing object: {e}")
        
        return True
    
    def return_to_rest_position(self):
        """Return arm to rest position."""
        print("Returning to rest position...")
        
        # Move arm to rest position
        self.move_arm_joint("waist", 0.0)  # Center waist
        self.move_arm_joint("shoulder", 0.0)  # Lower shoulder
        self.move_arm_joint("elbow", 0.0)  # Straighten elbow
        self.move_arm_joint("forearm_roll", 0.0)  # Neutral forearm
        self.move_arm_joint("wrist_angle", 0.0)  # Neutral wrist
        self.move_arm_joint("wrist_rotate", 0.0)  # Neutral wrist rotation
        
        print("\n===== Pick and Place Sequence Completed =====\n")
        return True
    
    def execute_pick_and_place(self):
        """Execute the full pick and place sequence non-blockingly."""
        print("\n===== Starting Pick and Place Sequence =====\n")
        
        # Check if book was loaded successfully
        if self.book_object is None:
            print("ERROR: Book object not available. Cannot execute pick and place.")
            return False
            
        # Start the execution in a separate thread
        if not self.is_executing:
            self.is_executing = True
            self.current_step = 0
            self.task_thread = threading.Thread(target=self._execute_steps)
            self.task_thread.daemon = True
            self.task_thread.start()
            return True
        else:
            print("Already executing a pick and place sequence")
            return False
    
    def _execute_steps(self):
        """Execute the steps in sequence in a separate thread."""
        try:
            for i, step_func in enumerate(self.task_sequence):
                self.current_step = i
                step_func()
                # Longer pause between steps for much slower overall sequence
                time.sleep(1.0)  # 1 second between major steps
            
            self.is_executing = False
        except Exception as e:
            print(f"Error in pick and place sequence: {e}")
            import traceback
            traceback.print_exc()
            self.is_executing = False
    
    def get_status(self):
        """Get the current status of the pick and place task."""
        if not self.is_executing:
            return "Idle"
        
        step_names = [
            "Moving to pre-grasp position",
            "Moving to grasp position",
            "Grasping object",
            "Lifting object",
            "Moving to place position",
            "Releasing object",
            "Returning to rest position"
        ]
        
        if 0 <= self.current_step < len(step_names):
            return step_names[self.current_step]
        else:
            return "Executing"
            
    def update_attached_object(self):
        """Update the position of any attached object."""
        if not hasattr(self, 'attached_object') or self.attached_object is None:
            return
            
        try:
            # Get gripper position
            wrist_link_id = self.dof_map.get("wrist_angle", -1)
            if wrist_link_id != -1:
                wrist_node = self.locobot.get_link_scene_node(wrist_link_id)
                # Update book position based on wrist position
                book_state = self.attached_object.rigid_state
                book_state.translation = wrist_node.translation - mn.Vector3(0, 0.05, 0)
                book_state.rotation = wrist_node.rotation
                self.attached_object.rigid_state = book_state
        except Exception as e:
            if hasattr(self, 'attach_error_reported') and self.attach_error_reported:
                pass  # Don't spam the console
            else:
                print(f"Error updating attached object: {e}")
                self.attach_error_reported = True

    def _align_with_camera(self):
        """Use camera feedback to align gripper with book."""
        try:
            # Get current image
            obs = self.sim.get_sensor_observations()
            rgb_img = obs['robot_rgb'].copy()
            
            # Create perception instance if needed
            if not hasattr(self, 'perception'):
                from perception import Perception
                self.perception = Perception(model_name="llava-phi3")
                self.perception.set_camera_params(self.sim, "robot_rgb")
            
            # Detect books in view
            books = self.perception.find_books_in_image(rgb_img)
            if not books:
                print("No books visible for alignment")
                return False
            
            # Get the closest book
            book = books[0]
            
            # Get 3D position of the book
            book_pos = self.perception.get_book_position(book['bbox'])
            if book_pos is None:
                print("Could not determine book position")
                return False
            
            # Validate position (prevent unrealistic values)
            if abs(book_pos.x) > 3 or abs(book_pos.z) > 3 or book_pos.y > 1.5:
                print(f"Book position {book_pos} appears invalid - using fallback position")
                book_pos = mn.Vector3(0.0, 0.5, -2.0)
            
            print(f"Aligning with book at position {book_pos}")
            
            # Get current joint positions
            current_positions = {}
            for joint_name in ["waist", "shoulder", "elbow", "wrist_angle"]:
                if joint_name in self.dof_map:
                    link_id = self.dof_map[joint_name]
                    pos_offset = self.locobot.get_link_joint_pos_offset(link_id)
                    if pos_offset >= 0:
                        current_positions[joint_name] = self.locobot.joint_positions[pos_offset]
                    else:
                        current_positions[joint_name] = 0.0
            
            # Get wrist position
            wrist_link_id = self.dof_map.get("wrist_angle", -1)
            if wrist_link_id == -1:
                print("Could not find wrist link")
                return False
                
            wrist_node = self.locobot.get_link_scene_node(wrist_link_id)
            wrist_pos = wrist_node.translation
            
            # Calculate adjustments needed (with limits)
            dx = book_pos.x - wrist_pos.x
            dy = book_pos.y - wrist_pos.y
            dz = book_pos.z - wrist_pos.z
            
            # Limit adjustments to reasonable values
            dx = max(-0.1, min(0.1, dx))
            dy = max(-0.1, min(0.1, dy))
            dz = max(-0.1, min(0.1, dz))
            
            print(f"Adjusting gripper: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}")
            
            # Apply small joint adjustments one at a time
            if abs(dx) > 0.02:
                waist_adjustment = current_positions.get("waist", 0.0) + dx * 0.3
                self.move_arm_joint("waist", waist_adjustment)
                print(f"Adjusted waist to {waist_adjustment:.3f}")
                
            # Wait between adjustments
            for _ in range(5):
                self.sim.step_physics(1/60.0)
            
            if abs(dy) > 0.02:
                shoulder_adjustment = current_positions.get("shoulder", 0.0) - dy * 0.3
                self.move_arm_joint("shoulder", shoulder_adjustment)
                print(f"Adjusted shoulder to {shoulder_adjustment:.3f}")
                
            # Wait between adjustments
            for _ in range(5):
                self.sim.step_physics(1/60.0)
                
            if abs(dz) > 0.02:
                elbow_adjustment = current_positions.get("elbow", 0.0) + dz * 0.2
                self.move_arm_joint("elbow", elbow_adjustment)
                print(f"Adjusted elbow to {elbow_adjustment:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error in camera alignment: {e}")
            import traceback
            traceback.print_exc()
            return False

    def camera_guided_grasp(self):
        """Use camera feedback to guide book grasping."""
        print("Starting camera-guided grasp sequence")

        # Move to pre-grasp position
        self.move_to_pre_grasp_position()

        # Open gripper
        self.open_gripper()

        # Use camera to fine-tune alignment
        self._align_with_camera()

        # Lower arm to grasp position
        self.move_arm_joint("shoulder", 0.2)  # Lower shoulder
        self.move_arm_joint("elbow", 0.8)     # Extend elbow
        self.move_arm_joint("wrist_angle", 0.5)  # Tilt wrist down

        # Fine-tune alignment again now that we're closer
        self._align_with_camera()

        # Close gripper
        self.close_gripper()

        # Make book kinematic and attach
        if self.book_object is not None:
            try:
                self.book_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                self.attached_object = self.book_object
            except Exception as e:
                print(f"Error attaching book: {e}")

        # Lift the grasped object
        self.lift_object()

        print("Camera-guided grasp completed")
        return True
    
    def get_joint_position(self, locobot, link_id):
        """Get the current position of a joint."""
        pos_offset = locobot.get_link_joint_pos_offset(link_id)
        if pos_offset >= 0:
            joint_pos = locobot.joint_positions
            return joint_pos[pos_offset]
        return 0.0