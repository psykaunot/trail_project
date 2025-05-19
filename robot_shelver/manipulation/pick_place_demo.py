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

        # Check movement distance to adjust motor strength accordingly
        movement_distance = abs(current_pos - angle)
        
        # DRAMATICALLY increase gain and impulse values for much stronger movement
        position_gain = 100.0 if movement_distance > 0.5 else 50.0  # Much higher gain for responsive movement
        max_impulse = 10000.0  # Extreme force to overcome any constraints

        # Configure joint motor for position control with increased power
        self.motor_settings[joint_id] = habitat_sim.physics.JointMotorSettings(
            position_target=angle,
            position_gain=position_gain,  # Higher gain for faster movement
            velocity_target=0.0,
            velocity_gain=5.0,          # Higher damping for more stability
            max_impulse=max_impulse     # Much higher force to ensure arm can move against gravity
        )

        # Ensure the joint is not locked (force kinematic instead of fixed)
        try:
            # Enhanced joint type checking and unlocking
            joint_type = self.locobot.get_link_joint_type(joint_id)
            is_locked = joint_type == habitat_sim.physics.JointType.Fixed
            if is_locked:
                print(f"WARNING: Joint {joint_name} appears to be locked (type: {joint_type})")
                
                # Method 1: Try to directly change joint type
                try:
                    # Force joint type to revolute - this is the most direct way
                    if hasattr(self.locobot, 'set_link_joint_type'):
                        self.locobot.set_link_joint_type(joint_id, habitat_sim.physics.JointType.Revolute)
                        print(f"Successfully changed {joint_name} type to Revolute")
                        # Verify the change
                        joint_type = self.locobot.get_link_joint_type(joint_id)
                        if joint_type != habitat_sim.physics.JointType.Revolute:
                            print(f"ERROR: Failed to change joint type! Still: {joint_type}")
                except Exception as e:
                    print(f"Error changing joint type directly: {e}")
                
                # Method 2: Try setting position directly
                try:
                    joint_positions = self.locobot.joint_positions
                    joint_positions[pos_offset] = current_pos + 0.01  # Slightly change position to force update
                    self.locobot.joint_positions = joint_positions
                    print(f"Applied position change to try to unlock joint")
                except Exception as e:
                    print(f"Error updating joint position: {e}")
            
            # Force motion type to kinematic for all relevant links
            try:
                link_obj = self.locobot.get_link_object(joint_id)
                if link_obj and hasattr(link_obj, 'motion_type'):
                    original_type = link_obj.motion_type
                    link_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                    print(f"Set motion type to KINEMATIC (was: {original_type})")
            except Exception as e:
                print(f"Error setting motion type: {e}")
        except Exception as e:
            print(f"Error checking joint lock status: {e}")

        # Apply motor settings with force
        if joint_id in self.motor_ids:
            # Apply the settings multiple times to ensure they take effect
            for _ in range(3):
                self.locobot.update_joint_motor(self.motor_ids[joint_id], self.motor_settings[joint_id])
                self.sim.step_physics(1/200.0)  # Small physics step to apply settings
        else:
            print(f"WARNING: No motor ID for joint {joint_name}")
            return False

        # Force environment flags to ensure arm can move
        import sys
        sys.path.append("..")
        try:
            import environment
            environment._is_arm_moving = True
            environment._last_command_time = time.time()
        except ImportError:
            pass

        # Wait for movement to complete with UI updates
        if wait:
            iterations = 0
            last_pos = current_pos
            stalled_count = 0
            
            while iterations < max_iterations:
                # Step physics with smaller steps for smoother motion
                self.sim.step_physics(1/100.0)  # Faster physics updates
                
                # Check current position
                current_pos = self.locobot.joint_positions[pos_offset]
                
                # Check if we're making progress
                if abs(current_pos - last_pos) < 0.001:
                    stalled_count += 1
                    # If stalled for too long, increase motor power
                    if stalled_count > 5:
                        print(f"Joint {joint_name} appears stalled, increasing power")
                        # Dramatically increase motor power
                        self.motor_settings[joint_id].position_gain *= 2.0
                        self.motor_settings[joint_id].max_impulse *= 2.0
                        self.locobot.update_joint_motor(self.motor_ids[joint_id], self.motor_settings[joint_id])
                        stalled_count = 0
                else:
                    stalled_count = 0  # Reset stall counter if moving
                
                last_pos = current_pos
                
                # Check if we've reached the target
                if abs(current_pos - angle) < 0.1:  # Wider tolerance
                    print(f"Joint {joint_name} reached {current_pos:.2f} (target: {angle:.2f})")
                    return True
                
                iterations += 1
                # Allow main loop to update UI with smaller pause
                time.sleep(0.02)  # 20ms pause for faster response
                
                # Periodically re-apply motor settings to ensure movement continues
                if iterations % 5 == 0:
                    self.locobot.update_joint_motor(self.motor_ids[joint_id], self.motor_settings[joint_id])
            
            # If we timed out, try direct position setting as a fallback
            print(f"WARNING: Timeout moving joint {joint_name} to {angle:.2f}, got to {current_pos:.2f}")
            print(f"Attempting direct position set as fallback")
            try:
                joint_positions = self.locobot.joint_positions
                joint_positions[pos_offset] = angle
                self.locobot.joint_positions = joint_positions
            except Exception as e:
                print(f"Error in direct position set: {e}")

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
        """Release the book with enhanced position tracking."""
        print("Releasing object...")
        
        # Open gripper
        self.open_gripper()
        
        # If the book is a rigid object, detach it and make it dynamic
        if hasattr(self, 'attached_object') and self.attached_object is not None:
            try:
                # Record the placement position for memory updates
                placement_position = list(self.attached_object.translation)
                print(f"Recording book placement position: {placement_position}")
                
                # Switch back to dynamic physics
                self.attached_object.motion_type = habitat_sim.physics.MotionType.DYNAMIC
                
                # Track position change from original position to placement
                if hasattr(self, 'original_book_position') and self.original_book_position:
                    if placement_position:
                        try:
                            distance = np.linalg.norm(np.array(placement_position) - np.array(self.original_book_position))
                            print(f"Book moved {distance:.3f}m from original position")
                            
                            # Update semantic memory with placement information
                            self._update_memory_with_placement(self.original_book_position, placement_position)
                        except Exception as e:
                            print(f"Error calculating distance: {e}")
                
                # Clean up tracking
                self.attached_object = None
                
            except Exception as e:
                print(f"Error releasing object: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    def _update_memory_with_placement(self, pickup_position, placement_position):
        """Update semantic memory with book placement information."""
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from main import book_search_agent
            
            if hasattr(book_search_agent, 'book_memory'):
                memory = book_search_agent.book_memory
                
                # If memory system supports placement tracking
                if hasattr(memory, 'record_place_action') and callable(memory.record_place_action):
                    success = memory.record_place_action(pickup_position, placement_position)
                    print(f"Recorded book placement in memory: {'Success' if success else 'Failed'}")
                else:
                    # Fall back to using standard tracking methods
                    print("Memory system doesn't support placement tracking")
                    
                    # For future improvement: implement placement tracking in the memory system
                    
        except Exception as e:
            print(f"Warning: Could not update memory with placement information: {e}")
            import traceback
            traceback.print_exc()
    
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
        """Update the position of any attached object with enhanced tracking."""
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
                
                # Track position changes for debugging
                if not hasattr(self, 'position_history'):
                    self.position_history = []
                    
                # Only add position to history if it's significantly different
                current_pos = list(book_state.translation)
                if not self.position_history or self._position_distance(current_pos, self.position_history[-1]["position"]) > 0.05:
                    self.position_history.append({
                        "position": current_pos,
                        "timestamp": time.time()
                    })
                    
                    # Keep history at a reasonable size
                    if len(self.position_history) > 20:
                        self.position_history = self.position_history[-20:]
        except Exception as e:
            if hasattr(self, 'attach_error_reported') and self.attach_error_reported:
                pass  # Don't spam the console
            else:
                print(f"Error updating attached object: {e}")
                self.attach_error_reported = True
                
    def _position_distance(self, pos1, pos2):
        """Calculate distance between two positions."""
        if not pos1 or not pos2:
            return float('inf')
            
        try:
            return np.linalg.norm(np.array(pos1) - np.array(pos2))
        except Exception:
            return float('inf')

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
        """Use camera feedback to guide book grasping with improved precision."""
        print("\n========== DEBUGGING ARM MOVEMENT ==========")
        print("Starting enhanced camera-guided grasp sequence")
        
        # First, check if the arm can move at all by directly accessing _is_arm_moving
        import sys
        sys.path.append("..")
        try:
            from environment import _is_arm_moving
            print(f"CRITICAL: Current _is_arm_moving flag state: {_is_arm_moving}")
            # Force arm movement to be enabled
            import environment
            environment._is_arm_moving = True
            print(f"CRITICAL: Manually set _is_arm_moving flag to: {environment._is_arm_moving}")
            
            # Set global flag variables directly in the environment module
            # This is crucial for the RL picking policy in Habitat
            print("Setting ALL relevant environment flags to enable arm movement")
            environment._is_gripper_moving = True
            # Check if additional flags need to be set
            if hasattr(environment, '_exploration_active'):
                environment._exploration_active = False
                print(f"Set _exploration_active flag to: {environment._exploration_active}")
            
            # Reset any timeout counters that might be preventing arm movement
            if hasattr(environment, '_last_command_time'):
                environment._last_command_time = sys.float_info.max  # Force high value
                print("Reset command timeout counter")
        except ImportError:
            print("CRITICAL: Could not import environment module to check arm movement flag")
        except Exception as e:
            print(f"CRITICAL: Error accessing arm movement flag: {e}")
            import traceback
            traceback.print_exc()
        
        # Set the picking flag in the camera controller if available
        try:
            from main import camera_controller
            if camera_controller:
                try:
                    camera_controller.is_picking = True
                    print("DEBUG: Set is_picking flag to True in camera controller")
                    
                    # Also try to directly modify the environment's flags through the camera controller
                    if hasattr(camera_controller, '_camera_env') and hasattr(camera_controller._camera_env, '_is_arm_moving'):
                        camera_controller._camera_env._is_arm_moving = True
                        print("DEBUG: Set _is_arm_moving flag to True via camera controller")
                        
                except Exception as e:
                    print(f"Error setting picking flag: {e}")
            else:
                print("CRITICAL: camera_controller not available from main!")
        except Exception as e:
            print(f"CRITICAL: Error importing camera_controller from main: {e}")
            
        # Print debug for robot joints and motor settings
        try:
            print("\nDEBUGGING ROBOT JOINTS:")
            locked_joints = []
            for joint_name in self.arm_joints:
                if joint_name in self.dof_map:
                    joint_id = self.dof_map[joint_name]
                    pos_offset = self.locobot.get_link_joint_pos_offset(joint_id)
                    if pos_offset >= 0:
                        current_pos = self.locobot.joint_positions[pos_offset]
                        print(f"  {joint_name}: {current_pos:.4f}")
                        
                        # Check joint motor settings
                        if hasattr(self, 'motor_settings') and joint_id in self.motor_settings:
                            motor = self.motor_settings[joint_id]
                            print(f"    Motor settings - target: {motor.position_target:.4f}, gain: {motor.position_gain:.1f}, max impulse: {motor.max_impulse:.1f}")
                        
                        # Check if joint is locked
                        joint_type = self.locobot.get_link_joint_type(joint_id)
                        is_locked = joint_type == habitat_sim.physics.JointType.Fixed
                        if is_locked:
                            locked_joints.append(joint_name)
                            print(f"    *** WARNING: {joint_name} is LOCKED (type: {joint_type}) ***")
                        
            print("\nDEBUGGING GRIPPER JOINTS:")
            for joint_name in self.gripper_joints:
                if joint_name in self.dof_map:
                    joint_id = self.dof_map[joint_name]
                    pos_offset = self.locobot.get_link_joint_pos_offset(joint_id)
                    if pos_offset >= 0:
                        current_pos = self.locobot.joint_positions[pos_offset]
                        print(f"  {joint_name}: {current_pos:.4f}")
                        
                        # Check if joint is locked
                        joint_type = self.locobot.get_link_joint_type(joint_id)
                        is_locked = joint_type == habitat_sim.physics.JointType.Fixed
                        if is_locked:
                            locked_joints.append(joint_name)
                            print(f"    *** WARNING: {joint_name} is LOCKED (type: {joint_type}) ***")
            
            # Check joint velocity states
            print("\nDEBUGGING JOINT VELOCITIES:")
            for joint_name in self.arm_joints:
                if joint_name in self.dof_map:
                    joint_id = self.dof_map[joint_name]
                    vel_offset = self.locobot.get_link_joint_vel_offset(joint_id)
                    if vel_offset >= 0:
                        current_vel = self.locobot.joint_velocities[vel_offset]
                        print(f"  {joint_name} velocity: {current_vel:.4f}")
            
            # Attempt to unlock any locked joints
            if locked_joints:
                print("\nATTEMPTING TO UNLOCK LOCKED JOINTS:")
                for joint_name in locked_joints:
                    print(f"  Unlocking {joint_name}...")
                    joint_id = self.dof_map[joint_name]
                    
                    # Force joint type to revolute
                    try:
                        # Method 1: Try to directly change joint type
                        if hasattr(self.locobot, 'set_link_joint_type'):
                            self.locobot.set_link_joint_type(joint_id, habitat_sim.physics.JointType.Revolute)
                            print(f"    Successfully changed {joint_name} type to Revolute")
                    except Exception as e:
                        print(f"    Error changing joint type: {e}")
                    
                    # Method 2: Modify joint motor settings with extreme values
                    try:
                        # Set extremely high motor force
                        if joint_id in self.motor_ids:
                            motor_settings = habitat_sim.physics.JointMotorSettings(
                                position_target=0.0,
                                position_gain=10000.0,    # Extreme position gain
                                velocity_target=0.0,
                                velocity_gain=500.0,      # High damping
                                max_impulse=100000.0      # Extremely high force
                            )
                            self.locobot.update_joint_motor(self.motor_ids[joint_id], motor_settings)
                            print(f"    Applied extreme motor settings to {joint_name}")
                            
                            # Step physics to apply settings
                            for _ in range(5):
                                self.sim.step_physics(1/60.0)
                    except Exception as e:
                        print(f"    Error applying motor settings: {e}")
                        
                    # Method 3: Try direct position modification
                    try:
                        pos_offset = self.locobot.get_link_joint_pos_offset(joint_id)
                        if pos_offset >= 0:
                            # Get current position
                            positions = self.locobot.joint_positions
                            current_pos = positions[pos_offset]
                            
                            # Slightly modify position to force update
                            new_pos = current_pos + 0.01
                            positions[pos_offset] = new_pos
                            self.locobot.joint_positions = positions
                            print(f"    Forced position change for {joint_name} from {current_pos:.4f} to {new_pos:.4f}")
                            
                            # Step physics to apply change
                            for _ in range(5):
                                self.sim.step_physics(1/60.0)
                    except Exception as e:
                        print(f"    Error forcing position change: {e}")
            
            # Verify if unlocking was successful
            print("\nVERIFYING JOINT STATUS AFTER UNLOCKING:")
            for joint_name in self.arm_joints + self.gripper_joints:
                if joint_name in self.dof_map:
                    joint_id = self.dof_map[joint_name]
                    joint_type = self.locobot.get_link_joint_type(joint_id)
                    is_locked = joint_type == habitat_sim.physics.JointType.Fixed
                    print(f"  {joint_name}: {'STILL LOCKED' if is_locked else 'UNLOCKED/MOVABLE'} (type: {joint_type})")
                    
        except Exception as e:
            print(f"Error debugging and unlocking robot joints: {e}")
            import traceback
            traceback.print_exc()
                
        print("============================================\n")

        # Check if book object is available
        if self.book_object is None:
            print("WARNING: No book object provided for camera-guided grasp")
            print("Attempting to search for books in the scene...")
            
            # Try to find any book object in the scene
            self._find_book_object()
            
            if self.book_object is None:
                print("ERROR: Could not find any book object for grasping")
                return False
        else:
            print(f"Book object provided at position: {self.book_object.translation}")
            
        # Print book object properties
        try:
            print(f"Book object ID: {self.book_object.object_id}")
            print(f"Book object handle: {self.book_object.handle}")
            print(f"Book object motion type: {self.book_object.motion_type}")
        except Exception as e:
            print(f"Error accessing book object properties: {e}")
            
        # Move arm carefully to pre-grasp position
        print("Moving to pre-grasp position...")
        
        # First go to a safe position
        try:
            # Make movements smaller and more careful
            print("Moving to safe initial position...")
            self.move_arm_joint("waist", 0.0)  # Center waist
            time.sleep(0.1)
            
            self.move_arm_joint("shoulder", 0.0)  # Lower shoulder 
            time.sleep(0.1)
            
            # Then move to pre-grasp in steps
            self.move_to_pre_grasp_position()
        except Exception as e:
            print(f"Error in pre-grasp positioning: {e}")
            import traceback
            traceback.print_exc()
            
        # Open gripper
        self.open_gripper()
        time.sleep(0.1)

        # Move toward book position
        try:
            # Get book position and orient arm toward it
            book_pos = self.book_object.translation
            robot_pos = self.locobot.translation
            
            # Calculate direction to book
            dir_x = book_pos[0] - robot_pos[0]
            dir_z = book_pos[2] - robot_pos[2]
            
            # Calculate waist angle to face book
            target_waist = np.arctan2(dir_x, dir_z)
            print(f"Rotating waist to {target_waist:.2f} to face book")
            self.move_arm_joint("waist", target_waist, wait=True)
            time.sleep(0.2)
        except Exception as e:
            print(f"Error orienting toward book: {e}")
            
        # Use camera to fine-tune alignment
        try:
            alignment_success = self._align_with_camera()
            if not alignment_success:
                print("Camera alignment failed, trying fallback position")
                # Try a fallback position
                self.move_arm_joint("waist", 0.0)  # Center waist
                self.move_arm_joint("shoulder", 0.4)  # Position shoulder
                self.move_arm_joint("elbow", 0.7)  # Position elbow
        except Exception as e:
            print(f"Error in camera alignment: {e}")
            import traceback
            traceback.print_exc()
        
        # Lower arm to grasp position with careful movements
        print("Moving to grasp position...")
        try:
            self.move_arm_joint("shoulder", 0.2, wait=True)  # Lower shoulder
            self.sim.step_physics(0.05)  # Small pause
            time.sleep(0.2)
            
            self.move_arm_joint("elbow", 0.8, wait=True)     # Extend elbow 
            self.sim.step_physics(0.05)  # Small pause
            time.sleep(0.2)
            
            self.move_arm_joint("wrist_angle", 0.5, wait=True)  # Tilt wrist down
            time.sleep(0.2)
        except Exception as e:
            print(f"Error moving to grasp position: {e}")

        # Step physics to stabilize
        for _ in range(10):
            self.sim.step_physics(1/60.0)
            
        # Close gripper with increased force
        print("Closing gripper...")
        self.close_gripper()
        
        # Wait for gripper to fully close
        for _ in range(20):
            self.sim.step_physics(1/60.0)
            time.sleep(0.02)  # Allow UI updates

        # Make book kinematic and attach
        attach_success = False
        if self.book_object is not None:
            try:
                print("Attaching book to gripper...")
                # Save the original motion type
                original_motion_type = self.book_object.motion_type
                
                # Make book kinematic for manipulation
                self.book_object.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                self.attached_object = self.book_object
                
                # Log original position for memory tracking
                self.original_book_position = list(self.book_object.translation)
                print(f"Recorded original book position: {self.original_book_position}")
                
                # Check if attachment was successful
                wrist_link_id = self.dof_map.get("wrist_angle", -1)
                if wrist_link_id != -1:
                    wrist_node = self.locobot.get_link_scene_node(wrist_link_id)
                    book_pos = self.book_object.translation
                    distance = np.linalg.norm(np.array([
                        wrist_node.translation[0] - book_pos[0],
                        wrist_node.translation[1] - book_pos[1],
                        wrist_node.translation[2] - book_pos[2]
                    ]))
                    print(f"Book distance from wrist: {distance:.3f} m")
                    
                    # Adjust book position if needed
                    if distance > 0.3:
                        print("Warning: Book appears to be far from gripper, adjusting...")
                        # Update book position to be closer to gripper
                        book_state = self.book_object.rigid_state
                        book_state.translation = wrist_node.translation - mn.Vector3(0, 0.05, 0)
                        self.book_object.rigid_state = book_state
                        
                        # Update the adjusted position in our tracking
                        self.original_book_position = list(book_state.translation)
                        print(f"Updated book pickup position: {self.original_book_position}")
                        
                    # If close enough, consider attachment successful
                    attach_success = True
            except Exception as e:
                print(f"Error attaching book: {e}")
                import traceback
                traceback.print_exc()

        # Lift the grasped object with more careful movements
        if attach_success:
            print("Lifting object...")
            try:
                self.move_arm_joint("shoulder", 0.4, wait=True)  # Raise shoulder a bit
                self.sim.step_physics(0.1)  # Longer pause
                time.sleep(0.2)
                
                self.move_arm_joint("elbow", 1.0, wait=True)  # Adjust elbow
                self.sim.step_physics(0.1)  # Longer pause
                time.sleep(0.2)
                
                self.move_arm_joint("shoulder", 0.7, wait=True)  # Raise shoulder more
                time.sleep(0.2)
                
                # Update attached object position
                self.update_attached_object()
            except Exception as e:
                print(f"Error lifting object: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Did not lift object because attachment was not successful")
        
        # Update semantic memory if available with enhanced tracking
        try:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from main import book_search_agent
            
            if hasattr(book_search_agent, 'book_memory'):
                # Record pick action in memory
                memory = book_search_agent.book_memory
                if hasattr(memory, 'record_pick_action') and callable(memory.record_pick_action):
                    # Use original book position if available, current position otherwise
                    position = None
                    if hasattr(self, 'original_book_position') and self.original_book_position:
                        position = self.original_book_position
                        print(f"Using original book position for memory: {position}")
                    else:
                        position = self.book_object.translation if self.book_object else None
                        print(f"Using current book position for memory: {position}")
                        
                    if position:
                        # Convert position to list if it's not already
                        if not isinstance(position, list):
                            position = list(position)
                            
                        success = memory.record_pick_action(position)
                        print(f"Recorded pick action in memory system: {'Success' if success else 'Failed'}")
                        
                        # Add additional debug info for book matching
                        if hasattr(memory, 'observation_stats'):
                            print(f"Memory stats: {memory.observation_stats['unique_books']} unique books, "
                                  f"{memory.observation_stats['duplicate_matches']} duplicates")
        except Exception as e:
            print(f"Warning: Could not update memory with pick action: {e}")
            import traceback
            traceback.print_exc()

        print("Enhanced camera-guided grasp completed")
        
        # Reset the picking flag in the camera controller
        from main import camera_controller
        if camera_controller:
            try:
                camera_controller.is_picking = False
                print("DEBUG: Reset is_picking flag to False in camera controller")
            except Exception as e:
                print(f"Error resetting picking flag: {e}")
        
        # Return success based on attachment
        return attach_success
    
    def get_joint_position(self, locobot, link_id):
        """Get the current position of a joint."""
        pos_offset = locobot.get_link_joint_pos_offset(link_id)
        if pos_offset >= 0:
            joint_pos = locobot.joint_positions
            return joint_pos[pos_offset]
        return 0.0