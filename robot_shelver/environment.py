#!/usr/bin/env python3
import os
import sys
import numpy as np
import magnum as mn
from magnum import Rad
import cv2
import time
import traceback
import math
import threading
import habitat_sim
import habitat_sim.gfx
import habitat_sim.physics as phys
from camera.camera_controller import CameraController  
from camera.camera_stabilizer import CameraStabilizer

# Resolve data paths relative to this file
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
# Globals for drift prevention
_reference_position = None
_reference_orientation = None
_last_command_time = 0
_is_arm_moving = False
_is_gripper_moving = False
_camera_reference_transforms = {}
_joint_limits_cache = {}  # Cache for joint limits to avoid repeated calls
_debug_mode = False  # Set to False to disable detailed debugging output
_exploration_active = False
_position_warnings_enabled = False  # For position constraint warnings

# Define CORRECT gripper limits globally so they can be used in multiple functions
LEFT_FINGER_MIN = 0.015
LEFT_FINGER_MAX = 0.037
RIGHT_FINGER_MIN = -0.037
RIGHT_FINGER_MAX = -0.015

# Define safe gripper positions within these limits
SAFE_LEFT_OPEN = 0.037     # Left finger fully open position
SAFE_LEFT_CLOSED = 0.015   # Left finger fully closed position
SAFE_RIGHT_OPEN = -SAFE_LEFT_OPEN   # Right finger fully open position
SAFE_RIGHT_CLOSED = -SAFE_LEFT_CLOSED # Right finger fully closed position


# Arm rest positions for consistent positioning
ARM_REST_POSITIONS = {
    "waist": 0.0,
    "shoulder": -1.5,
    "elbow": 1.5,
    "forearm_roll": 0.0,
    "wrist_angle": 1.0,
    "wrist_rotate": 0.0,
}


def set_exploration_active(active=True):
    """Set the exploration mode flag."""
    global _exploration_active
    _exploration_active = active
    print(f"Exploration mode: {'ACTIVE' if active else 'INACTIVE'}")
    
    # Also update other flags for consistent behavior
    global _is_arm_moving
    if active:
        # During exploration, arm should be at rest
        _is_arm_moving = False
    else:
        # When not exploring, allow arm movement
        _is_arm_moving = True

# Utility: create a color camera sensor specification
def make_cam(name, pos, ori):
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = name
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.resolution = [480, 640]
    cam.position = mn.Vector3(*pos)
    cam.orientation = mn.Vector3(*ori)
    return cam


def unlock_arm_for_picking(force=True):
    """Force unlock arm joints for picking operations"""
    global _is_arm_moving, _is_gripper_moving, _exploration_active, _last_command_time
    
    # Force flags to allow movement
    _is_arm_moving = True
    _is_gripper_moving = True
    _exploration_active = False
    _last_command_time = time.time()
    
    print(f"ARM UNLOCKED FOR PICKING: _is_arm_moving={_is_arm_moving}, _is_gripper_moving={_is_gripper_moving}")
    return True


def fix_ar_tag_position(locobot):
    """Force AR tag to maintain fixed position relative to parent with improved stability."""
    try:
        # Get link IDs
        ar_tag_link_id = locobot.get_link_id_from_name("locobot/ar_tag_link")
        parent_link_id = locobot.get_link_id_from_name("locobot/ee_arm_link")
        
        if ar_tag_link_id == -1 or parent_link_id == -1:
            return
            
        # Get nodes
        parent_node = locobot.get_link_scene_node(parent_link_id)
        ar_tag_node = locobot.get_link_scene_node(ar_tag_link_id)
        
        # Make AR tag kinematic (not affected by physics)
        try:
            # Force kinematic motion type
            ar_tag_obj = locobot.get_link_object(ar_tag_link_id)
            if ar_tag_obj:
                ar_tag_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        except:
            pass
        
        # Exact fixed position offset - measured for stability
        fixed_local_pos = mn.Vector3(-0.05, 0.0, 0.03)
        
        # Force position and rotation directly
        ar_tag_node.translation = parent_node.translation + parent_node.rotation.transform_vector(fixed_local_pos)
        ar_tag_node.rotation = parent_node.rotation
        
        # Set extremely high friction 
        locobot.set_link_friction(ar_tag_link_id, 1000.0)
    except Exception as e:
        print(f"Error stabilizing AR tag: {e}")

def stabilize_gripper_prop(locobot, dof_map, motor_ids, motor_settings):
    """Force proper positioning for gripper prop only."""
    try:
        # Stabilize gripper prop by locking its rotation
        if "gripper" in dof_map:
            gripper_id = dof_map["gripper"]
            # Only proceed if gripper_id exists in motor_ids or we can create it
            if gripper_id not in motor_ids:
                # Try to create the motor
                motor_settings[gripper_id] = habitat_sim.physics.JointMotorSettings(
                    position_target=0.0,
                    position_gain=1000.0,
                    velocity_target=0.0,
                    velocity_gain=200.0,
                    max_impulse=1000.0
                )
                try:
                    motor_id = locobot.create_joint_motor(gripper_id, motor_settings[gripper_id])
                    motor_ids[gripper_id] = motor_id
                    print(f"Created new motor for gripper joint (id: {gripper_id})")
                except Exception as e:
                    print(f"Could not create motor for gripper joint: {e}")
                    return
            
            # Now we can safely update the motor if it exists
            if gripper_id in motor_ids:
                pos_offset = locobot.get_link_joint_pos_offset(gripper_id)
                if pos_offset >= 0:
                    # Force exact neutral position
                    joint_positions = locobot.joint_positions
                    joint_positions[pos_offset] = 0.0
                    locobot.joint_positions = joint_positions
                    
                    # Update motor settings
                    locobot.update_joint_motor(motor_ids[gripper_id], motor_settings[gripper_id])
        
    except Exception as e:
        print(f"Error stabilizing gripper prop: {e}")


def get_joint_position(locobot, link_id, check_limits=False):
    """Get the current position of a joint by link ID with enhanced limit checking."""
    try:
        pos_offset = locobot.get_link_joint_pos_offset(link_id)
        if pos_offset >= 0:
            joint_pos = locobot.joint_positions
            position = joint_pos[pos_offset]
            
            # Only perform limit checking when requested
            if check_limits:
                # Get actual URDF limits for this joint
                joint_limits = locobot.joint_position_limits
                lower_limit = joint_limits[0][pos_offset]
                upper_limit = joint_limits[1][pos_offset]
                
                if position < lower_limit or position > upper_limit:
                    joint_name = locobot.get_link_joint_name(link_id)
                    print(f"WARNING: Joint {joint_name} position {position:.4f} is outside URDF limits [{lower_limit:.4f}, {upper_limit:.4f}]")
                    
                    # Log configured vs actual limits 
                    if joint_name == "left_finger":
                        print(f"  Configured limits: [{LEFT_FINGER_MIN:.4f}, {LEFT_FINGER_MAX:.4f}]")
                    elif joint_name == "right_finger":
                        print(f"  Configured limits: [{RIGHT_FINGER_MIN:.4f}, {RIGHT_FINGER_MAX:.4f}]")
            
            return position
        return 0.0
    except Exception as e:
        if _debug_mode:
            print(f"Error getting joint position for link {link_id}: {e}")
            traceback.print_exc()
        return 0.0

# Get joint limits for a link
def get_joint_limits(locobot, link_id):
    """Get the limits for a joint by link ID."""
    try:
        if link_id in _joint_limits_cache:
            # Use cached values if available
            return _joint_limits_cache[link_id]
        
        # Otherwise compute the limits
        pos_offset = locobot.get_link_joint_pos_offset(link_id)
        dofs = locobot.get_link_num_dofs(link_id)
        
        if pos_offset >= 0 and dofs > 0:
            joint_limits = locobot.joint_position_limits
            lower = joint_limits[0][pos_offset:pos_offset+dofs]
            upper = joint_limits[1][pos_offset:pos_offset+dofs]
            # Cache for future calls
            _joint_limits_cache[link_id] = (lower, upper)
            return lower, upper
        
        # Default if no limits are found
        return [-float('inf')], [float('inf')]
    except Exception as e:
        if _debug_mode:
            print(f"Error getting joint limits for link {link_id}: {e}")
            traceback.print_exc()
        return [-float('inf')], [float('inf')]
    

def set_arm_to_rest_position(locobot, motor_ids, motor_settings, dof_map, sim=None, debug=False):
    """
    Move the robot arm to a compact rest position that doesn't block camera view.
    This position folds the arm close to the body.
    """
    arm_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    
    # Set targets for each joint with high force
    for joint_name in arm_joints:
        if joint_name not in dof_map:
            continue
            
        joint_id = dof_map[joint_name]
        target_position = ARM_REST_POSITIONS[joint_name]
        
        # Get current position
        pos_offset = locobot.get_link_joint_pos_offset(joint_id)
        
        if pos_offset >= 0:
            # Configure motor with high forces
            motor_settings[joint_id] = habitat_sim.physics.JointMotorSettings(
                position_target=target_position,
                position_gain=500.0,
                velocity_target=0.0,
                velocity_gain=100.0,
                max_impulse=3000.0
            )
            
            # Apply motor settings
            if joint_id in motor_ids:
                locobot.update_joint_motor(motor_ids[joint_id], motor_settings[joint_id])
    
    # Let physics settle with the new positions
    if sim is not None:
        convergence_threshold = 0.05
        max_attempts = 200
        
        for i in range(max_attempts):
            # Step physics
            sim.step_physics(1.0/60.0)
            
            # Force position updates for stubborn joints
            all_converged = True
            for joint_name in arm_joints:
                if joint_name not in dof_map:
                    continue
                    
                joint_id = dof_map[joint_name]
                pos_offset = locobot.get_link_joint_pos_offset(joint_id)
                
                if pos_offset >= 0:
                    current_pos = locobot.joint_positions[pos_offset]
                    target_pos = ARM_REST_POSITIONS[joint_name]
                    error = abs(current_pos - target_pos)
                    
                    if error > convergence_threshold:
                        all_converged = False
                        
                        # Force position if physics isn't converging
                        if i > 50 and error > 0.2:
                            joint_positions = locobot.joint_positions
                            # Gradual forced movement
                            new_pos = current_pos + (target_pos - current_pos) * 0.1
                            joint_positions[pos_offset] = new_pos
                            locobot.joint_positions = joint_positions
            
            if all_converged:
                break
    
    # Final forced position set
    for joint_name in arm_joints:
        if joint_name not in dof_map:
            continue
            
        joint_id = dof_map[joint_name]
        target_position = ARM_REST_POSITIONS[joint_name]
        pos_offset = locobot.get_link_joint_pos_offset(joint_id)
        
        if pos_offset >= 0:
            joint_positions = locobot.joint_positions
            joint_positions[pos_offset] = target_position
            locobot.joint_positions = joint_positions
    
    # Lock in position with extreme stiffness
    for joint_name in arm_joints:
        if joint_name not in dof_map:
            continue
            
        joint_id = dof_map[joint_name]
        target_position = ARM_REST_POSITIONS[joint_name]
        
        motor_settings[joint_id] = habitat_sim.physics.JointMotorSettings(
            position_target=target_position,
            position_gain=5000.0,
            velocity_target=0.0,
            velocity_gain=500.0,
            max_impulse=10000.0
        )
        
        if joint_id in motor_ids:
            locobot.update_joint_motor(motor_ids[joint_id], motor_settings[joint_id])
    
    return True

def hold_arm_at_rest(locobot, motor_ids, motor_settings, dof_map):
    """
    Maintain arm at rest position during scanning operations.
    Call this periodically to ensure arm doesn't drift.
    """
    arm_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    
    for joint_name in arm_joints:
        if joint_name not in dof_map:
            continue
            
        joint_id = dof_map[joint_name]
        target_position = ARM_REST_POSITIONS[joint_name]
        
        # Force exact position
        pos_offset = locobot.get_link_joint_pos_offset(joint_id)
        if pos_offset >= 0:
            joint_positions = locobot.joint_positions
            current_pos = joint_positions[pos_offset]
            
            # Check if correction needed
            if abs(current_pos - target_position) > 0.05:
                joint_positions[pos_offset] = target_position
                locobot.joint_positions = joint_positions
                
                # Reinforce with motor command
                if joint_id in motor_ids:
                    motor_settings[joint_id].position_target = target_position
                    locobot.update_joint_motor(motor_ids[joint_id], motor_settings[joint_id])

# Helper function to check if key is for arm or gripper command
def is_arm_or_gripper_command(key):
    arm_keys = [
        ord('u'), ord('j'),  # waist
        ord('i'), ord('k'),  # shoulder
        ord('o'), ord('l'),  # elbow
        ord('p'), ord(';'),  # forearm_roll
        ord('['), ord(']'),  # wrist_angle
        ord('n'), ord('m'),  # wrist_rotate
    ]
    gripper_keys = [ord('g'), ord('h')]
    return key in arm_keys or key in gripper_keys

# Helper function to check if key is for camera pan/tilt movement
def is_camera_movement_command(key):
    camera_keys = [ord('c'), ord('v'), ord('f'), ord('r')]
    return key in camera_keys

# Add to environment.py
def visualize_gripper_state(combined_image, locobot, dof_map):
    """Add visual indicators of gripper state to the display image."""
    if not isinstance(combined_image, np.ndarray) or combined_image.size == 0:
        return combined_image
        
    h, w = combined_image.shape[:2]
    left_id = dof_map.get("left_finger", -1)
    right_id = dof_map.get("right_finger", -1)
    
    if left_id == -1 or right_id == -1:
        return combined_image
        
    left_pos = get_joint_position(locobot, left_id, check_limits=True)
    right_pos = get_joint_position(locobot, right_id, check_limits=True)
    
    # Create visualization overlay
    overlay = np.zeros((100, w, 3), dtype=np.uint8)
    
    # Draw left finger position
    left_norm = (left_pos - LEFT_FINGER_MIN) / (LEFT_FINGER_MAX - LEFT_FINGER_MIN)
    left_x = int(w * 0.25 * left_norm) + int(w * 0.25)
    cv2.line(overlay, (int(w * 0.25), 30), (int(w * 0.5), 30), (100, 100, 100), 2)
    cv2.circle(overlay, (left_x, 30), 5, (0, 255, 0), -1)
    cv2.putText(overlay, f"Left: {left_pos:.4f}", (int(w * 0.25), 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
    # Draw right finger position
    right_norm = (right_pos - RIGHT_FINGER_MIN) / (RIGHT_FINGER_MAX - RIGHT_FINGER_MIN)
    right_x = int(w * 0.25 * right_norm) + int(w * 0.5)
    cv2.line(overlay, (int(w * 0.5), 80), (int(w * 0.75), 80), (100, 100, 100), 2)
    cv2.circle(overlay, (right_x, 80), 5, (0, 255, 0), -1)
    cv2.putText(overlay, f"Right: {right_pos:.4f}", (int(w * 0.5), 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Combine with original image
    result = np.vstack([overlay, combined_image])
    return result

def set_gripper_state(locobot, motor_ids, motor_settings, dof_map, state="open"):
    """Set the gripper to a specific state with improved physics stability."""
    if "left_finger" not in dof_map or "right_finger" not in dof_map:
        print("Warning: Gripper joints not found in dof_map")
        return
    
    left_id = dof_map["left_finger"]
    right_id = dof_map["right_finger"]
    
    # Verify current URDF limits first
    left_pos_offset = locobot.get_link_joint_pos_offset(left_id)
    right_pos_offset = locobot.get_link_joint_pos_offset(right_id)
    joint_limits = locobot.joint_position_limits
    
    if left_pos_offset >= 0 and right_pos_offset >= 0:
        left_lower = joint_limits[0][left_pos_offset]
        left_upper = joint_limits[1][left_pos_offset]
        right_lower = joint_limits[0][right_pos_offset]
        right_upper = joint_limits[1][right_pos_offset]
        
        print(f"URDF LEFT FINGER LIMITS: [{left_lower:.4f}, {left_upper:.4f}]")
        print(f"URDF RIGHT FINGER LIMITS: [{right_lower:.4f}, {right_upper:.4f}]")
        print(f"CONFIGURED LEFT LIMITS: [{LEFT_FINGER_MIN:.4f}, {LEFT_FINGER_MAX:.4f}]")
        print(f"CONFIGURED RIGHT LIMITS: [{RIGHT_FINGER_MIN:.4f}, {RIGHT_FINGER_MAX:.4f}]")
    
    # Set target positions based on state
    if state == "open":
        # Use 95% of max range to avoid hitting limits
        left_target = min(LEFT_FINGER_MAX * 0.95, left_upper * 0.95 if 'left_upper' in locals() else LEFT_FINGER_MAX * 0.95)
        right_target = max(RIGHT_FINGER_MIN * 0.95, right_lower * 0.95 if 'right_lower' in locals() else RIGHT_FINGER_MIN * 0.95)
    else:  # closed
        left_target = max(LEFT_FINGER_MIN * 1.05, left_lower * 1.05 if 'left_lower' in locals() else LEFT_FINGER_MIN * 1.05)
        right_target = min(RIGHT_FINGER_MAX * 1.05, right_upper * 1.05 if 'right_upper' in locals() else RIGHT_FINGER_MAX * 1.05)
    
    print(f"Setting gripper to {state}: left={left_target:.4f}, right={right_target:.4f}")
    
    # Create very stiff position control settings
    motor_settings[left_id] = phys.JointMotorSettings(
        position_target=left_target,
        position_gain=600.0,        # Very high position stiffness
        velocity_target=0.0,
        velocity_gain=120.0,        # High damping
        max_impulse=1000.0          # Strong force limit
    )
    
    motor_settings[right_id] = phys.JointMotorSettings(
        position_target=right_target,
        position_gain=600.0,        # Very high position stiffness
        velocity_target=0.0,
        velocity_gain=120.0,        # High damping
        max_impulse=1000.0          # Strong force limit
    )
    
    # Apply motor settings immediately
    locobot.update_joint_motor(motor_ids[left_id], motor_settings[left_id])
    locobot.update_joint_motor(motor_ids[right_id], motor_settings[right_id])

# Setup the Habitat simulator and LoCoBot with physics tuning
# Add to environment.py
def move_robot_for_exploration(sim, locobot, new_pos, max_attempts=3):
    """Move robot with proper collision detection for exploration mode."""
    global _reference_position
    
    # First check if position is valid using path finding
    pathfinder = sim.pathfinder
    is_navigable = pathfinder.is_navigable(mn.Vector3(new_pos[0], new_pos[1], new_pos[2]))
    
    if not is_navigable:
        print(f"Target position {new_pos} is not navigable!")
        return False
    
    # Get current position
    current_pos = locobot.translation
    
    # Create a ray to check for obstacles
    direction = mn.Vector3(new_pos[0] - current_pos[0], 0, new_pos[2] - current_pos[2])
    distance = direction.length()
    
    if distance > 0:
        direction = direction / distance
        ray = habitat_sim.geo.Ray(current_pos, direction)
        raycast_results = sim.cast_ray(ray, distance * 1.1)
        
        if raycast_results.has_hits():
            print(f"Obstacle detected in path!")
            return False
    
    # Position is valid, update both the robot and reference position
    try:
        # Set position directly with physics disabled temporarily
        state = locobot.rigid_state
        state.translation = mn.Vector3(new_pos[0], new_pos[1], new_pos[2])
        locobot.rigid_state = state
        
        # CRITICAL: Update reference position so it doesn't get reset
        _reference_position = mn.Vector3(new_pos[0], new_pos[1], new_pos[2])
        
        # Zero velocities
        locobot.root_linear_velocity = mn.Vector3(0, 0, 0)
        locobot.root_angular_velocity = mn.Vector3(0, 0, 0)
        
        return True
    except Exception as e:
        print(f"Error moving robot: {e}")
        return False
    
def calculate_wheel_velocities(dx, dz, speed_factor=1.0):
    """Calculate differential wheel velocities for a desired movement direction."""
    # Normalize direction vector
    magnitude = math.sqrt(dx*dx + dz*dz)
    if magnitude < 1e-6:
        return 0.0, 0.0  # No movement
    
    dx_norm = dx / magnitude
    dz_norm = dz / magnitude
    
    # Base speed with scaling factor
    base_speed = 1.0 * speed_factor
    
    # For pure forward/backward motion, both wheels same direction
    if abs(dx_norm) < 0.1:
        # Forward or backward
        direction = -1 if dz_norm < 0 else 1  # Negate because forward is -z in habitat
        return base_speed * direction, base_speed * direction
        
    # For pure left/right rotation, wheels opposite direction
    if abs(dz_norm) < 0.1:
        # Left or right
        direction = 1 if dx_norm > 0 else -1  # Positive dx is right
        return base_speed * direction, -base_speed * direction
    
    # For diagonal movement, calculate differential steering
    left_factor = -dz_norm + dx_norm * 0.5  # Forward + partial turning component
    right_factor = -dz_norm - dx_norm * 0.5  # Forward - partial turning component
    
    # Scale to maintain consistent overall speed
    max_factor = max(abs(left_factor), abs(right_factor))
    if max_factor > 0:
        left_factor = left_factor / max_factor * base_speed
        right_factor = right_factor / max_factor * base_speed
    
    return left_factor, right_factor

def apply_smooth_movement(locobot, target_position, speed_factor=0.5, duration=1.0, steps=20):
    """Apply gradual movement toward target position using physics."""
    current_pos = locobot.translation
    
    # Calculate direction vector
    direction = mn.Vector3(
        target_position[0] - current_pos[0],
        0.0,  # Keep y at floor level
        target_position[2] - current_pos[2]
    )
    
    # Skip if no meaningful movement
    distance = direction.length()
    if distance < 0.001:
        return
        
    # Normalize direction and apply speed
    direction = direction / distance
    velocity = direction * speed_factor
    
    # Apply velocity
    locobot.root_linear_velocity = velocity
    
    # Step physics gradually
    step_size = duration / steps
    for i in range(steps):
        time.sleep(step_size)

# Add to environment.py

def move_with_physics(sim, locobot, motor_ids, motor_settings, dof_map, direction, speed=0.5, duration=1.0):
    """
    Move robot using physics and wheel motors in the specified direction.
    
    Args:
        direction: "forward", "backward", "left", "right", or [dx, dz] vector
        speed: Movement speed (0.0-1.0)
        duration: How long to apply the movement
    """
    # Calculate wheel velocities based on direction
    left_speed = 0.0
    right_speed = 0.0
    
    if direction == "forward":
        left_speed = speed
        right_speed = speed
    elif direction == "backward":
        left_speed = -speed
        right_speed = -speed
    elif direction == "left":
        left_speed = -speed
        right_speed = speed
    elif direction == "right":
        left_speed = speed
        right_speed = -speed
    elif isinstance(direction, (list, tuple)) and len(direction) == 2:
        # Convert [dx, dz] vector to differential drive
        dx, dz = direction
        left_speed, right_speed = calculate_wheel_velocities(dx, dz, speed)
    
    # Apply wheel velocities
    if "wheel_left_joint" in dof_map and "wheel_right_joint" in dof_map:
        motor_settings[dof_map["wheel_left_joint"]].velocity_target = left_speed
        motor_settings[dof_map["wheel_right_joint"]].velocity_target = right_speed
        
        # Update motors
        locobot.update_joint_motor(motor_ids[dof_map["wheel_left_joint"]], 
                                   motor_settings[dof_map["wheel_left_joint"]])
        locobot.update_joint_motor(motor_ids[dof_map["wheel_right_joint"]], 
                                   motor_settings[dof_map["wheel_right_joint"]])
    
    # Run physics for the specified duration
    steps = int(duration * 30)  # 30 steps per second
    dt = duration / steps
    
    for _ in range(steps):
        sim.step_physics(dt)
    
    # Stop wheels
    motor_settings[dof_map["wheel_left_joint"]].velocity_target = 0.0
    motor_settings[dof_map["wheel_right_joint"]].velocity_target = 0.0
    locobot.update_joint_motor(motor_ids[dof_map["wheel_left_joint"]], 
                               motor_settings[dof_map["wheel_left_joint"]])
    locobot.update_joint_motor(motor_ids[dof_map["wheel_right_joint"]], 
                               motor_settings[dof_map["wheel_right_joint"]])


def explore_with_collision_avoidance(sim, locobot, motor_ids, motor_settings, dof_map, move_distance=0.15):
    """Move robot with collision avoidance using physics-based movement."""
    global _reference_position, _exploration_active
    
    # Check if exploration is active
    if not _exploration_active:
        return False
    
    # Get current position
    current_pos = locobot.translation
    print(f"Exploring from position: {current_pos}")
    
    # Try different directions (45 degrees apart)
    for i in range(8):
        # Calculate angle and direction
        angle = i * (math.pi/4)
        dx = math.cos(angle)
        dz = math.sin(angle)
        
        # Calculate target position for checking
        target_pos = [
            current_pos[0] + dx * move_distance,
            0.0,
            current_pos[2] + dz * move_distance
        ]
        
        # Check if navigable
        is_navigable = sim.pathfinder.is_navigable(mn.Vector3(*target_pos))
        if not is_navigable:
            continue
            
        # Check for obstacles with raycast
        direction = mn.Vector3(dx, 0, dz).normalized()
        ray = habitat_sim.geo.Ray(current_pos, direction)
        hit_results = sim.cast_ray(ray, move_distance * 1.2)
        
        if hit_results.has_hits():
            continue
            
        print(f"Found valid direction - moving using physics")
        
        # Apply physics-based movement using wheel motors
        move_with_physics(sim, locobot, motor_ids, motor_settings, dof_map, 
                          [dx, dz], speed=0.4, duration=1.5)
        
        # Update reference position after movement
        _reference_position = locobot.translation
        return True
        
    print("Could not find valid movement direction")
    return False

def initialize_navmesh(sim):
    """Initialize the NavMesh for the current scene for better navigation."""
    
    # Configure NavMesh parameters
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    
    # Adjust settings for LoCoBot
    navmesh_settings.agent_radius = 0.3  # Match robot's physical radius
    navmesh_settings.agent_height = 1.5  # Match robot's height
    navmesh_settings.include_static_objects = True
    navmesh_settings.cell_size = 0.05  # Higher detail for smoother paths
    
    print("Generating NavMesh (this may take a moment)...")
    success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    
    if success:
        print(f"NavMesh generation successful!")
        print(f"NavMesh area: {sim.pathfinder.navigable_area}m²")
        islands = sim.pathfinder.num_islands
        print(f"Number of islands: {islands}")
        return True
    else:
        print("WARNING: NavMesh generation failed!")
        return False

def setup_simulator():
    # Scene dataset config
    ds_cfg = os.path.join(DATA_DIR, "scene_datasets", "hssd-hab", "hssd-hab.scene_dataset_config.json")
    if not os.path.isfile(ds_cfg):
        print(f"ERROR: dataset config not found: {ds_cfg}")
        sys.exit(1)

    # Simulator configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = ds_cfg
    sim_cfg.scene_id = "102344094.scene_instance.json"
    sim_cfg.enable_physics = True
    sim_cfg.load_semantic_mesh = False
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    
    # Increase physics solver iterations for stability
    sim_cfg.physics_config_file = "data/default.physics_config.json"

    # Define camera sensors
    top_cam = make_cam("top_rgb", (0.0, 0.0, 1.6), (0.0, 0.0, -np.pi/2))
    cam_pos = (0.1, 0.0, 0.4)
    cam_ori = (-np.pi/2, -np.pi, np.pi/2)
    robot_cam_pos = (0.0, 0.0, 0.0)
    robot_cam_ori = (0.0, 0.0, 0.0)
    front_rgb = make_cam("robot_rgb", robot_cam_pos, robot_cam_ori)
    front_depth = habitat_sim.CameraSensorSpec()
    front_depth.uuid = "depth_robot"
    front_depth.sensor_type = habitat_sim.SensorType.DEPTH
    front_depth.resolution = [480, 640]
    front_depth.position = mn.Vector3(*cam_pos)
    front_depth.orientation = mn.Vector3(*cam_ori)
    back_cam = make_cam("back_rgb", (-1.0, 0.0, 0.8), (-np.pi/1.5, np.pi, np.pi/2))
    front_facing_cam = make_cam("front_facing_rgb", (1.0, 0.0, 0.8), (np.pi/3, 0.0, np.pi/2))

    # Agent with sensors
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [top_cam, front_rgb, front_depth, back_cam, front_facing_cam]

    # Create simulator
    try:
        sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        agent = sim.get_agent(0)
    except Exception as e:
        print(f"ERROR: Failed to create simulator: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Load LoCoBot URDF
    urdf_fp = os.path.join(DATA_DIR, "robots", "locobot_wx250s.urdf")
    if not os.path.isfile(urdf_fp):
        print(f"ERROR: URDF not found: {urdf_fp}")
        sys.exit(1)
    
    try:
        ao_mgr = sim.get_articulated_object_manager()
        locobot = ao_mgr.add_articulated_object_from_urdf(
            urdf_fp,
            fixed_base=False,
            global_scale=1.0,
            mass_scale=1.0,
            force_reload=True,  # Force reload for consistent initialization
            maintain_link_order=True,   # Preserve original link order
            intertia_from_urdf=True,    # Use URDF inertia for better physics
            light_setup_key=habitat_sim.gfx.DEFAULT_LIGHTING_KEY,
        )
        if locobot is None:
            print("ERROR: failed to instantiate LoCoBot")
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load LoCoBot URDF: {e}")
        traceback.print_exc()
        sys.exit(1)
    
    # Set Enforcing joint limits
    locobot.auto_clamp_joint_limits = True

    # Link robot_rgb sensor to camera link with proper transform
    try:
        camera_link_id = locobot.get_link_id_from_name("camera_locobot_link")
        if camera_link_id != -1:
            camera_node = locobot.get_link_scene_node(camera_link_id)

            # Find robot_rgb sensor
            for sensor_spec in agent_cfg.sensor_specifications:
                if sensor_spec.uuid == "robot_rgb":
                    # Update sensor spec to match camera position
                    sensor_spec.position = mn.Vector3(0.0, 0.0, 0.0)
                    sensor_spec.orientation = mn.Vector3(0.0, 0.0, 0.0)
                    break
                
            # After agent creation, attach sensor to camera
            agent = sim.get_agent(0)
            rgb_sensor = sim._sensors.get("robot_rgb")
            if rgb_sensor:
                sensor_obj = rgb_sensor._sensor_object
                sensor_obj.node.parent = camera_node
                sensor_obj.node.translation = mn.Vector3(0, 0, 0)
                base_rot = mn.Quaternion(mn.Vector3(0.5, -0.5, -0.5), 0.5)
                sensor_obj.node.rotation = base_rot
                print("robot_rgb sensor attached to camera_locobot_link")
    except Exception as e:
        print(f"Error linking camera sensor: {e}")


    # ── INITIAL ROBOT POSE ──
    try:
        # Reset velocities first
        locobot.root_linear_velocity = mn.Vector3(0, 0, 0)
        locobot.root_angular_velocity = mn.Vector3(0, 0, 0)
        
        # Set initial state - properly in a single operation
        state = locobot.rigid_state
        state.translation = mn.Vector3(-10.4, 0.0, -2.0)
        upright_q = mn.Quaternion.rotation(Rad(-np.pi/2), mn.Vector3(1, 0, 0))
        yaw_q = mn.Quaternion.rotation(Rad(np.pi), mn.Vector3(0, 1, 0))
        state.rotation = yaw_q * upright_q
        locobot.rigid_state = state
        
        # Verify state was set correctly
        print(f"DEBUG: Initial robot position: {locobot.translation}")
        print(f"DEBUG: Initial robot rotation: {locobot.rotation}")
    except Exception as e:
        print(f"ERROR: Failed to set initial robot pose: {e}")
        traceback.print_exc()

    # Store initial references (copy to avoid swizzle issues)
    global _reference_position, _reference_orientation
    _reference_position = mn.Vector3(locobot.translation)
    _reference_orientation = mn.Quaternion(locobot.rotation.vector, locobot.rotation.scalar)
    
    # ―― PHYSICS TUNING: friction & damping ――
    # Define exact camera link names
    CAMERA_LINK_NAMES = ["locobot/pan_link", "locobot/tilt_link", "camera_locobot_link"]
    
    for lid in locobot.get_link_ids():
        try:
            # Increase friction for better grip
            link_name = locobot.get_link_name(lid)
            
            # Apply higher friction for camera-related links (using exact names)
            if link_name in CAMERA_LINK_NAMES:
                locobot.set_link_friction(lid, 30.0)  # Increased friction for stability
                #print(f"Applied higher friction (30.0) to {link_name}")
            else:
                locobot.set_link_friction(lid, 10.0)  # Moderate friction for non-camera links
        except Exception as e:
            print(f"Error setting friction: {e}")

    # Joint motor configuration with specific mapping for each joint
    dof_map = {locobot.get_link_joint_name(i): i
               for i in locobot.get_link_ids() if locobot.get_link_num_dofs(i) > 0}
    motor_ids = {}
    motor_settings = {}
    
    def mk_motor(name, impulse):
        try:
            lid = dof_map[name]
            # Configure different motor settings based on joint type
            if name == "left_finger":
                # For left gripper finger: strong position-based control
                s = phys.JointMotorSettings(
                    position_target=SAFE_LEFT_OPEN,  # Initialize to OPEN position (MAX)
                    position_gain=600.0,             # INCREASED position gain significantly
                    velocity_target=0.0,
                    velocity_gain=120.0,             # INCREASED damping
                    max_impulse=impulse
                )
            elif name == "right_finger":
                # For right gripper finger: strong position-based control
                s = phys.JointMotorSettings(
                    position_target=SAFE_RIGHT_OPEN,  # Initialize to OPEN position (MAX)
                    position_gain=600.0,              # INCREASED position gain significantly
                    velocity_target=0.0,
                    velocity_gain=120.0,              # INCREASED damping
                    max_impulse=impulse
                )
            elif name == "pan":
                # Camera pan: enhanced stabilization settings
                s = phys.JointMotorSettings(
                    position_target=0.0,  # Neutral pan position
                    position_gain=1000.0,   # Very strong position control
                    velocity_target=0.0,
                    velocity_gain=200.0,   # High damping to prevent oscillation
                    max_impulse=1500.0
                )
            elif name == "tilt":
                # Camera tilt: enhanced stabilization settings
                s = phys.JointMotorSettings(
                    position_target=-0.26,  # Default downward tilt
                    velocity_target=0.0,
                    position_gain=1000.0,     # Very strong position control
                    velocity_gain=200.0,     # High damping to prevent oscillation
                    max_impulse=1500.0
                )
            elif "wheel" in name:
                # For wheels: pure velocity control with increased damping
                s = phys.JointMotorSettings(
                    position_target=0.0,
                    position_gain=0.0,
                    velocity_target=0.0,
                    velocity_gain=10.0,      # Higher velocity gain for better damping
                    max_impulse=impulse * 0.5  # Reduced to avoid overwhelming other joints
                )
            elif name in ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]:
                s = phys.JointMotorSettings(
                    position_target=0.0,        # Neutral position
                    position_gain=10.0,         # Higher position gain for better holding
                    velocity_target=0.0,
                    velocity_gain=5.0,          # Higher damping to prevent drift
                    max_impulse=impulse * 0.8   # Slightly reduced for smoother motion
                )
                #print(f"Created enhanced stability motor for arm joint '{name}'")
            else:
                # For arm joints: balanced position/velocity control with increased damping
                s = phys.JointMotorSettings(
                    position_target=0.0,
                    position_gain=0.8,          # Increased position gain
                    velocity_target=0.0,
                    velocity_gain=3.0,          # Increased damping for smoother motion
                    max_impulse=impulse * 0.8   # Slightly reduced for smoother motion
                )

            try:
                mid = locobot.create_joint_motor(lid, s)
                motor_ids[lid] = mid
                motor_settings[lid] = s
                return s
            except Exception as e:
                print(f"Error creating motor for {name}: {e}")
                traceback.print_exc()
                return None

        except Exception as e:
            print(f"Error configuring motor for {name}: {e}")
            traceback.print_exc()
            return None

    # Define motor impulses with values tuned for stability
    joint_motor_config = [
        ("wheel_left_joint",  800.0),  # Reduced from 1000 for less overwhelming force
        ("wheel_right_joint", 800.0),  # Reduced from 1000 for less overwhelming force
        ("waist", 150.0),
        ("shoulder", 150.0),
        ("elbow", 150.0),
        ("forearm_roll", 150.0),
        ("wrist_angle", 120.0),
        ("wrist_rotate", 120.0),
        ("left_finger", 1000.0),  # INCREASED for stronger position control
        ("right_finger", 1000.0), # INCREASED for stronger position control
        ("pan", 300.0),  # Increased for better camera stability
        ("tilt", 300.0)   # Increased for better camera stability
    ]
    
    # Create all joint motors
    for name, imp in joint_motor_config:
        if name in dof_map:
            s = mk_motor(name, imp)
            if s:
                print(f"Created motor for {name}: pos_gain={s.position_gain}, vel_gain={s.velocity_gain}, max_impulse={s.max_impulse}")
        else:
            print(f"Warning: Joint {name} not found in dof_map")

    # Explicitly initialize gripper to a consistent fully open state
    try:
        print("Explicitly initializing gripper to fully open position...")
        set_gripper_state(locobot, motor_ids, motor_settings, dof_map, "open")
    except Exception as e:
        print(f"Error in explicit gripper initialization: {e}")
        traceback.print_exc()

    # Initialize camera pan/tilt to resting pose
    try:
        if "pan" in dof_map:
            pan_lid = dof_map["pan"]
            if pan_lid in motor_ids:
                pan_mid = motor_ids[pan_lid]
                motor_settings[pan_lid].position_target = 0.0
                locobot.update_joint_motor(pan_mid, motor_settings[pan_lid])
        
        if "tilt" in dof_map:
            tilt_lid = dof_map["tilt"]
            if tilt_lid in motor_ids:
                tilt_mid = motor_ids[tilt_lid]
                motor_settings[tilt_lid].position_target = -0.26  # Default downward tilt
                locobot.update_joint_motor(tilt_mid, motor_settings[tilt_lid])
    except Exception as e:
        print(f"Error initializing camera pose: {e}")
        traceback.print_exc()

    # Create camera controller
    try:
        camera_controller = CameraController(
            locobot=locobot,
            motor_ids=motor_ids,
            motor_settings=motor_settings,
            dof_map=dof_map,
            initial_pan=0.0,
            initial_tilt=-0.26,  # Slightly downward default view
            debug=False  # Set to False in production
        )

        # Force initial camera position to match controller's locked state
        if "pan" in dof_map and "tilt" in dof_map:
            pan_id = dof_map["pan"]
            tilt_id = dof_map["tilt"]
            joint_positions = locobot.joint_positions
            pos_offset_pan = locobot.get_link_joint_pos_offset(pan_id)
            pos_offset_tilt = locobot.get_link_joint_pos_offset(tilt_id)

            if pos_offset_pan >= 0 and pos_offset_tilt >= 0:
                joint_positions[pos_offset_pan] = camera_controller.locked_pan
                joint_positions[pos_offset_tilt] = camera_controller.locked_tilt
                locobot.joint_positions = joint_positions

    except Exception as e:
        print(f"Error creating camera controller: {e}")
        traceback.print_exc()
        # Create a dummy camera stabilizer as fallback
        camera_controller = None
    
    # Create camera stabilizer
    try:
        camera_stabilizer = CameraStabilizer(
            locobot=locobot,
            motor_ids=motor_ids,
            motor_settings=motor_settings,
            dof_map=dof_map,
            debug=False
        )
    except Exception as e:
        print(f"Error creating camera stabilizer: {e}")
        traceback.print_exc()
        # Define minimal camera stabilizer to avoid errors
        camera_stabilizer = type('', (), {'stabilize_motors': lambda x: None})()
    
    # Initialize all joint velocities to zero
    for lid in motor_ids:
        try:
            joint_name = locobot.get_link_joint_name(lid)
            motor_settings[lid].velocity_target = 0.0
            locobot.update_joint_motor(motor_ids[lid], motor_settings[lid])
            print(f"DEBUG: Initial velocity for joint '{joint_name}' set to 0.0")
        except Exception as e:
            print(f"Error initializing velocity for joint {lid}: {e}")


    # Force arm to initial positions
    for joint_name, target_pos in ARM_REST_POSITIONS.items():
        if joint_name in dof_map:
            joint_id = dof_map[joint_name]
            pos_offset = locobot.get_link_joint_pos_offset(joint_id)
            if pos_offset >= 0:
                joint_positions = locobot.joint_positions
                joint_positions[pos_offset] = target_pos
                locobot.joint_positions = joint_positions

    # Take several small physics steps to let the initial configuration settle
    print("Taking initial settling steps...")
    for i in range(50):  # More steps for better initial stabilization
        try:
            sim.step_physics(1/240.0)  # Smaller timestep for stability
            
            # REAPPLY ARM REST POSITION EVERY FEW STEPS
            if i % 5 == 0:
                for joint_name, target_pos in ARM_REST_POSITIONS.items():
                    if joint_name in dof_map:
                        joint_id = dof_map[joint_name]
                        pos_offset = locobot.get_link_joint_pos_offset(joint_id)
                        if pos_offset >= 0:
                            joint_positions = locobot.joint_positions
                            joint_positions[pos_offset] = target_pos
                            locobot.joint_positions = joint_positions
        

            
            # Every 5 steps, verify the robot position hasn't drifted
            if i % 5 == 0:
                state = locobot.rigid_state
                state.translation = _reference_position
                state.rotation = _reference_orientation
                locobot.rigid_state = state
                
            # Stabilize gripper positions every few steps
            if i % 3 == 0 and "left_finger" in dof_map and "right_finger" in dof_map:
                # Reinforce gripper open position (crucial for consistency)
                left_id = dof_map["left_finger"]
                right_id = dof_map["right_finger"]
                left_pos_offset = locobot.get_link_joint_pos_offset(left_id)
                right_pos_offset = locobot.get_link_joint_pos_offset(right_id)
                
                if left_pos_offset >= 0 and right_pos_offset >= 0:
                    joint_positions = locobot.joint_positions
                    joint_positions[left_pos_offset] = SAFE_LEFT_OPEN
                    joint_positions[right_pos_offset] = SAFE_RIGHT_OPEN
                    locobot.joint_positions = joint_positions
                    
                    # Also update motor settings periodically
                    motor_settings[left_id].position_target = SAFE_LEFT_OPEN
                    motor_settings[right_id].position_target = SAFE_RIGHT_OPEN
                    locobot.update_joint_motor(motor_ids[left_id], motor_settings[left_id])
                    locobot.update_joint_motor(motor_ids[right_id], motor_settings[right_id])
                
            # Also actively stabilize camera joints
            if i % 5 == 0 and camera_stabilizer:
                try:
                    camera_stabilizer.stabilize_motors(False)
                except:
                    pass
        except Exception as e:
            print(f"Error during initial settling step {i}: {e}")
    
    # Verify gripper state after settling
    try:
        if "left_finger" in dof_map and "right_finger" in dof_map:
            left_id = dof_map["left_finger"]
            right_id = dof_map["right_finger"]
            left_pos = get_joint_position(locobot, left_id)
            right_pos = get_joint_position(locobot, right_id)
            print(f"Final gripper positions after settling: left={left_pos:.4f}, right={right_pos:.4f}")
            
            # Force one final gripper correction if needed
            if abs(left_pos - SAFE_LEFT_OPEN) > 0.001 or abs(right_pos - SAFE_RIGHT_OPEN) > 0.001:
                print("Applying final gripper position correction...")
                set_gripper_state(locobot, motor_ids, motor_settings, dof_map, "open")
    except Exception as e:
        print(f"Error in final gripper verification: {e}")
    
    initialize_navmesh(sim)
    debug_navmesh(sim)
    print("Robot initialization complete.")
    return sim, agent, locobot, motor_ids, motor_settings, dof_map, camera_controller


def debug_navmesh(sim):
    """Debug the NavMesh generation and verify it's suitable for navigation."""
    if not sim.pathfinder.is_loaded:
        print("ERROR: NavMesh not loaded")
        return False
    
    # Try to get several random points to verify sampling works
    successes = 0
    for i in range(5):
        try:
            point = sim.pathfinder.get_random_navigable_point()
            successes += 1
        except Exception as e:
            pass
    
    if successes == 0:
        print("CRITICAL: Could not sample any random points!")
        return False
    
    # Check if current robot position is navigable
    robot_pos = sim.robots[0].translation if hasattr(sim, 'robots') else None
    if robot_pos:
        is_navigable = sim.pathfinder.is_navigable(robot_pos)
        if not is_navigable:
            print(f"Warning: Robot position is NOT navigable")
    
    return successes > 0


# Position warning state is controlled at the top level

def enforce_robot_position_constraints(locobot, reference_position=None, max_y_drift=0.2):
    """Monitor and enforce robot position constraints during exploration."""
    global _position_warnings_enabled, _exploration_active

    current_pos = locobot.translation

    # In exploration mode, strictly enforce y-axis (height)
    # Check exploration mode safely
    try:
        in_exploration = _exploration_active
    except NameError:
        in_exploration = False
    
    if in_exploration:
        # Always constrain height to floor level during exploration
        if current_pos[1] < -0.05 or current_pos[1] > 0.1:
            state = locobot.rigid_state
            state.translation = mn.Vector3(
                current_pos[0],  # Keep current x
                0.0,             # Reset y to exactly floor level
                current_pos[2]   # Keep current z
            )
            # Zero ALL velocities to prevent instability
            locobot.root_linear_velocity = mn.Vector3(0, 0, 0)
            locobot.root_angular_velocity = mn.Vector3(0, 0, 0)
            locobot.rigid_state = state
            return True
        return False

    # Rest of the function remains the same...

    if hasattr(locobot, "_exploration_active") and locobot._exploration_active:
        return False
    
    # Skip all warnings if disabled
    if not _position_warnings_enabled:
        return False
        
    # Get current state (for monitoring only)
    current_pos = locobot.translation
    
    # Only log a warning if needed - no actual correction
    if reference_position is not None:
        y_drift = abs(current_pos[1] - reference_position[1])
        if y_drift > 1.0:
            print(f"[INFO] Y-drift: {y_drift:.3f}m (normal in simulation)")
    
    if current_pos[1] < -0.3:
        print(f"[INFO] Robot below floor: {current_pos[1]:.3f}m (normal in simulation)")
    
    # Never apply corrections, just monitor
    return False


def run_simulator_step(
    sim, agent, locobot, motor_ids, motor_settings, dof_map,
    key, DRIVE_SPEED, TURN_SPEED, ARM_SPEED, GRIP_SPEED, dt, camera_controller
):
    """
    Run one simulation step with enhanced stability for cameras and gripper.
    """
    global _reference_position, _reference_orientation, _last_command_time
    global _is_arm_moving, _is_gripper_moving, _debug_mode, _exploration_active
    
    controller_command = None
    # Track command/state
    current_time = time.time()
    is_camera_cmd = False
    
    # Track initial camera positions for drift detection
    if "pan" in dof_map and "tilt" in dof_map:
        pan_id = dof_map["pan"]
        tilt_id = dof_map["tilt"]
        initial_pan_pos = get_joint_position(locobot, pan_id)
        initial_tilt_pos = get_joint_position(locobot, tilt_id)
        #print(f"START STEP: Camera Pan={initial_pan_pos:.6f}, Tilt={initial_tilt_pos:.6f}")
    
    if camera_controller:
        # In autonomous mode, just stabilize the camera
        is_camera_cmd = False
        try:
            camera_controller.stabilize()
        except Exception as e:
            print(f"Camera stabilization error: {e}")
    
    
    # Update arm/gripper movement state based on key
    if key is not None and key != 0 and key != -1:
        if key in [ord('g'), ord('h')]:
            _last_command_time = current_time
            _is_gripper_moving = True
            if _debug_mode:
                print(f" GRIPPER COMMAND: {chr(key)}")
        elif key in [ord('u'), ord('j'), ord('i'), ord('k'),
                     ord('o'), ord('l'), ord('p'), ord(';'),
                     ord('['), ord(']'), ord('n'), ord('m')]:
            _last_command_time = current_time
            _is_arm_moving = True
            if _debug_mode:
                print(f" ARM COMMAND: {chr(key)}")
    
    # Auto-stop after timeout (0.5s for faster stabilization)
    if (_is_arm_moving or _is_gripper_moving) and current_time - _last_command_time > 0.5:
        if _is_arm_moving:
            _is_arm_moving = False
            if _debug_mode:
                print(" ARM MOTION TIMEOUT - SWITCHING TO STATIC MODE")
        if _is_gripper_moving:
            _is_gripper_moving = False
            if _debug_mode:
                print(" GRIPPER MOTION TIMEOUT - SWITCHING TO STATIC MODE")
    
    # Define arm and gripper joints
    arm_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
    gripper_joints = ["left_finger", "right_finger"]
    
    # ---- STORE CURRENT JOINT POSITIONS ----
    arm_current_positions = {}
    for name in arm_joints:
        if name in dof_map:
            lid = dof_map[name]
            pos = get_joint_position(locobot, lid)
            arm_current_positions[name] = pos
    
    # Store current gripper positions
    gripper_current_positions = {}
    for name in gripper_joints:
        if name in dof_map:
            lid = dof_map[name]
            pos = get_joint_position(locobot, lid)
            gripper_current_positions[name] = pos
    
    # ---- ZERO ALL VELOCITIES ----
    for lid in motor_settings:
        motor_settings[lid].velocity_target = 0.0
    
    # Zero robot base velocity
    locobot.root_linear_velocity = mn.Vector3(0, 0, 0)
    locobot.root_angular_velocity = mn.Vector3(0, 0, 0)
    
    
    # ---- PROCESS ARM MOVEMENT COMMANDS ----
    arm_map = {
        ord('u'): ("waist", ARM_SPEED),        ord('j'): ("waist", -ARM_SPEED),
        ord('i'): ("shoulder", ARM_SPEED),     ord('k'): ("shoulder", -ARM_SPEED),
        ord('o'): ("elbow", ARM_SPEED),        ord('l'): ("elbow", -ARM_SPEED),
        ord('p'): ("forearm_roll", ARM_SPEED), ord(';'): ("forearm_roll", -ARM_SPEED),
        ord('['): ("wrist_angle", ARM_SPEED),  ord(']'): ("wrist_angle", -ARM_SPEED),
        ord('n'): ("wrist_rotate", ARM_SPEED), ord('m'): ("wrist_rotate", -ARM_SPEED)
    }
    
    # Only hold arm at rest during exploration, not when picking is active
    # Check if we have an active picking operation
    active_picking = False
    if camera_controller and hasattr(camera_controller, 'is_picking') and camera_controller.is_picking:
        active_picking = True
        # Mark picking is active and allow arm movement
        _is_arm_moving = True
        _is_gripper_moving = True
        _exploration_active = False
        _last_command_time = time.time()  # Reset timeout counter
        print("DEBUG: Active picking operation detected, allowing arm movement and setting all relevant flags")
    
    # Allow arm movement during picking operations
    if not active_picking and not _is_arm_moving:
        hold_arm_at_rest(locobot, motor_ids, motor_settings, dof_map)
    elif _is_arm_moving:
        print("DEBUG: Arm is moving, not enforcing rest position")
    
    # Process arm movement if arm is active
    if key in arm_map and _is_arm_moving:
        joint_name, velocity = arm_map[key]
        
        if joint_name in dof_map:
            joint_id = dof_map[joint_name]
            
            current_pos = arm_current_positions[joint_name]
            lower_limit, upper_limit = get_joint_limits(locobot, joint_id)
            
            # Check if at joint limits
            at_min_limit = current_pos <= lower_limit[0] + 0.02 and velocity < 0
            at_max_limit = current_pos >= upper_limit[0] - 0.02 and velocity > 0
            
            if not at_min_limit and not at_max_limit:
                # Use velocity control for movement
                motor_settings[joint_id] = phys.JointMotorSettings(
                    position_target=current_pos,
                    position_gain=0.5,
                    velocity_target=velocity,
                    velocity_gain=1.0,
                    max_impulse=200.0
                )
                
                locobot.update_joint_motor(motor_ids[joint_id], motor_settings[joint_id])
            else:
                # At joint limit - lock position
                motor_settings[joint_id] = phys.JointMotorSettings(
                    position_target=current_pos,
                    position_gain=500.0,
                    velocity_target=0.0,
                    velocity_gain=100.0,
                    max_impulse=1000.0
                )
                
                locobot.update_joint_motor(motor_ids[joint_id], motor_settings[joint_id])
    
    # ---- IMPROVED GRIPPER CONTROL ----
    if key == ord('g') and _is_gripper_moving:  # Open gripper
        set_gripper_state(locobot, motor_ids, motor_settings, dof_map, "open")
        if _debug_mode:
            print(f" OPENING GRIPPER: Left={SAFE_LEFT_OPEN:.4f}, Right={SAFE_RIGHT_OPEN:.4f}")
                
    elif key == ord('h') and _is_gripper_moving:  # Close gripper
        set_gripper_state(locobot, motor_ids, motor_settings, dof_map, "closed")
        if _debug_mode:
            print(f" CLOSING GRIPPER: Left={SAFE_LEFT_CLOSED:.4f}, Right={SAFE_RIGHT_CLOSED:.4f}")
    
    # ---- DRIVING COMMANDS ----
    is_driving = False
    
    if key == 65362:  # Forward arrow
        motor_settings[dof_map["wheel_left_joint"]].velocity_target = DRIVE_SPEED
        motor_settings[dof_map["wheel_right_joint"]].velocity_target = DRIVE_SPEED
        is_driving = True
    elif key == 65364:  # Backward arrow
        motor_settings[dof_map["wheel_left_joint"]].velocity_target = -DRIVE_SPEED
        motor_settings[dof_map["wheel_right_joint"]].velocity_target = -DRIVE_SPEED
        is_driving = True
    elif key in (65361, ord('q')):  # Left arrow or 'q'
        motor_settings[dof_map["wheel_left_joint"]].velocity_target = -TURN_SPEED
        motor_settings[dof_map["wheel_right_joint"]].velocity_target = TURN_SPEED
        is_driving = True
    elif key in (65363, ord('w')):  # Right arrow or 'w'
        motor_settings[dof_map["wheel_left_joint"]].velocity_target = TURN_SPEED
        motor_settings[dof_map["wheel_right_joint"]].velocity_target = -TURN_SPEED
        is_driving = True
    
    # ---- CHECK FOR SPECIAL CONTROLLER COMMANDS ----
    # Handle RESET_POSITION and STOP commands
    if isinstance(controller_command, dict):
        action_type = controller_command.get("type")
        if action_type == "RESET_POSITION":
            try:
                reset_pos = controller_command.get("value")
                if reset_pos is not None:
                    state = locobot.rigid_state
                    # Keep original X,Z but reset Y to original position
                    state.translation = mn.Vector3(
                        state.translation.x,
                        reset_pos[1],  # Reset Y to starting position
                        state.translation.z
                    )
                    # Zero out velocities
                    locobot.root_linear_velocity = mn.Vector3(0, 0, 0)
                    locobot.root_angular_velocity = mn.Vector3(0, 0, 0)
                    locobot.rigid_state = state
                    print(f"Reset robot position Y coordinate to {reset_pos[1]}")
            except Exception as e:
                print(f"Error resetting position: {e}")
        elif action_type == "STOP":
            # Set all wheel velocities to zero
            if "wheel_left_joint" in dof_map and "wheel_right_joint" in dof_map:
                motor_settings[dof_map["wheel_left_joint"]].velocity_target = 0.0
                motor_settings[dof_map["wheel_right_joint"]].velocity_target = 0.0
                print("Emergency stop command received")
    
    # ---- APPLY MOTOR SETTINGS ----
    try:
        for lid, mid in motor_ids.items():
            if lid in motor_settings:
                locobot.update_joint_motor(mid, motor_settings[lid])
    except Exception as e:
        print(f" Error updating motors: {e}")
        traceback.print_exc()
    
    # ---- PHYSICS STEPS WITH CAMERA STABILIZATION ----
        
    # Store target positions for post-physics correction
    joint_target_positions = {}
    
    # Store arm joint targets
    if not _is_arm_moving:
        for joint_name in arm_joints:
            if joint_name in dof_map:
                joint_id = dof_map[joint_name]
                if joint_id in motor_settings:
                    joint_target_positions[joint_id] = motor_settings[joint_id].position_target
    
    # Store gripper joint targets
    if not _is_gripper_moving:
        for joint_name in gripper_joints:
            if joint_name in dof_map:
                joint_id = dof_map[joint_name]
                if joint_id in motor_settings:
                    joint_target_positions[joint_id] = motor_settings[joint_id].position_target
    elif key == ord('g'):  # Opening
        if "left_finger" in dof_map and "right_finger" in dof_map:
            joint_target_positions[dof_map["left_finger"]] = SAFE_LEFT_OPEN
            joint_target_positions[dof_map["right_finger"]] = SAFE_RIGHT_OPEN
    elif key == ord('h'):  # Closing
        if "left_finger" in dof_map and "right_finger" in dof_map:
            joint_target_positions[dof_map["left_finger"]] = SAFE_LEFT_CLOSED
            joint_target_positions[dof_map["right_finger"]] = SAFE_RIGHT_CLOSED
    
    # Run physics with smaller substeps and active stabilization
    try:
        # Set up default values and number of substeps based on exploration mode
        exploration_mode = _exploration_active  # Use the global variable directly
        num_substeps = 12  # Default for manipulation mode
        
        # If arm is moving, disable exploration mode
        if _is_arm_moving:
            _exploration_active = False
            exploration_mode = False
            # Print only when actually changing the mode to avoid spam
            if _debug_mode:
                print("Arm movement active - disabling exploration mode")
        
        # Use different settings for exploration mode
        if exploration_mode:
            num_substeps = 4  # Fewer steps when in exploration mode
        else:
            num_substeps = 12  # More steps for better physics when manipulating
            
        substep_dt = dt / num_substeps
        enforce_robot_position_constraints(locobot, _reference_position, max_y_drift=1.0)
        
        for i in range(num_substeps):        
            # Force camera positions BEFORE physics step
            # Only stabilize when camera is locked, not during movement
            if camera_controller and camera_controller.should_apply_physics_constraint():
                if "pan" in dof_map and "tilt" in dof_map:
                    pan_id = dof_map["pan"]
                    tilt_id = dof_map["tilt"]
                    pos_offset_pan = locobot.get_link_joint_pos_offset(pan_id)
                    pos_offset_tilt = locobot.get_link_joint_pos_offset(tilt_id)

                    if pos_offset_pan >= 0 and pos_offset_tilt >= 0:
                        joint_positions = locobot.joint_positions
                        joint_positions[pos_offset_pan] = camera_controller.locked_pan
                        joint_positions[pos_offset_tilt] = camera_controller.locked_tilt
                        locobot.joint_positions = joint_positions
                    
            # Step physics simulation
            sim.step_physics(substep_dt)

            if camera_controller:
                camera_controller.stabilize()

            if camera_controller:
                camera_state = camera_controller.get_current_state()
                #print(f"Step {i}: pan={camera_state['pan']:.4f}, tilt={camera_state['tilt']:.4f}")

            fix_ar_tag_position(locobot)
            stabilize_gripper_prop(locobot, dof_map, motor_ids, motor_settings)
            
            # Apply strict position constraint during exploration
            # Check exploration mode safely
            try:
                exploration_mode = _exploration_active
            except NameError:
                exploration_mode = False
                
            if exploration_mode:
                current_pos = locobot.translation
                if current_pos[1] < -0.05 or current_pos[1] > 0.1:
                    state = locobot.rigid_state
                    state.translation = mn.Vector3(
                        current_pos[0],          # Keep current x
                        0.0,                     # Reset y to floor level
                        current_pos[2]           # Keep current z
                    )
                    # Zero velocities
                    locobot.root_linear_velocity = mn.Vector3(0, 0, 0)
                    locobot.root_angular_velocity = mn.Vector3(0, 0, 0)
                    locobot.rigid_state = state
            
            # Force camera positions AFTER physics step too
            if "pan" in dof_map and "tilt" in dof_map:
                pan_id = dof_map["pan"]
                tilt_id = dof_map["tilt"]
                pos_offset_pan = locobot.get_link_joint_pos_offset(pan_id)
                pos_offset_tilt = locobot.get_link_joint_pos_offset(tilt_id)
    
                if pos_offset_pan >= 0 and pos_offset_tilt >= 0:
                    joint_positions = locobot.joint_positions
                    # Force exact positions
                    joint_positions[pos_offset_pan] = 0.0  # Fixed pan position
                    joint_positions[pos_offset_tilt] = -0.26  # Fixed tilt position
                    locobot.joint_positions = joint_positions
            
            # Apply camera stabilization every substep
            if camera_controller and i % 2 == 0:
                try:
                    camera_controller.stabilize()
                except Exception as e:
                    if _debug_mode:
                        print(f"Camera stabilization error: {e}")
            
            # Apply joint position correction every substep for stability
            if joint_target_positions and i % 2 == 0:
                for joint_id, target_pos in joint_target_positions.items():
                    # Force joint positions to match targets exactly
                    pos_offset = locobot.get_link_joint_pos_offset(joint_id)
                    if pos_offset >= 0:
                        joint_positions = locobot.joint_positions
                        joint_positions[pos_offset] = target_pos
                        locobot.joint_positions = joint_positions
    
    except Exception as e:
        print(f" Error in physics advancement: {e}")
        traceback.print_exc()
        
        # Fallback physics
        for i in range(5):
            sim.step_physics(dt / 5)
            
            # Apply camera stabilization in fallback
            if camera_controller and i % 2 == 0:
                try:
                    camera_controller.stabilize()
                except Exception as e:
                    pass
    
    # ---- FINAL STABILIZATION ----
    
    # Apply one final camera stabilization
    if camera_controller:
        try:
            camera_controller.stabilize()
        except Exception as e:
            if _debug_mode:
                print(f"Final camera stabilization error: {e}")
    
    # Apply one final correction to ensure exact positions
    if joint_target_positions:
        for joint_id, target_pos in joint_target_positions.items():
            pos_offset = locobot.get_link_joint_pos_offset(joint_id)
            if pos_offset >= 0:
                joint_positions = locobot.joint_positions
                joint_positions[pos_offset] = target_pos
                locobot.joint_positions = joint_positions
    
    # ---- BASE POSITION MANAGEMENT ----
    
    # Update reference position if driving
    if is_driving:
        _reference_position = mn.Vector3(locobot.translation)
    
    # Update reference orientation if turning
    turning = key in [65361, 65363, ord('q'), ord('w')]
    if turning:
        _reference_orientation = mn.Quaternion(
            locobot.rotation.vector, 
            locobot.rotation.scalar
        )
    
    # Apply one final position constraint to ensure robot stays grounded
    enforce_robot_position_constraints(locobot, _reference_position, max_y_drift=0.3)
    
    # Prevent base position drift when not driving
    if not is_driving and not turning:
        state = locobot.rigid_state
        state.translation = _reference_position
        state.rotation = _reference_orientation
        locobot.rigid_state = state
    
    # ---- AGENT CAMERA SYNCHRONIZATION ----

    try:
        # Get the robot's base position and rotation
        robot_state = locobot.rigid_state
        robot_position = robot_state.translation
        robot_rotation = robot_state.rotation

        # Update agent position to match robot
        agent.scene_node.translation = robot_position
        agent.scene_node.rotation = robot_rotation

        # Sync camera sensor with pan/tilt joints
        if camera_controller and "robot_rgb" in sim._sensors:
            rgb_sensor = sim._sensors["robot_rgb"]
            sensor_obj = rgb_sensor._sensor_object

            # Get current pan/tilt positions
            pan_pos = camera_controller.pan_current
            tilt_pos = camera_controller.tilt_current

            # Apply pan/tilt rotations to sensor
            base_rot = mn.Quaternion(mn.Vector3(0.5, -0.5, -0.5), 0.5)
            pan_rot = mn.Quaternion.rotation(mn.Rad(pan_pos), mn.Vector3(0, 1, 0))
            tilt_rot = mn.Quaternion.rotation(mn.Rad(tilt_pos), mn.Vector3(1, 0, 0))
            sensor_obj.node.rotation = base_rot * pan_rot * tilt_rot

    except Exception as e:
        print(f" Error syncing agent camera: {e}")
        traceback.print_exc()
    
    # ---- CAPTURE OBSERVATIONS ----
    
    try:
        # Get sensor observations
        obs = sim.get_sensor_observations()
        
        # Extract camera views
        top_img = obs['top_rgb']
        rob_img = obs['robot_rgb']
        back_img = obs['back_rgb']
        front_facing_img = obs['front_facing_rgb']
        
        # Add view labels
        cv2.putText(top_img, 'TOP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(rob_img, 'FRONT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(back_img, 'BACK', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(front_facing_img, 'FRONT VIEW', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        top_row = np.hstack((top_img, rob_img))
        bottom_row = np.hstack((back_img, front_facing_img))
        combined = np.vstack((top_row, bottom_row))

        
    except Exception as e:
        print(f" Error capturing observations: {e}")
        traceback.print_exc()
        
        # Return empty observations if error occurs
        obs = {}
        combined = np.zeros((480, 1920, 3), dtype=np.uint8)
    
    # Check for any camera drift that occurred during this step
    if "pan" in dof_map and "tilt" in dof_map:
        pan_id = dof_map["pan"]
        tilt_id = dof_map["tilt"]
        final_pan_pos = get_joint_position(locobot, pan_id)
        final_tilt_pos = get_joint_position(locobot, tilt_id)

    return obs, combined