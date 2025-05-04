#!/usr/bin/env python3
import os
import sys
import numpy as np
import magnum as mn
from magnum import Rad
import cv2
import time
import traceback

import habitat_sim
import habitat_sim.gfx
import habitat_sim.physics as phys
from camera_controller import CameraController  
from camera_stabilizer import CameraStabilizer

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
_debug_mode = True  # Set to True for detailed debugging output
_last_link_print_time = 0 

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

# Utility: create a color camera sensor specification
def make_cam(name, pos, ori):
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = name
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.resolution = [480, 640]
    cam.position = mn.Vector3(*pos)
    cam.orientation = mn.Vector3(*ori)
    return cam

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

# Print detailed information about all robot links and joints
def print_robot_links(locobot):
    """Print detailed information about all robot links and joints."""
    print("\n======== ROBOT LINK HIERARCHY ========")
    print(f"Robot has {locobot.num_links} links and {len(locobot.joint_positions)} DOFs")
    
    # Print base link info
    print("\nBASE LINK:")
    print(f"  Link ID: -1 (base)")
    try:
        print(f"  Link Name: {locobot.get_link_name(-1)}")
        print(f"  Position: {locobot.get_link_scene_node(-1).translation}")
        print(f"  Rotation: {locobot.get_link_scene_node(-1).rotation}")
    except Exception as e:
        print(f"  Error getting base link info: {e}")
    
    # Print all other links
    print("\nROBOT LINKS:")
    for link_id in locobot.get_link_ids():
        try:
            link_name = locobot.get_link_name(link_id)
            joint_name = locobot.get_link_joint_name(link_id)
            joint_type = locobot.get_link_joint_type(link_id)
            dofs = locobot.get_link_num_dofs(link_id)
            pos_offset = locobot.get_link_joint_pos_offset(link_id) if dofs > 0 else -1
            
            # Get joint limits if applicable
            limits_str = "N/A"
            if dofs > 0:
                try:
                    joint_limits = locobot.joint_position_limits
                    lower = joint_limits[0][pos_offset:pos_offset+dofs]
                    upper = joint_limits[1][pos_offset:pos_offset+dofs]
                    limits_str = f"{lower}, {upper}"
                    
                    # Cache the joint limits for later use
                    _joint_limits_cache[link_id] = (lower, upper)
                except Exception as e:
                    limits_str = f"Error getting limits: {e}"
            
            print(f"\n  Link ID: {link_id}")
            print(f"  Link Name: {link_name}")
            print(f"  Joint Name: {joint_name}")
            print(f"  Joint Type: {joint_type}")
            print(f"  DOFs: {dofs}")
            print(f"  Position Offset: {pos_offset}")
            print(f"  Joint Limits: {limits_str}")
            
            # Print current position and scene node info
            node = locobot.get_link_scene_node(link_id)
            print(f"  Scene Node Position: {node.translation}")
            print(f"  Scene Node Rotation: {node.rotation}")
            
            # Add flags for camera-related links
            if "camera" in link_name or "pan" in link_name or "tilt" in link_name:
                print(f"  ** CAMERA-RELATED LINK **")
        except Exception as e:
            print(f"  Error processing link {link_id}: {e}")
    
    # Print joint motors configuration
    print("\n======== JOINT MOTOR CONFIGURATION ========")
    try:
        motor_ids = locobot.existing_joint_motor_ids
        print(f"Robot has {len(motor_ids)} joint motors")
        
        for link_id, motor_id in motor_ids.items():
            try:
                settings = locobot.get_joint_motor_settings(motor_id)
                link_name = locobot.get_link_name(link_id)
                joint_name = locobot.get_link_joint_name(link_id)
                
                print(f"\n  Motor ID: {motor_id} (for Link ID: {link_id})")
                print(f"  Link/Joint Name: {link_name} / {joint_name}")
                print(f"  Position Gain: {settings.position_gain}")
                print(f"  Velocity Gain: {settings.velocity_gain}")
                print(f"  Max Impulse: {settings.max_impulse}")
                print(f"  Position Target: {settings.position_target}")
                print(f"  Velocity Target: {settings.velocity_target}")
            except Exception as e:
                print(f"  Error getting motor info for motor {motor_id}: {e}")
    except Exception as e:
        print(f"  Error accessing motor configuration: {e}")
    
    print("\n======== END ROBOT INFO ========\n")

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
def setup_simulator():
    # Scene dataset config
    ds_cfg = os.path.join(DATA_DIR, "scene_datasets", "hssd-hab", "hssd-hab.scene_dataset_config.json")
    if not os.path.isfile(ds_cfg):
        print(f"ERROR: dataset config not found: {ds_cfg}")
        sys.exit(1)

    # Simulator configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = ds_cfg
    sim_cfg.scene_id = "102344049.scene_instance.json"
    sim_cfg.enable_physics = True
    sim_cfg.load_semantic_mesh = False
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    
    # Increase physics solver iterations for stability
    sim_cfg.physics_config_file = "data/default.physics_config.json"

    # Define camera sensors
    top_cam = make_cam("top_rgb", (0.0, 0.0, 1.2), (0.0, 0.0, -np.pi/2))
    cam_pos = (0.1, 0.0, 0.4)
    cam_ori = (-np.pi/2, -np.pi, np.pi/2)
    front_rgb = make_cam("robot_rgb", cam_pos, cam_ori)
    front_depth = habitat_sim.CameraSensorSpec()
    front_depth.uuid = "depth_robot"
    front_depth.sensor_type = habitat_sim.SensorType.DEPTH
    front_depth.resolution = [480, 640]
    front_depth.position = mn.Vector3(*cam_pos)
    front_depth.orientation = mn.Vector3(*cam_ori)
    back_cam = make_cam("back_rgb", (-1.0, 0.0, 0.8), (-np.pi/1.5, np.pi, np.pi/2))

    # Agent with sensors
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [top_cam, front_rgb, front_depth, back_cam]
    
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
    
    # Print detailed information about all robot links and joints
    print_robot_links(locobot)

    # Set Enforcing joint limits
    locobot.auto_clamp_joint_limits = True

    # ── INITIAL ROBOT POSE ──
    try:
        # Reset velocities first
        locobot.root_linear_velocity = mn.Vector3(0, 0, 0)
        locobot.root_angular_velocity = mn.Vector3(0, 0, 0)
        
        # Set initial state - properly in a single operation
        state = locobot.rigid_state
        state.translation = mn.Vector3(1.0, 0.0, -2.0)
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
                print(f"Applied higher friction (30.0) to {link_name}")
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
                print(f"Created enhanced stability motor for arm joint '{name}'")
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

    # Take several small physics steps to let the initial configuration settle
    print("Taking initial settling steps...")
    for i in range(50):  # More steps for better initial stabilization
        try:
            sim.step_physics(1/240.0)  # Smaller timestep for stability
            
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
    
    print("Robot initialization complete.")
    return sim, agent, locobot, motor_ids, motor_settings, dof_map, camera_controller

def run_simulator_step(
    sim, agent, locobot, motor_ids, motor_settings, dof_map,
    key, DRIVE_SPEED, TURN_SPEED, ARM_SPEED, GRIP_SPEED, dt, camera_controller
):
    """
    Run one simulation step with enhanced stability for cameras and gripper.
    """
    global _reference_position, _reference_orientation, _last_command_time
    global _is_arm_moving, _is_gripper_moving, _debug_mode, _last_link_print_time
    
    # Track command/state
    current_time = time.time()
    is_camera_cmd = False
    
    # Track initial camera positions for drift detection
    if "pan" in dof_map and "tilt" in dof_map:
        pan_id = dof_map["pan"]
        tilt_id = dof_map["tilt"]
        initial_pan_pos = get_joint_position(locobot, pan_id)
        initial_tilt_pos = get_joint_position(locobot, tilt_id)
        print(f"START STEP: Camera Pan={initial_pan_pos:.6f}, Tilt={initial_tilt_pos:.6f}")
    
    if camera_controller:
        try:
            is_camera_cmd = camera_controller.process_key_command(key)
        except Exception as e:
            print(f"Error processing camera command: {e}")
            is_camera_cmd = False
    
    # Debug logging for key presses
    if _debug_mode and key is not None and key != 0 and key != -1:
        print(f"\n▓▓▓▓ NEW STEP - KEY {key} PRESSED ▓▓▓▓")
        if key >= 32 and key <= 126:
            print(f"Key: '{chr(key)}'")
    elif _debug_mode and key == -1:
        print(f"\n▓▓▓▓ NEW STEP - KEY -1 PRESSED ▓▓▓▓")
    
    # Update arm/gripper movement state based on key
    if key is not None and key != 0 and key != -1:
        if key in [ord('g'), ord('h')]:
            _last_command_time = current_time
            _is_gripper_moving = True
            if _debug_mode:
                print(f"🤖 GRIPPER COMMAND: {chr(key)}")
        elif key in [ord('u'), ord('j'), ord('i'), ord('k'),
                     ord('o'), ord('l'), ord('p'), ord(';'),
                     ord('['), ord(']'), ord('n'), ord('m')]:
            _last_command_time = current_time
            _is_arm_moving = True
            if _debug_mode:
                print(f"🦾 ARM COMMAND: {chr(key)}")
    
    # Auto-stop after timeout (0.5s for faster stabilization)
    if (_is_arm_moving or _is_gripper_moving) and current_time - _last_command_time > 0.5:
        if _is_arm_moving:
            _is_arm_moving = False
            if _debug_mode:
                print("⏱️ ARM MOTION TIMEOUT - SWITCHING TO STATIC MODE")
        if _is_gripper_moving:
            _is_gripper_moving = False
            if _debug_mode:
                print("⏱️ GRIPPER MOTION TIMEOUT - SWITCHING TO STATIC MODE")
    
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
    
    # ---- ARM STABILIZATION ----
    if not _is_arm_moving:
        if _debug_mode:
            print("\n┌────── ARM STABILITY CONTROL ──────┐")
        
        for joint_name in arm_joints:
            if joint_name in dof_map:
                joint_id = dof_map[joint_name]
                if joint_id in motor_ids:
                    current_pos = arm_current_positions[joint_name]
                    
                    motor_settings[joint_id] = phys.JointMotorSettings(
                        position_target=current_pos,
                        position_gain=500.0,
                        velocity_target=0.0,
                        velocity_gain=100.0,
                        max_impulse=1000.0
                    )
                    
                    locobot.update_joint_motor(motor_ids[joint_id], motor_settings[joint_id])
                    
                    if _debug_mode:
                        print(f"  {joint_name:12}: Locked at {current_pos:.4f}")
        
        if _debug_mode:
            print("└────────────────────────────────────┘")
    
    # ---- GRIPPER STABILITY CONTROL ----
    if not _is_gripper_moving:
        if _debug_mode:
            print("\n┌────── GRIPPER STABILITY CONTROL ────┐")
    
        for joint_name in ["left_finger", "right_finger"]:
            if joint_name in dof_map:
                joint_id = dof_map[joint_name]
                if joint_id in motor_ids:
                    current_pos = gripper_current_positions[joint_name]
    
                    # Clamp to limits
                    if joint_name == "left_finger":
                        if current_pos < LEFT_FINGER_MIN:
                            current_pos = LEFT_FINGER_MIN
                            if _debug_mode:
                                print(f"  {joint_name:12}: Position clamped to min limit {LEFT_FINGER_MIN:.4f}")
                        elif current_pos > LEFT_FINGER_MAX:
                            current_pos = LEFT_FINGER_MAX
                            if _debug_mode:
                                print(f"  {joint_name:12}: Position clamped to max limit {LEFT_FINGER_MAX:.4f}")
                    elif joint_name == "right_finger":
                        if current_pos < RIGHT_FINGER_MIN:
                            current_pos = RIGHT_FINGER_MIN
                            if _debug_mode:
                                print(f"  {joint_name:12}: Position clamped to min limit {RIGHT_FINGER_MIN:.4f}")
                        elif current_pos > RIGHT_FINGER_MAX:
                            current_pos = RIGHT_FINGER_MAX
                            if _debug_mode:
                                print(f"  {joint_name:12}: Position clamped to max limit {RIGHT_FINGER_MAX:.4f}")
    
                    # Create stiff motor setting
                    motor_settings[joint_id] = phys.JointMotorSettings(
                        position_target=current_pos,
                        position_gain=600.0,
                        velocity_target=0.0,
                        velocity_gain=120.0,
                        max_impulse=1200.0
                    )
    
                    # Apply motor update
                    locobot.update_joint_motor(motor_ids[joint_id], motor_settings[joint_id])
    
                    # Force joint position directly
                    pos_offset = locobot.get_link_joint_pos_offset(joint_id)
                    if pos_offset >= 0:
                        joint_positions = locobot.joint_positions
                        joint_positions[pos_offset] = current_pos
                        locobot.joint_positions = joint_positions
    
                    if _debug_mode:
                        print(f"  {joint_name:12}: Locked at {current_pos:.4f}")
    
        if _debug_mode:
            print("└────────────────────────────────────┘")
    
    # ---- PROCESS ARM MOVEMENT COMMANDS ----
    arm_map = {
        ord('u'): ("waist", ARM_SPEED),        ord('j'): ("waist", -ARM_SPEED),
        ord('i'): ("shoulder", ARM_SPEED),     ord('k'): ("shoulder", -ARM_SPEED),
        ord('o'): ("elbow", ARM_SPEED),        ord('l'): ("elbow", -ARM_SPEED),
        ord('p'): ("forearm_roll", ARM_SPEED), ord(';'): ("forearm_roll", -ARM_SPEED),
        ord('['): ("wrist_angle", ARM_SPEED),  ord(']'): ("wrist_angle", -ARM_SPEED),
        ord('n'): ("wrist_rotate", ARM_SPEED), ord('m'): ("wrist_rotate", -ARM_SPEED)
    }
    
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
            print(f"👐 OPENING GRIPPER: Left={SAFE_LEFT_OPEN:.4f}, Right={SAFE_RIGHT_OPEN:.4f}")
                
    elif key == ord('h') and _is_gripper_moving:  # Close gripper
        set_gripper_state(locobot, motor_ids, motor_settings, dof_map, "closed")
        if _debug_mode:
            print(f"✊ CLOSING GRIPPER: Left={SAFE_LEFT_CLOSED:.4f}, Right={SAFE_RIGHT_CLOSED:.4f}")
    
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
    
    # ---- APPLY MOTOR SETTINGS ----
    try:
        for lid, mid in motor_ids.items():
            if lid in motor_settings:
                locobot.update_joint_motor(mid, motor_settings[lid])
    except Exception as e:
        print(f"⚠️ Error updating motors: {e}")
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
        # Split into smaller substeps for stability
        num_substeps = 12
        substep_dt = dt / num_substeps
        
        for i in range(num_substeps):
            # Force camera positions BEFORE physics step
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
                    
            sim.step_physics(substep_dt)
            
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
        print(f"⚠️ Error in physics advancement: {e}")
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
    
    # Prevent base position drift when not driving
    if not is_driving and not turning:
        state = locobot.rigid_state
        state.translation = _reference_position
        state.rotation = _reference_orientation
        locobot.rigid_state = state
    
    # ---- AGENT CAMERA SYNCHRONIZATION ----
    
    # Sync agent camera with robot camera but maintain stabilization
    try:
        # Get the camera link from the stabilized scene node
        cam_link_id = locobot.get_link_id_from_name("camera_locobot_link")
        if cam_link_id != -1:
            cam_node = locobot.get_link_scene_node(cam_link_id)
            
            # Use a stabilized transform instead of the raw one
            if camera_controller and hasattr(camera_controller, "reference_transforms"):
                ref = camera_controller.reference_transforms.get("camera_locobot_link")
                if ref:
                    # Use the reference transform for stability
                    trans = ref["translation"]
                    rot = ref["rotation"]
                    
                    # Update agent's scene node with stable reference
                    agent.scene_node.translation = trans
                    agent.scene_node.rotation = rot
                else:
                    # Fallback to actual transform (less stable)
                    trans = cam_node.absolute_translation
                    rot_mat = cam_node.absolute_transformation().rotation()
                    agent.scene_node.translation = trans
                    agent.scene_node.rotation = mn.Quaternion.from_matrix(rot_mat)
            else:
                # No camera controller, use actual transform
                trans = cam_node.absolute_translation
                rot_mat = cam_node.absolute_transformation().rotation()
                agent.scene_node.translation = trans
                agent.scene_node.rotation = mn.Quaternion.from_matrix(rot_mat)
    except Exception as e:
        if _debug_mode:
            print(f"⚠️ Error syncing agent camera: {e}")
    
    # ---- CAPTURE OBSERVATIONS ----
    
    try:
        # Get sensor observations
        obs = sim.get_sensor_observations()
        
        # Extract camera views
        top_img = obs['top_rgb']
        rob_img = obs['robot_rgb']
        back_img = obs['back_rgb']
        
        # Add view labels
        cv2.putText(top_img, 'TOP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(rob_img, 'FRONT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(back_img, 'BACK', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Combine views
        combined = np.hstack((top_img, rob_img, back_img))
        
    except Exception as e:
        print(f"⚠️ Error capturing observations: {e}")
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
        if abs(final_pan_pos - initial_pan_pos) > 1e-6 or abs(final_tilt_pos - initial_tilt_pos) > 1e-6:
            print(f"CAMERA DRIFT DETECTED: Pan={final_pan_pos:.6f} (Δ={final_pan_pos-initial_pan_pos:.6f}), Tilt={final_tilt_pos:.6f} (Δ={final_tilt_pos-initial_tilt_pos:.6f})")

    # Replace the scene node logging with this
    if current_time - _last_link_print_time > 5.0:
        _last_link_print_time = current_time
        print("\n=== CAMERA LINKS ===")
        # Print specific camera-related links
        for link_name in ["locobot/pan_link", "locobot/tilt_link", "camera_locobot_link"]:
            link_id = locobot.get_link_id_from_name(link_name)
            if link_id != -1:
                node = locobot.get_link_scene_node(link_id)
                print(f"{link_name}: position={node.translation}, rotation={node.rotation}")

        # Return observations and combined image
    return obs, combined