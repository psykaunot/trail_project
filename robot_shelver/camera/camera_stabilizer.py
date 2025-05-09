import habitat_sim
import habitat_sim.physics as phys
import magnum as mn
import numpy as np

class CameraStabilizer:
    """
    Class for completely locking LoCoBot's RGB camera in Habitat Sim.
    
    Ensures absolute camera stability by:
    1. Completely locking camera-related links to fixed reference positions
    2. Applying position lock on every physics step
    3. Ignoring all camera movement commands
    4. Disabling joint motors for camera links
    """

    # These exact link names must be used for stabilization
    CAMERA_LINK_NAMES = [
        "locobot/pan_link",
        "locobot/tilt_link",
        "camera_locobot_link",
    ]

    def __init__(
        self,
        locobot,
        motor_settings,
        motor_ids,
        dof_map,
        pan_target: float = 0.0,
        tilt_target: float = -0.26,  # Default tilt looking slightly down
        debug: bool = False,
    ):
        """
        Initialize the static camera stabilizer.
        
        Args:
            locobot: The articulated object representing the robot
            motor_settings: Dictionary mapping motor IDs to JointMotorSettings
            motor_ids: Dictionary mapping link IDs to motor IDs
            dof_map: Dictionary mapping joint names to link IDs
            pan_target: Initial pan angle target (radians) - will be locked at this value
            tilt_target: Initial tilt angle target (radians) - will be locked at this value
            debug: Whether to print debug information
        """
        self.locobot = locobot
        self.motor_settings = motor_settings
        self.motor_ids = motor_ids
        self.dof_map = dof_map
        self.pan_target = pan_target
        self.tilt_target = tilt_target
        self.debug = debug
        
        # Dictionary to store reference transforms for each camera link
        self.reference_transforms = {}
        
        # Camera links IDs for faster access
        self.camera_link_ids = {}
        
        # Initialize camera links to desired positions
        self._initialize_camera_pose()
        
        # Store reference transforms
        self._initialize_reference_transforms()
        
        # Set high stiffness to camera joints
        self._configure_camera_motors()
        
        # Suppress camera motion commands completely
        self.ignore_camera_commands = True
        
        if self.debug:
            print("Static camera stabilizer initialized")
            print(f"Camera links will be completely fixed at current pose")
            print(f"Tracking links: {self.CAMERA_LINK_NAMES}")

    def _initialize_camera_pose(self):
        """Set initial camera pose to the desired pan/tilt configuration."""
        # Set pan joint to target angle
        if "pan" in self.dof_map:
            pan_id = self.dof_map["pan"]
            self.camera_link_ids["pan"] = pan_id
            # Get joint positions array and modify pan angle
            if pan_id in self.motor_ids:
                joint_positions = self.locobot.joint_positions
                pos_offset = self.locobot.get_link_joint_pos_offset(pan_id)
                if pos_offset >= 0:
                    joint_positions[pos_offset] = self.pan_target
                    self.locobot.joint_positions = joint_positions
                    if self.debug:
                        print(f"Set initial pan angle to {self.pan_target}")
        
        # Set tilt joint to target angle
        if "tilt" in self.dof_map:
            tilt_id = self.dof_map["tilt"]
            self.camera_link_ids["tilt"] = tilt_id
            # Get joint positions array and modify tilt angle
            if tilt_id in self.motor_ids:
                joint_positions = self.locobot.joint_positions
                pos_offset = self.locobot.get_link_joint_pos_offset(tilt_id)
                if pos_offset >= 0:
                    joint_positions[pos_offset] = self.tilt_target
                    self.locobot.joint_positions = joint_positions
                    if self.debug:
                        print(f"Set initial tilt angle to {self.tilt_target}")

    def _initialize_reference_transforms(self):
        """Store reference transforms for all camera links - these will be enforced constantly."""
        for link_name in self.CAMERA_LINK_NAMES:
            link_id = self.locobot.get_link_id_from_name(link_name)
            if link_id != -1:
                node = self.locobot.get_link_scene_node(link_id)
                # Create deep copies to avoid reference issues
                self.reference_transforms[link_name] = {
                    "translation": mn.Vector3(node.translation),
                    "rotation": mn.Quaternion(node.rotation.vector, node.rotation.scalar),
                    "link_id": link_id
                }
                if self.debug:
                    print(f"Locked reference transform for {link_name}: pos={node.translation}, rot={node.rotation}")

    def _configure_camera_motors(self):
        """Configure extremely stiff motor settings for camera stability."""
        # Configure pan motor with very high stiffness
        if "pan" in self.dof_map:
            pan_id = self.dof_map["pan"]
            motor_id = self.motor_ids.get(pan_id, pan_id)
            self.motor_settings[motor_id] = phys.JointMotorSettings(
                position_target=self.pan_target,   # Fixed target position
                position_gain=100.0,               # Very high stiffness
                velocity_target=0.0,
                velocity_gain=50.0,                # Very high damping
                max_impulse=1000.0,                # Very high force limit
            )
            if self.debug:
                print(f"Configured pan motor with maximum stiffness for static locking")
            
        # Configure tilt motor with very high stiffness
        if "tilt" in self.dof_map:
            tilt_id = self.dof_map["tilt"]
            motor_id = self.motor_ids.get(tilt_id, tilt_id)
            self.motor_settings[motor_id] = phys.JointMotorSettings(
                position_target=self.tilt_target,  # Fixed target position
                position_gain=100.0,               # Very high stiffness
                velocity_target=0.0,
                velocity_gain=50.0,                # Very high damping
                max_impulse=1000.0,                # Very high force limit
            )
            if self.debug:
                print(f"Configured tilt motor with maximum stiffness for static locking")
        
        # Apply motor settings immediately
        try:
            for lid, mid in self.motor_ids.items():
                if lid in [self.dof_map.get("pan"), self.dof_map.get("tilt")]:
                    self.locobot.update_joint_motor(mid, self.motor_settings[lid])
        except Exception as e:
            if self.debug:
                print(f"Error applying motor settings: {e}")

    def _force_static_camera(self):
        """Force all camera links to exactly match their reference transforms."""
        for link_name, ref in self.reference_transforms.items():
            link_id = ref["link_id"]
            node = self.locobot.get_link_scene_node(link_id)
            
            # Force the node to exactly match reference transform
            node.translation = ref["translation"]
            node.rotation = ref["rotation"]

    def stabilize(self, sim, dt: float, is_camera_cmd: bool) -> None:
        """
        Step physics while forcing camera links to remain perfectly static.
        
        Args:
            sim: The Habitat Simulator instance
            dt: Time delta for the frame
            is_camera_cmd: Whether user is trying to move camera (will be ignored)
        """
        # Use small timesteps for more consistent physics
        sub_dt = dt / 10.0  # Split into smaller substeps
        num_substeps = 10   # Fixed number of substeps
        
        # Step physics with static camera enforcement on every substep
        for i in range(num_substeps):
            # First enforce static transforms before physics step
            self._force_static_camera()
            
            # Step physics simulation
            sim.step_physics(sub_dt)
            
            # Immediately force static position again after physics
            self._force_static_camera()
        
        # Final enforcement after all substeps
        self._force_static_camera()

    def move_camera(self, pan: float, tilt: float) -> None:
        """
        Do nothing - camera movement is completely disabled.
        
        Args:
            pan: Ignored pan angle change
            tilt: Ignored tilt angle change
        """
        if self.debug:
            print(f"Ignoring camera movement command (pan={pan}, tilt={tilt})")
        # Camera movement is completely disabled
        pass