import habitat_sim
import habitat_sim.physics as phys
import magnum as mn
import numpy as np

class CameraStabilizer:
    """
    Class for locking LoCoBot's RGB camera in Habitat Sim for automated missions.
    
    Ensures camera stability by:
    1. Locking camera-related links to fixed reference positions
    2. Applying position lock on every physics step
    3. Applying high-gain position control
    """

    # Camera link names
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
        tilt_target: float = -0.26,
        debug: bool = False,
    ):
        """
        Initialize the camera stabilizer.
        
        Args:
            locobot: The articulated object representing the robot
            motor_settings: Dictionary mapping motor IDs to JointMotorSettings
            motor_ids: Dictionary mapping link IDs to motor IDs
            dof_map: Dictionary mapping joint names to link IDs
            pan_target: Initial pan angle target (radians)
            tilt_target: Initial tilt angle target (radians)
            debug: Whether to print debug information
        """
        self.locobot = locobot
        self.motor_settings = motor_settings
        self.motor_ids = motor_ids
        self.dof_map = dof_map
        self.pan_target = pan_target
        self.tilt_target = tilt_target
        self.debug = debug
        
        # Dictionary to store reference transforms for camera links
        self.reference_transforms = {}
        
        # Camera links IDs for faster access
        self.camera_link_ids = {}
        
        # Initialize camera links and store reference transforms
        self._initialize_camera_pose()
        self._initialize_reference_transforms()
        
        # Configure camera motors for stability
        self._configure_camera_motors()
        
        # Track pan/tilt offsets for automatic recalibration
        self.pan_offset = 0.0
        self.tilt_offset = 0.0

    def _initialize_camera_pose(self):
        """Set initial camera pose to the target configuration."""
        # Set pan joint to target angle
        if "pan" in self.dof_map:
            pan_id = self.dof_map["pan"]
            self.camera_link_ids["pan"] = pan_id
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
            if tilt_id in self.motor_ids:
                joint_positions = self.locobot.joint_positions
                pos_offset = self.locobot.get_link_joint_pos_offset(tilt_id)
                if pos_offset >= 0:
                    joint_positions[pos_offset] = self.tilt_target
                    self.locobot.joint_positions = joint_positions
                    if self.debug:
                        print(f"Set initial tilt angle to {self.tilt_target}")

    def _initialize_reference_transforms(self):
        """Store reference transforms for all camera links."""
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

    def _configure_camera_motors(self):
        """Configure motors for optimal camera stability."""
        # Configure pan motor with high stiffness
        if "pan" in self.dof_map:
            pan_id = self.dof_map["pan"]
            motor_id = self.motor_ids.get(pan_id, pan_id)
            self.motor_settings[motor_id] = phys.JointMotorSettings(
                position_target=self.pan_target,
                position_gain=1000.0,         # Very high stiffness
                velocity_target=0.0,
                velocity_gain=200.0,          # High damping
                max_impulse=1500.0            # High force limit
            )
            
        # Configure tilt motor with high stiffness
        if "tilt" in self.dof_map:
            tilt_id = self.dof_map["tilt"]
            motor_id = self.motor_ids.get(tilt_id, tilt_id)
            self.motor_settings[motor_id] = phys.JointMotorSettings(
                position_target=self.tilt_target,
                position_gain=1000.0,         # Very high stiffness
                velocity_target=0.0,
                velocity_gain=200.0,          # High damping
                max_impulse=1500.0            # High force limit
            )
        
        # Apply motor settings
        try:
            for lid, mid in self.motor_ids.items():
                if lid in [self.dof_map.get("pan"), self.dof_map.get("tilt")]:
                    self.locobot.update_joint_motor(mid, self.motor_settings[lid])
        except Exception as e:
            if self.debug:
                print(f"Error applying motor settings: {e}")

    def stabilize(self, dt: float = 1/30.0, force_static: bool = True):
        """
        Stabilize camera by applying reference transforms and high-gain motor control.
        
        Args:
            dt: Physics timestep duration
            force_static: Whether to force camera to exact position
        """
        if force_static:
            # Force all camera links to exactly match reference transforms
            for link_name, ref in self.reference_transforms.items():
                link_id = ref["link_id"]
                node = self.locobot.get_link_scene_node(link_id)
                
                # Force the node to exactly match reference transform
                node.translation = ref["translation"]
                node.rotation = ref["rotation"]
        
        # Ensure pan/tilt joints are at target positions
        if "pan" in self.dof_map:
            pan_id = self.dof_map["pan"]
            pan_offset = self.locobot.get_link_joint_pos_offset(pan_id)
            if pan_offset >= 0:
                joint_positions = self.locobot.joint_positions
                
                # Check if we need to correct
                current_pan = joint_positions[pan_offset]
                if abs(current_pan - self.pan_target) > 0.005:
                    # Direct position correction
                    joint_positions[pan_offset] = self.pan_target
                    self.locobot.joint_positions = joint_positions
                    
                    # Update motor for reinforcement
                    if pan_id in self.motor_ids:
                        motor_settings = self.motor_settings[pan_id]
                        motor_settings.position_target = self.pan_target
                        self.locobot.update_joint_motor(self.motor_ids[pan_id], motor_settings)
        
        if "tilt" in self.dof_map:
            tilt_id = self.dof_map["tilt"]
            tilt_offset = self.locobot.get_link_joint_pos_offset(tilt_id)
            if tilt_offset >= 0:
                joint_positions = self.locobot.joint_positions
                
                # Check if we need to correct
                current_tilt = joint_positions[tilt_offset]
                if abs(current_tilt - self.tilt_target) > 0.005:
                    # Direct position correction
                    joint_positions[tilt_offset] = self.tilt_target
                    self.locobot.joint_positions = joint_positions
                    
                    # Update motor for reinforcement
                    if tilt_id in self.motor_ids:
                        motor_settings = self.motor_settings[tilt_id]
                        motor_settings.position_target = self.tilt_target
                        self.locobot.update_joint_motor(self.motor_ids[tilt_id], motor_settings)
    
    def update_targets(self, pan_target=None, tilt_target=None):
        """
        Update pan/tilt target positions with smoothing.
        
        Args:
            pan_target: New pan target angle (radians)
            tilt_target: New tilt target angle (radians)
            
        Returns:
            bool: True if targets were updated
        """
        updated = False
        
        if pan_target is not None and pan_target != self.pan_target:
            self.pan_target = pan_target
            
            # Update pan motor settings
            if "pan" in self.dof_map:
                pan_id = self.dof_map["pan"]
                if pan_id in self.motor_ids:
                    motor_id = self.motor_ids[pan_id]
                    settings = self.motor_settings[pan_id]
                    settings.position_target = pan_target
                    self.locobot.update_joint_motor(motor_id, settings)
                    updated = True
        
        if tilt_target is not None and tilt_target != self.tilt_target:
            self.tilt_target = tilt_target
            
            # Update tilt motor settings
            if "tilt" in self.dof_map:
                tilt_id = self.dof_map["tilt"]
                if tilt_id in self.motor_ids:
                    motor_id = self.motor_ids[tilt_id]
                    settings = self.motor_settings[tilt_id]
                    settings.position_target = tilt_target
                    self.locobot.update_joint_motor(motor_id, settings)
                    updated = True
        
        # Update reference transforms if targets changed
        if updated:
            # Give physics time to apply
            self._initialize_reference_transforms()
        
        return updated