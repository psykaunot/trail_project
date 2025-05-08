import habitat_sim
import habitat_sim.physics as phys
import magnum as mn
import numpy as np

class CameraController:
    """
    Comprehensive camera control system for LoCoBot in Habitat Simulator.
    Handles camera movement, stabilization, and reference management.
    """

    # Constants for camera links identification
    CAMERA_LINKS = ["locobot/pan_link", "locobot/tilt_link", "camera_locobot_link"]
    
    # Default camera control parameters
    DEFAULT_PAN_RANGE = (-1.5, 1.5)   # Pan range in radians (-90° to +90°)
    DEFAULT_TILT_RANGE = (-0.7, 0.4)  # Tilt range in radians (-40° to +20°)
    DEFAULT_SPEED = 0.15              # Default movement speed (radians/sec)
    
    def __init__(
        self, 
        locobot, 
        motor_ids, 
        motor_settings, 
        dof_map,
        initial_pan=0.0,
        initial_tilt=-0.26,  # Slightly downward default view
        movement_speed=0.15,
        debug=False
    ):
        """Initialize the camera controller."""
        self.locobot = locobot
        self.motor_ids = motor_ids
        self.motor_settings = motor_settings
        self.dof_map = dof_map
        self.debug = debug
        
        # Current pan/tilt targets
        self.pan_target = initial_pan
        self.tilt_target = initial_tilt
        self.movement_speed = movement_speed
        
        # Store link IDs for quick access
        self.link_ids = self._get_camera_link_ids()
        
        # Reference transforms for each camera link
        self.reference_transforms = {}
        
        # Movement tracking
        self.is_moving = False
        self.last_command_time = 0
        
        # Initialize with optimal motor settings
        self._configure_motors()
        
        # Initialize reference transforms
        self._initialize_reference_transforms()
        
        if self.debug:
            self._print_debug_info()

    def _get_camera_link_ids(self):
        """Map camera link names to their IDs for efficient access."""
        link_ids = {}
        for link_name in self.CAMERA_LINKS:
            link_id = self.locobot.get_link_id_from_name(link_name)
            if link_id != -1:
                link_ids[link_name] = link_id
            elif self.debug:
                print(f"Warning: Could not find link ID for {link_name}")
        return link_ids

    def _configure_motors(self):
        """Configure motor settings for optimal camera control."""
        # Configure pan motor
        if "pan" in self.dof_map:
            pan_id = self.dof_map["pan"]
            if pan_id in self.motor_ids:
                motor_id = self.motor_ids[pan_id]
                self.motor_settings[pan_id] = phys.JointMotorSettings(
                    position_target=self.pan_target,
                    position_gain=1000.0,        # Very strong position control
                    velocity_target=0.0,
                    velocity_gain=200.0,        # High damping to prevent oscillation
                    max_impulse=1500.0          # Sufficient force for movement
                )
                self.locobot.update_joint_motor(motor_id, self.motor_settings[pan_id])
                
        # Configure tilt motor
        if "tilt" in self.dof_map:
            tilt_id = self.dof_map["tilt"]
            if tilt_id in self.motor_ids:
                motor_id = self.motor_ids[tilt_id]
                self.motor_settings[tilt_id] = phys.JointMotorSettings(
                    position_target=self.tilt_target,
                    position_gain=1000.0,        # Very strong position control
                    velocity_target=0.0,
                    velocity_gain=200.0,        # High damping to prevent oscillation
                    max_impulse=1500.0          # Sufficient force for movement
                )
                self.locobot.update_joint_motor(motor_id, self.motor_settings[tilt_id])

    def _initialize_reference_transforms(self):
        """Initialize reference transforms for all camera links."""
        for link_name, link_id in self.link_ids.items():
            node = self.locobot.get_link_scene_node(link_id)
            # Create deep copies to avoid reference issues
            self.reference_transforms[link_name] = {
                "translation": mn.Vector3(node.translation),
                "rotation": mn.Quaternion(node.rotation.vector, node.rotation.scalar),
            }
            if self.debug:
                print(f"Stored reference transform for {link_name}")

    def _print_debug_info(self):
        """Print debug information about the camera controller."""
        if not self.debug:
            return
            
        print("\n=== Camera Controller Debug Info ===")
        print(f"Tracking {len(self.link_ids)} camera links:")
        for name, id in self.link_ids.items():
            print(f"  - {name}: ID {id}")
        
        print("\nCurrent pan/tilt targets:")
        print(f"  Pan: {self.pan_target:.4f} rad")
        print(f"  Tilt: {self.tilt_target:.4f} rad")
        
        print("\nMotor configuration:")
        if "pan" in self.dof_map:
            pan_id = self.dof_map["pan"]
            if pan_id in self.motor_settings:
                settings = self.motor_settings[pan_id]
                print(f"  Pan motor: pos_gain={settings.position_gain}, vel_gain={settings.velocity_gain}")
        
        if "tilt" in self.dof_map:
            tilt_id = self.dof_map["tilt"]
            if tilt_id in self.motor_settings:
                settings = self.motor_settings[tilt_id]
                print(f"  Tilt motor: pos_gain={settings.position_gain}, vel_gain={settings.velocity_gain}")
        
        print("=====================================\n")

    def move_camera(self, pan_delta=0.0, tilt_delta=0.0):
        """
        Move the camera by the specified pan and tilt deltas.
        
        Args:
            pan_delta: Change in pan angle (radians)
            tilt_delta: Change in tilt angle (radians)
        """
        if pan_delta == 0.0 and tilt_delta == 0.0:
            return False  # No movement requested
            
        # Update target positions with limits
        new_pan = self.pan_target + pan_delta
        new_tilt = self.tilt_target + tilt_delta
        
        # Apply joint limits
        new_pan = np.clip(new_pan, self.DEFAULT_PAN_RANGE[0], self.DEFAULT_PAN_RANGE[1])
        new_tilt = np.clip(new_tilt, self.DEFAULT_TILT_RANGE[0], self.DEFAULT_TILT_RANGE[1])
        
        # Only update if there's actual change
        changed = False
        
        if new_pan != self.pan_target:
            self.pan_target = new_pan
            changed = True
            
        if new_tilt != self.tilt_target:
            self.tilt_target = new_tilt
            changed = True
            
        if not changed:
            return False
        
        if changed and self.debug:
            print(f"Camera moved to Pan: {self.pan_target:.4f}, Tilt: {self.tilt_target:.4f}")
            
        # Apply to pan motor
        if "pan" in self.dof_map:
            pan_id = self.dof_map["pan"]
            if pan_id in self.motor_ids:
                motor_id = self.motor_ids[pan_id]
                settings = self.motor_settings[pan_id]
                settings.position_target = self.pan_target
                self.locobot.update_joint_motor(motor_id, settings)
            
        # Apply to tilt motor
        if "tilt" in self.dof_map:
            tilt_id = self.dof_map["tilt"]
            if tilt_id in self.motor_ids:
                motor_id = self.motor_ids[tilt_id]
                settings = self.motor_settings[tilt_id]
                settings.position_target = self.tilt_target
                self.locobot.update_joint_motor(motor_id, settings)
                
        # Track that we're moving
        self.is_moving = True
        import time
        self.last_command_time = time.time()
        
        if self.debug:
            print(f"Camera moved: pan={self.pan_target:.4f}, tilt={self.tilt_target:.4f}")
            
        return True

    def process_key_command(self, key):
        """
        Process a key command for camera movement.
        
        Args:
            key: The key code
            
        Returns:
            bool: True if the key was processed as a camera command
        """
        # Map keys to pan/tilt movements
        camera_keys = {
            ord('c'): ("pan", -self.movement_speed),   # Pan left
            ord('v'): ("pan", self.movement_speed),    # Pan right
            ord('f'): ("tilt", self.movement_speed),   # Tilt up
            ord('r'): ("tilt", -self.movement_speed),  # Tilt down
        }
        
        if key in camera_keys:
            joint, delta = camera_keys[key]
            if joint == "pan":
                return self.move_camera(pan_delta=delta)
            else:  # tilt
                return self.move_camera(tilt_delta=delta)
                
        return False  # Not a camera control key

    def update_references(self):
        """Update reference transforms after camera movement."""
        for link_name, link_id in self.link_ids.items():
            node = self.locobot.get_link_scene_node(link_id)
            self.reference_transforms[link_name] = {
                "translation": mn.Vector3(node.translation),
                "rotation": mn.Quaternion(node.rotation.vector, node.rotation.scalar),
            }
            if self.debug:
                print(f"Updated reference transform for {link_name}")
                
    def stabilize(self):
        """
        Apply stabilization to camera links using stored reference transforms.
        Call this after physics steps when not actively commanding the camera.
        """
        if self.is_moving:
            # Check if movement has timed out (500ms since last command)
            import time
            if time.time() - self.last_command_time > 0.5:
                self.is_moving = False
                # Update references after movement completes
                self.update_references()
            else:
                # Still moving, don't stabilize
                return False
                
        # Apply stabilization to each camera link
        for link_name, link_id in self.link_ids.items():
            ref = self.reference_transforms.get(link_name)
            if not ref:
                continue
                
            # Apply position stabilization to the scene node
            node = self.locobot.get_link_scene_node(link_id)
            
            # Force the node to match the reference transform exactly
            node.translation = ref["translation"]
            node.rotation = ref["rotation"]
            
        return True

    def advance_physics(self, sim, dt, force_stabilize=False):
        """
        Advance physics with proper camera stabilization.
        
        Args:
            sim: The Habitat Simulator instance
            dt: Time delta for the frame
            force_stabilize: Force stabilization even if camera is moving
        """
        # Use smaller substeps for more stable physics
        sub_dt = 1.0 / 120.0
        num_substeps = max(8, int(dt / sub_dt))
        
        # Step physics with interleaved stabilization
        for i in range(num_substeps):
            # Step physics simulation
            sim.step_physics(sub_dt)
            
            # Apply stabilization every other substep when not moving camera
            if (not self.is_moving or force_stabilize) and i % 2 == 0:
                self.stabilize()

    def get_current_state(self):
        """Get the current camera state."""
        return {
            "pan": self.pan_target,
            "tilt": self.tilt_target,
            "is_moving": self.is_moving,
            "time_since_last_command": time.time() - self.last_command_time if self.is_moving else float('inf')
        }