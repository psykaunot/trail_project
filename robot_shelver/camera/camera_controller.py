import habitat_sim
import habitat_sim.physics as phys
import magnum as mn
import numpy as np
import time
import threading
import math
import queue

class CameraController:
    """
    Autonomous camera control system for LoCoBot in Habitat Simulator.
    Provides automated scanning patterns and LLM-directed movements with strict boundary enforcement.
    """

    # Camera control parameters - strict limits
    PAN_RANGE = (-1.57, 1.57)   # Pan range in radians (≈±90°)
    TILT_RANGE = (-1.57, 1.2)   # Tilt range in radians (≈-90° to +70°)
    DEFAULT_SPEED = 0.5         # Default movement speed (radians/sec)
    
    def __init__(
        self, 
        locobot, 
        motor_ids, 
        motor_settings, 
        dof_map,
        initial_pan=0.0,
        initial_tilt=-0.26,
        movement_speed=0.2,
        debug=False
    ):
        self.locobot = locobot
        self.motor_ids = motor_ids
        self.motor_settings = motor_settings
        self.dof_map = dof_map
        self.debug = debug
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Current pan/tilt values and targets
        self.pan_current = initial_pan
        self.pan_target = initial_pan
        self.tilt_current = initial_tilt
        self.tilt_target = initial_tilt
        self.movement_speed = movement_speed
        
        # Movement tracking and control
        self.is_moving = False
        self.is_scanning = False
        self.scanning_thread = None
        self.stop_scanning_flag = threading.Event()
        self.movement_lock = threading.Lock()
        
        # Predefined scan patterns
        self._initialize_scan_patterns()
        self.scan_pause_time = 5.0
        self.current_scan_index = 0
        
        # Link IDs for fast access
        self.pan_link_id = self.dof_map.get("pan", -1)
        self.tilt_link_id = self.dof_map.get("tilt", -1)
        
        # Movement mode tracking
        self.movement_mode = "hold"  # "hold" or "move"

        self.locked = True
        self.has_movement_command = False
        self.command_complete_time = 0.0
        
        self._update_joint_positions()
        self.locked_pan = self.pan_current
        self.locked_tilt = self.tilt_current
        
        if self.pan_link_id == -1 or self.tilt_link_id == -1:
            print("WARNING: Pan/tilt joints not found in dof_map")
        
        # Initialize with optimal motor settings
        self._configure_motors()
    
    @property
    def is_locked_and_stationary(self):
        """Check if camera should be held at locked position."""
        return self.locked and not self.is_moving and not self.has_movement_command


    def sync_camera_sensor(self, agent):
        """Sync the robot_rgb sensor orientation with pan/tilt joints."""
        if agent is None or len(agent.sensors) == 0:
            return

        # Get sensor by name
        sensor = None
        for s in agent.sensors:
            if s.uuid == "robot_rgb":
                sensor = s
                break
            
        if sensor is None:
            return

        # Update sensor orientation based on pan/tilt
        sensor.node.rotation = mn.Quaternion.rotation(
            mn.Rad(self.pan_current), mn.Vector3(0, 1, 0)
        ) * mn.Quaternion.rotation(
            mn.Rad(self.tilt_current), mn.Vector3(1, 0, 0)
        )

    def _initialize_scan_patterns(self):
        """Initialize scanning patterns with corrected tilt values."""
        self.patterns = {
            "book_search": [
                # Floor level - primary focus (positive tilt = looking down)
                {"pan": -0.5, "tilt": 1.0},
                {"pan": -0.25, "tilt": 1.1},  
                {"pan": 0.0, "tilt": 1.2},     # Maximum downward
                {"pan": 0.25, "tilt": 1.1},
                {"pan": 0.5, "tilt": 1.0},

                # Mid level - tables/low surfaces
                {"pan": -0.5, "tilt": 0.6},
                {"pan": 0.0, "tilt": 0.7},
                {"pan": 0.5, "tilt": 0.6},

                # Upper level - shelves (negative tilt = looking up)
                {"pan": -0.5, "tilt": -0.2},
                {"pan": 0.0, "tilt": -0.1},
                {"pan": 0.5, "tilt": -0.2},
            ],
            "floor_focused": [
                # Dense floor scanning
                {"pan": -0.5, "tilt": 1.2},
                {"pan": -0.3, "tilt": 1.1},
                {"pan": -0.1, "tilt": 1.2},
                {"pan": 0.1, "tilt": 1.2},
                {"pan": 0.3, "tilt": 1.1},
                {"pan": 0.5, "tilt": 1.2},
            ],
            "vertical_sweep": [
                # Vertical scans at key positions
                {"pan": -0.4, "tilt": -0.2},
                {"pan": -0.4, "tilt": 0.6},
                {"pan": -0.4, "tilt": 1.2},

                {"pan": 0.0, "tilt": -0.2},
                {"pan": 0.0, "tilt": 0.6},
                {"pan": 0.0, "tilt": 1.2},

                {"pan": 0.4, "tilt": -0.2},
                {"pan": 0.4, "tilt": 0.6},
                {"pan": 0.4, "tilt": 1.2},
            ]
        }
        
        # Set default pattern
        self.scan_positions = self.patterns["book_search"]
    
    def _generate_continuous_pattern(self, pan_steps, tilt_steps):
        """Generate a dense continuous scan pattern."""
        pattern = []
        
        # Calculate evenly spaced positions
        pan_range = self.PAN_RANGE[1] - self.PAN_RANGE[0]
        tilt_range = self.TILT_RANGE[1] - self.TILT_RANGE[0]
        
        for i in range(tilt_steps):
            tilt = self.TILT_RANGE[0] + (tilt_range * i / (tilt_steps - 1))
            # Alternate direction for more efficient scanning
            pan_values = range(pan_steps) if i % 2 == 0 else reversed(range(pan_steps))
            
            for j in pan_values:
                pan = self.PAN_RANGE[0] + (pan_range * j / (pan_steps - 1))
                pattern.append({"pan": pan, "tilt": tilt})
        
        return pattern
    
    def _configure_motors(self):
        """Configure motor settings based on current movement mode."""
        if self.movement_mode == "hold":
            # Strong position control for holding position
            pan_gain = 2000.0
            tilt_gain = 2000.0
            damping = 200.0
            max_impulse = 5000.0
        else:
            # INCREASED gains for movement mode - this is the key fix
            pan_gain = 800.0  # Increased from 0.3
            tilt_gain = 800.0  # Increased from 0.3
            damping = 100.0    # Increased from 0.05
            max_impulse = 2000.0  # Increased from 1.0

        # Configure pan motor
        if self.pan_link_id in self.motor_ids:
            self.motor_settings[self.pan_link_id] = phys.JointMotorSettings(
                position_target=self.pan_target,
                position_gain=pan_gain,
                velocity_target=0.0,
                velocity_gain=damping,
                max_impulse=max_impulse
            )
            self.locobot.update_joint_motor(
                self.motor_ids[self.pan_link_id], 
                self.motor_settings[self.pan_link_id]
            )

        # Configure tilt motor
        if self.tilt_link_id in self.motor_ids:
            self.motor_settings[self.tilt_link_id] = phys.JointMotorSettings(
                position_target=self.tilt_target,
                position_gain=tilt_gain,
                velocity_target=0.0,
                velocity_gain=damping,
                max_impulse=max_impulse
            )
            self.locobot.update_joint_motor(
                self.motor_ids[self.tilt_link_id], 
                self.motor_settings[self.tilt_link_id]
            )

        if self.debug:
            print(f"Motors configured for {self.movement_mode} mode: gain={pan_gain}")
    
    def process_commands(self):
        """Process pending commands from the queue - must be called from main thread."""
        try:
            command = self.command_queue.get_nowait()
            if command['action'] == 'stabilize':
                self.stabilize()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing camera command: {e}")

    def _update_joint_positions(self):
        """Update internal tracking of current joint positions."""
        # Get current pan position
        if self.pan_link_id != -1:
            pos_offset = self.locobot.get_link_joint_pos_offset(self.pan_link_id)
            if pos_offset >= 0:
                self.pan_current = self.locobot.joint_positions[pos_offset]
        
        # Get current tilt position
        if self.tilt_link_id != -1:
            pos_offset = self.locobot.get_link_joint_pos_offset(self.tilt_link_id)
            if pos_offset >= 0:
                self.tilt_current = self.locobot.joint_positions[pos_offset]
    
    def move_camera(self, pan_delta=0.0, tilt_delta=0.0, absolute=False):
        """Move the camera with smooth motion."""
        with self.movement_lock:
            self.has_movement_command = True
            self._update_joint_positions()
            self.is_initialized = True

            if self.locked:
                self.locked = False
                if self.debug:
                    print("DEBUG: Camera UNLOCKED for movement")

            # Debug current state
            if self.debug:
                print(f"DEBUG: Camera move - Current: pan={self.pan_current:.4f}, tilt={self.tilt_current:.4f}")

            # Set target position
            if absolute:
                new_pan = pan_delta
                new_tilt = tilt_delta
            else:
                new_pan = self.pan_current + pan_delta
                new_tilt = self.tilt_current + tilt_delta

            # Apply limits
            new_pan = np.clip(new_pan, self.PAN_RANGE[0], self.PAN_RANGE[1])
            new_tilt = np.clip(new_tilt, self.TILT_RANGE[0], self.TILT_RANGE[1])

            if self.debug:
                print(f"DEBUG: Target: pan={new_pan:.4f}, tilt={new_tilt:.4f}")

            # Only update if there's actual change
            if abs(new_pan - self.pan_target) < 0.001 and abs(new_tilt - self.tilt_target) < 0.001:
                if self.debug:
                    print("DEBUG: No significant movement needed")
                return False

            # Update target positions
            self.pan_target = new_pan
            self.tilt_target = new_tilt
            
            # Switch to movement mode and reconfigure motors
            self.movement_mode = "move"
            self._configure_motors()
            
            self.is_moving = True
            return True
    
    def stabilize(self):
        """Stabilize camera position with active movement control."""
        # Always update current positions
        self._update_joint_positions()

        # Calculate position differences
        pan_diff = abs(self.pan_current - self.pan_target)
        tilt_diff = abs(self.tilt_current - self.tilt_target)
        tolerance = 0.02

        # Check if we need to move
        needs_movement = pan_diff > tolerance or tilt_diff > tolerance

        if needs_movement:
            # Switch to movement mode if not already
            if self.movement_mode != "move":
                self.movement_mode = "move"
                self._configure_motors()

            # Apply direct position updates
            self._apply_position_updates()

            # Update motor commands
            self._update_motor_commands()

            self.is_moving = True
        else:
            # Target reached - switch to hold mode
            if self.movement_mode != "hold":
                self.movement_mode = "hold"
                self._configure_motors()

            # Lock to current position
            self._lock_current_position()
            self.is_moving = False

    def _apply_position_updates(self):
        joint_positions = self.locobot.joint_positions

        # Force positions more directly
        if self.pan_link_id != -1:
            pos_offset = self.locobot.get_link_joint_pos_offset(self.pan_link_id)
            if pos_offset >= 0:
                joint_positions[pos_offset] = self.pan_target

        if self.tilt_link_id != -1:
            pos_offset = self.locobot.get_link_joint_pos_offset(self.tilt_link_id)
            if pos_offset >= 0:
                joint_positions[pos_offset] = self.tilt_target

        self.locobot.joint_positions = joint_positions


    def _update_motor_commands(self):
        """Update motor commands for active movement."""
        # Pan motor
        if self.pan_link_id in self.motor_ids:
            self.motor_settings[self.pan_link_id].position_target = self.pan_target
            self.locobot.update_joint_motor(
                self.motor_ids[self.pan_link_id], 
                self.motor_settings[self.pan_link_id]
            )

        # Tilt motor
        if self.tilt_link_id in self.motor_ids:
            self.motor_settings[self.tilt_link_id].position_target = self.tilt_target
            self.locobot.update_joint_motor(
                self.motor_ids[self.tilt_link_id], 
                self.motor_settings[self.tilt_link_id]
            )

    def _lock_current_position(self):
        """Lock joints to current position."""
        joint_positions = self.locobot.joint_positions

        # Force exact target positions
        if self.pan_link_id != -1:
            pos_offset = self.locobot.get_link_joint_pos_offset(self.pan_link_id)
            if pos_offset >= 0:
                joint_positions[pos_offset] = self.pan_target

        if self.tilt_link_id != -1:
            pos_offset = self.locobot.get_link_joint_pos_offset(self.tilt_link_id)
            if pos_offset >= 0:
                joint_positions[pos_offset] = self.tilt_target

        self.locobot.joint_positions = joint_positions
    
    def _force_joint_positions(self):
        """Directly force joint positions toward targets."""
        if not self.is_moving:
            return

        # Pan movement
        if self.pan_link_id != -1:
            pos_offset = self.locobot.get_link_joint_pos_offset(self.pan_link_id)
            if pos_offset >= 0:
                joint_positions = self.locobot.joint_positions
                current = joint_positions[pos_offset]

                # Larger step size for faster movement
                max_step = 0.05  # Increased from 0.02
                diff = self.pan_target - current

                if abs(diff) > max_step:
                    new_pos = current + max_step * (1 if diff > 0 else -1)
                else:
                    new_pos = self.pan_target

                joint_positions[pos_offset] = new_pos
                self.locobot.joint_positions = joint_positions

        # Same for tilt
        if self.tilt_link_id != -1:
            pos_offset = self.locobot.get_link_joint_pos_offset(self.tilt_link_id)
            if pos_offset >= 0:
                joint_positions = self.locobot.joint_positions
                current = joint_positions[pos_offset]

                max_step = 0.05  # Increased step size
                diff = self.tilt_target - current

                if abs(diff) > max_step:
                    new_pos = current + max_step * (1 if diff > 0 else -1)
                else:
                    new_pos = self.tilt_target

                joint_positions[pos_offset] = new_pos
                self.locobot.joint_positions = joint_positions
    
    def start_scanning(self, pattern_name=None, pause_time=None):
        """
        Start automated environment scanning with the camera.

        Args:
            pattern_name (str): Name of scan pattern to use ("wide", "detailed", etc.)
            pause_time (float): Time in seconds to pause at each position
        Returns:
            bool: True if scanning started successfully
        """
        if self.is_scanning:
            print("Already scanning")
            return False

        # Unlock if necessary
        if self.locked:
            self.locked = False
            if self.debug:
                print("DEBUG: Camera UNLOCKED for scanning")

        # Stop any ongoing scan thread
        self.stop_scanning_flag.set()
        if self.scanning_thread and self.scanning_thread.is_alive():
            self.scanning_thread.join(1.0)

        # Reset for new scan
        self.stop_scanning_flag.clear()

        # Select pattern
        if pattern_name and pattern_name in self.patterns:
            self.scan_positions = self.patterns[pattern_name]
            print(f"Using scan pattern '{pattern_name}' with {len(self.scan_positions)} positions")

        # Apply custom pause time if given
        if pause_time is not None:
            self.scan_pause_time = pause_time

        # Launch scanning thread
        self.is_scanning = True
        self.scanning_thread = threading.Thread(target=self._scanning_task, daemon=True)
        self.scanning_thread.start()

        print(f"Started automated scanning with {len(self.scan_positions)} positions (pause {self.scan_pause_time}s each)")
        return True
    
    def stop_scanning(self):
        """Stop the automated scanning."""
        if not self.is_scanning:
            return False
        
        self.stop_scanning_flag.set()
        if self.scanning_thread and self.scanning_thread.is_alive():
            self.scanning_thread.join(1.0)
        
        self.is_scanning = False
        print("Stopped automated scanning")
        return True
    
    def _scanning_task(self):
        """Execute scanning pattern with movement and observation."""
        try:
            scan_index = 0
            total_positions = len(self.scan_positions)
    
            if total_positions == 0:
                print("ERROR: No scan positions defined")
                return
    
            print(f"Beginning scan with {total_positions} positions")
    
            while not self.stop_scanning_flag.is_set() and scan_index < total_positions:
                # Get target position
                target = self.scan_positions[scan_index]
                target_pan = target["pan"]
                target_tilt = target["tilt"]
    
                print(f"Scanning position {scan_index + 1}/{total_positions}")
                
                # Force position directly for extreme angles
                if abs(target_tilt) > 1.0:
                    joint_positions = self.locobot.joint_positions
                    if self.tilt_link_id != -1:
                        pos_offset = self.locobot.get_link_joint_pos_offset(self.tilt_link_id)
                        if pos_offset >= 0:
                            joint_positions[pos_offset] = target_tilt
                    self.locobot.joint_positions = joint_positions
    
                # Start movement
                self.move_camera(target_pan, target_tilt, absolute=True)
    
                # Wait for position to be reached
                start_time = time.time()
                timeout = 10.0
                tolerance = 0.05
                position_reached = False
                
                while time.time() - start_time < timeout and not self.stop_scanning_flag.is_set():
                    self._update_joint_positions()
                    pan_diff = abs(self.pan_current - target_pan)
                    tilt_diff = abs(self.tilt_current - target_tilt)
    
                    if pan_diff <= tolerance and tilt_diff <= tolerance:
                        position_reached = True
                        break
                    
                    self.stabilize()
                    time.sleep(0.05)
                
                # If position reached, pause for observation
                if position_reached:
                    # Switch to hold mode
                    self.movement_mode = "hold"
                    self._configure_motors()
                    
                    # Observe for specified time
                    observe_start = time.time()
                    checked_at_this_position = False
                    
                    while time.time() - observe_start < self.scan_pause_time and not self.stop_scanning_flag.is_set():
                        self.stabilize()
                        self._force_joint_positions()
                        
                        # Check for books once at each position after 1 second
                        if time.time() - observe_start > 1.0 and not checked_at_this_position:
                            checked_at_this_position = True
                            print(f"Checking for books at position {scan_index + 1}/{total_positions}")
                            
                            # Queue book check command for main thread to process
                            self.command_queue.put({
                                'action': 'check_books_at_position',
                                'position': scan_index,
                                'pan': self.pan_current,
                                'tilt': self.tilt_current
                            })
                        
                        time.sleep(0.1)
    
                # Move to next position
                scan_index += 1
    
        except Exception as e:
            print(f"ERROR in scanning task: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_scanning = False
            self.movement_mode = "hold"
            self._configure_motors()
    
    def get_current_state(self):
        """Get the current camera state."""
        self._update_joint_positions()
        return {
            "pan": self.pan_current,
            "tilt": self.tilt_current,
            "pan_target": self.pan_target,
            "tilt_target": self.tilt_target,
            "is_moving": self.is_moving,
            "is_scanning": self.is_scanning,
            "pan_range": self.PAN_RANGE,
            "tilt_range": self.TILT_RANGE
        }
    
    def execute_llm_command(self, command_obj):
        """
        Execute a camera command from the LLM with step-by-step reasoning.
        
        Args:
            command_obj: Dictionary with command parameters
                - action: Command action name
                - reasoning: LLM reasoning about the command
                - parameters: Command-specific parameters
                
        Returns:
            Dictionary with command result and observation
        """
        # Extract command components
        action = command_obj.get("action", "").lower()
        reasoning = command_obj.get("reasoning", "No reasoning provided")
        parameters = command_obj.get("parameters", {})
        
        print(f"Camera command received: {action}")
        print(f"Reasoning: {reasoning}")
        
        result = {"success": False, "action": action, "observation": ""}
        
        if action == "move":
            # Handle absolute or relative movement
            if parameters.get("absolute", False):
                pan = parameters.get("pan", self.pan_current)
                tilt = parameters.get("tilt", self.tilt_current)
                success = self.move_camera(pan, tilt, absolute=True)
                result["observation"] = f"Camera moving to absolute position pan={pan:.2f}, tilt={tilt:.2f}"
            else:
                pan_delta = parameters.get("pan_delta", 0.0)
                tilt_delta = parameters.get("tilt_delta", 0.0)
                success = self.move_camera(pan_delta, tilt_delta)
                result["observation"] = f"Camera moving by pan_delta={pan_delta:.2f}, tilt_delta={tilt_delta:.2f}"
            
            result["success"] = success
            
        elif action == "scan":
            # Handle scanning commands
            pattern = parameters.get("pattern")
            if not pattern:
                pattern = "wide"  # Default to wide pattern
            
            pause_time = parameters.get("pause_time")
            success = self.start_scanning(pattern_name=pattern, pause_time=pause_time)
            
            result["success"] = success
            result["observation"] = f"Started scanning with pattern '{pattern}'"
            
        elif action == "stop":
            # Stop scanning
            success = self.stop_scanning()
            result["success"] = success
            result["observation"] = "Stopped scanning" if success else "No active scanning to stop"
            
        elif action == "look_at":
            # Look at specific direction
            direction = parameters.get("direction", "").lower()
            
            if direction == "center":
                success = self.move_camera(0.0, -0.3, absolute=True)
                result["observation"] = "Looking at center"
            elif direction == "left":
                success = self.move_camera(-1.0, -0.3, absolute=True)
                result["observation"] = "Looking left"
            elif direction == "right":
                success = self.move_camera(1.0, -0.3, absolute=True)
                result["observation"] = "Looking right"
            elif direction == "up":
                success = self.move_camera(self.pan_current, -0.1, absolute=True)
                result["observation"] = "Looking up"
            elif direction == "down":
                success = self.move_camera(self.pan_current, -0.5, absolute=True)
                result["observation"] = "Looking down"
            else:
                success = False
                result["observation"] = f"Unknown direction: {direction}"
                
            result["success"] = success
            
        elif action == "sweep":
            # Perform a gradual sweep from one pan position to another
            start_pan = parameters.get("start_pan", -1.5)
            end_pan = parameters.get("end_pan", 1.5)
            tilt = parameters.get("tilt", -0.3)
            steps = parameters.get("steps", 5)
            
            # Ensure within limits
            start_pan = max(self.PAN_RANGE[0], min(start_pan, self.PAN_RANGE[1]))
            end_pan = max(self.PAN_RANGE[0], min(end_pan, self.PAN_RANGE[1]))
            tilt = max(self.TILT_RANGE[0], min(tilt, self.TILT_RANGE[1]))
            
            # Generate sweep pattern
            sweep_pattern = []
            for i in range(steps):
                pan = start_pan + (end_pan - start_pan) * i / (steps - 1)
                sweep_pattern.append({"pan": pan, "tilt": tilt})
            
            # Set and start custom sweep
            self.scan_positions = sweep_pattern
            success = self.start_scanning(pause_time=parameters.get("pause_time", 2.0))
            
            result["success"] = success
            result["observation"] = f"Starting sweep from pan={start_pan:.2f} to pan={end_pan:.2f} at tilt={tilt:.2f}"
            
        else:
            # Unknown command
            result["observation"] = f"Unknown action: {action}"
        
        return result
    
    def lock_camera(self):
        """Lock the camera at its current position."""
        self._update_joint_positions()
        self.locked = True
        self.pan_target = self.pan_current
        self.tilt_target = self.tilt_current
        self.movement_mode = "hold"
        self._configure_motors()
        if self.debug:
            print(f"DEBUG: Camera LOCKED at current position")
        return True
    
    def should_apply_physics_constraint(self):
        """Check if physics constraint should be applied."""
        return self.locked and not self.has_movement_command and not self.is_moving