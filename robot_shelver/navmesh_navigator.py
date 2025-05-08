import numpy as np
import magnum as mn
import habitat_sim
import math
import time

class NavMeshNavigator:
    def __init__(self, sim, locobot, motor_ids, motor_settings, dof_map, 
                 drive_speed=0.8, turn_speed=0.4, dt=1/30.0,
                 allow_sliding=True, debug=False):
        """
        NavMesh-based navigation using velocity control and path finding.
        
        Args:
            sim: Habitat simulator instance
            locobot: Articulated robot object
            motor_ids: Dictionary mapping link IDs to motor IDs
            motor_settings: Dictionary of motor settings
            dof_map: Dictionary mapping joint names to link IDs
            drive_speed: Forward movement speed
            turn_speed: Rotation speed
            dt: Physics time step
            allow_sliding: Whether to allow sliding along navmesh edges
            debug: Print debugging information
        """
        self.sim = sim
        self.locobot = locobot
        self.motor_ids = motor_ids
        self.motor_settings = motor_settings
        self.dof_map = dof_map
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed
        self.dt = dt
        self.allow_sliding = allow_sliding
        self.debug = debug
        
        # Get pathfinder from simulator
        self.pathfinder = sim.pathfinder
        if not self.pathfinder.is_loaded:
            print("ERROR: No NavMesh loaded. Cannot navigate.")
        
        # For tracking navigation state
        self.current_path = []
        self.current_waypoint_idx = 0
        self.goal_position = None
        self.is_navigating = False
        
        # Initialize velocity control 
        if "wheel_left_joint" not in dof_map or "wheel_right_joint" not in dof_map:
            print("WARNING: Wheel joints not found in dof_map!")
        
        print("NavMesh Navigator initialized")
        if self.pathfinder.is_loaded:
            print(f"NavMesh area: {self.pathfinder.navigable_area}m²")

    
    def _apply_wheel_velocities(self, left_velocity, right_velocity):
       """Apply velocities directly to wheel joints."""
       if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
           left_id = self.dof_map["wheel_left_joint"]
           right_id = self.dof_map["wheel_right_joint"]
           
           # Add these lines to ensure proper velocity control
           self.motor_settings[left_id].position_gain = 0.0  # Must be zero for velocity control
           self.motor_settings[right_id].position_gain = 0.0
           
           # Then set velocities
           self.motor_settings[left_id].velocity_target = left_velocity
           self.motor_settings[right_id].velocity_target = right_velocity
           
           # Print for debugging
           print(f"Setting wheel velocities: L={left_velocity:.2f}, R={right_velocity:.2f}")
           
           # Apply to motors
           self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
           self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])
        
    def navigate_to(self, target_position, max_distance=0.5):
        """
        Navigate to a target position using NavMesh pathfinding.
        
        Args:
            target_position: 3D target position (numpy array or Vector3)
            max_distance: Success distance threshold
            
        Returns:
            bool: True if navigation started successfully
        """
        if not self.pathfinder.is_loaded:
            print("ERROR: NavMesh not loaded. Cannot navigate.")
            return False
            
        # Convert target to Vector3 if needed
        if isinstance(target_position, (list, tuple, np.ndarray)):
            goal_pos = mn.Vector3(target_position[0], target_position[1], target_position[2])
        else:
            goal_pos = target_position
            
        # Check if target is navigable
        if not self.pathfinder.is_navigable(goal_pos):
            print(f"Target position {goal_pos} is not navigable")
            # Try to fix y-coordinate first
            corrected_pos = mn.Vector3(goal_pos.x, 0.159348, goal_pos.z)
            if self.pathfinder.is_navigable(corrected_pos):
                print(f"Using height-corrected position: {corrected_pos}")
                goal_pos = corrected_pos
            else:
                # Try to find closest navigable point
                closest_point = self.pathfinder.snap_point(corrected_pos)
                if (closest_point - corrected_pos).length() > 2.0:
                    print(f"No navigable point found near target")
                    return False
                print(f"Using closest navigable point: {closest_point}")
                goal_pos = closest_point
        
        # Plan path to target
        self.goal_position = goal_pos
        start_pos = self.locobot.translation
        path = habitat_sim.nav.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = goal_pos
        
        found_path = self.pathfinder.find_path(path)
        if not found_path:
            print(f"No path found from {start_pos} to {goal_pos}")
            return False
            
        # Store path for navigation
        self.current_path = path.points
        self.current_waypoint_idx = 0
        self.is_navigating = True
        
        if self.debug:
            print(f"Path found with {len(self.current_path)} waypoints")
            print(f"Path length: {path.geodesic_distance}m")
            
        return True
    
    def update(self, max_distance=0.3):
        """
        Update navigation, should be called each frame.
        
        Args:
            max_distance: Distance to consider waypoint reached
            
        Returns:
            bool: True if still navigating, False if finished or error
        """
        if not self.is_navigating or len(self.current_path) == 0:
            return False
            
        # Get current position
        current_pos = self.locobot.translation
        
        # Check if we've reached the goal
        if (current_pos - self.goal_position).length() < max_distance:
            self._stop_movement()
            self.is_navigating = False
            if self.debug:
                print(f"Goal reached! Distance: {(current_pos - self.goal_position).length():.2f}m")
            return False
            
        # Determine current waypoint to follow
        if self.current_waypoint_idx < len(self.current_path):
            current_waypoint = self.current_path[self.current_waypoint_idx]
            dist_to_waypoint = (current_pos - current_waypoint).length()
            
            # If reached current waypoint, move to next one
            if dist_to_waypoint < max_distance:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.current_path):
                    # Target the final position
                    current_waypoint = self.goal_position
                else:
                    current_waypoint = self.current_path[self.current_waypoint_idx]
                    if self.debug:
                        print(f"Moving to waypoint {self.current_waypoint_idx}/{len(self.current_path)}")
        else:
            # All waypoints processed, target final position
            current_waypoint = self.goal_position
            
        # Calculate direction and control robot
        direction = current_waypoint - current_pos
        direction.y = 0  # Ignore height differences
        
        # Skip if no meaningful movement needed
        if direction.length() < 0.01:
            return True
            
        # Calculate angle between robot's forward direction and target
        robot_forward = self.locobot.rotation.transform_vector(mn.Vector3(0, 0, -1))
        robot_forward.y = 0
        robot_forward = robot_forward.normalized()
        
        target_dir = direction.normalized()
        
        # Calculate angle between vectors (positive = turn right, negative = turn left)
        dot = robot_forward.x * target_dir.x + robot_forward.z * target_dir.z
        det = robot_forward.x * target_dir.z - robot_forward.z * target_dir.x
        angle = math.atan2(det, dot)
        
        # Apply movement based on angle
        turn_threshold = 0.3  # radians (~17 degrees)
        
        if abs(angle) > turn_threshold:
            # Need to turn first before moving forward
            lin_vel = mn.Vector3(0, 0, 0)
            ang_vel = mn.Vector3(0, angle * self.turn_speed, 0)
            if self.debug and abs(angle) > 1.0:
                print(f"Turning: {math.degrees(angle):.1f}°")
        else:
            # Mostly aligned, move forward with some turning
            # Scale forward speed based on alignment (slower when turning)
            forward_scale = 1.0 - min(1.0, abs(angle) / turn_threshold * 0.5)
            lin_vel = mn.Vector3(0, 0, -self.drive_speed * forward_scale)
            ang_vel = mn.Vector3(0, angle * self.turn_speed * 0.5, 0)
            
        # Convert linear/angular velocities to differential drive wheel speeds
        if abs(angle) > turn_threshold:
            # Need to turn first before moving forward
            if angle > 0:  # Turn right
                self._apply_wheel_velocities(self.turn_speed, -self.turn_speed)
            else:  # Turn left
                self._apply_wheel_velocities(-self.turn_speed, self.turn_speed)
        else:
            # Mostly aligned, move forward with some turning
            # Scale forward speed based on alignment
            forward_speed = self.drive_speed * (1.0 - min(1.0, abs(angle) / turn_threshold * 0.5))
            turn_factor = angle * 0.5  # Reduce turning factor for smoother motion
            self._apply_wheel_velocities(
                forward_speed - turn_factor * self.turn_speed,
                forward_speed + turn_factor * self.turn_speed
            )
        
        # Snap position to navmesh if needed
        if not self.allow_sliding:
            self._snap_position_to_navmesh()
            
        return True
        
    def _snap_position_to_navmesh(self):
        """Snap agent position to navmesh to prevent sliding."""
        current_pos = self.locobot.translation
        snapped_pos = self.pathfinder.snap_point(current_pos)
        
        # Only apply if meaningful difference
        if (current_pos - snapped_pos).length() > 0.05:
            state = self.locobot.rigid_state
            state.translation = snapped_pos
            self.locobot.rigid_state = state
            
    def _stop_movement(self):
        """Stop all movement."""
        # Set velocity control to zero
        self.vel_control.linear_velocity = mn.Vector3(0, 0, 0)
        self.vel_control.angular_velocity = mn.Vector3(0, 0, 0)
        
        # Also zero wheel velocities via motors for redundancy
        if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
            left_id = self.dof_map["wheel_left_joint"]
            right_id = self.dof_map["wheel_right_joint"]
            
            # Set wheel velocities to zero
            self.motor_settings[left_id].velocity_target = 0.0
            self.motor_settings[right_id].velocity_target = 0.0
            
            # Apply settings to motors
            self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
            self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])