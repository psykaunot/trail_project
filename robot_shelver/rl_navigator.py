import os
import time
import numpy as np
import magnum as mn
import habitat_sim
from habitat_sim.nav import GreedyGeodesicFollower, ShortestPath

class RLNavigator:
    def __init__(self, sim, agent, locobot, motor_ids, motor_settings, dof_map, 
                 drive_speed=1.0, turn_speed=0.5, dt=1/30.0):
        """Initialize the RL Navigator with proper follower configuration."""
        self.sim = sim
        self.agent = agent
        self.locobot = locobot
        self.motor_ids = motor_ids
        self.motor_settings = motor_settings
        self.dof_map = dof_map
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed
        self.dt = dt
        
        # Critical: Create a proper agent state synchronized with the robot
        self.initialize_agent_state()
        
        # Initialize path follower with proper action space mapping
        self.pathfinder = sim.pathfinder
        self.path_follower = None
        
        if self.pathfinder is not None and self.pathfinder.is_loaded:
            print("Initializing GreedyGeodesicFollower for navigation...")
            try:
                # Define action space mapping for the follower
                action_mapping = {
                    "move_forward": habitat_sim.agent.ActionSpec(
                        "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
                    ),
                    "turn_left": habitat_sim.agent.ActionSpec(
                        "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
                    ),
                    "turn_right": habitat_sim.agent.ActionSpec(
                        "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
                    ),
                }
                
                # Make sure agent has this action space
                self.agent.agent_config.action_space = action_mapping
                
                # Create the follower with proper parameters
                self.path_follower = GreedyGeodesicFollower(
                    pathfinder=self.pathfinder,
                    agent=self.agent,
                    forward_key="move_forward",
                    left_key="turn_left",
                    right_key="turn_right"
                )
                print("GreedyGeodesicFollower initialized successfully")
            except Exception as e:
                print(f"Error initializing GreedyGeodesicFollower: {e}")
                import traceback
                traceback.print_exc()
                self.path_follower = None
        else:
            print("WARNING: NavMesh not loaded. Path follower cannot be initialized.")
            
    def initialize_agent_state(self):
        """Initialize agent state to match robot position."""
        try:
            robot_state = self.locobot.rigid_state
            self.agent.state.position = robot_state.translation
            self.agent.state.rotation = robot_state.rotation
            print(f"Agent state initialized to match robot at {robot_state.translation}")
        except Exception as e:
            print(f"Error initializing agent state: {e}")

    def explore_environment(self, num_points=3, max_distance=2.0, verbose=True):
        """
        Explore the environment by navigating to random navigable points.

        Args:
            num_points: Number of random points to visit
            max_distance: Maximum distance for random points
            verbose: Whether to print progress
        """
        if self.pathfinder is None or not self.pathfinder.is_loaded:
            if verbose:
                print("Pathfinder not available. Cannot explore.")
            return False

        if verbose:
            print(f"Starting RL-based exploration with {num_points} random points")

        # Get current position
        current_pos = self.locobot.translation

        # Visit random points
        points_visited = 0
        for i in range(num_points):
            if verbose:
                print(f"\nExploring point {i+1}/{num_points}")

            # Get random navigable point near current position
            try:
                target_point = self.pathfinder.get_random_navigable_point_near(
                    circle_center=current_pos,
                    radius=max_distance
                )
            except Exception as e:
                if verbose:
                    print(f"Error getting random point: {e}")
                try:
                    # Fallback to any random point
                    target_point = self.pathfinder.get_random_navigable_point()
                except:
                    if verbose:
                        print("Could not get any random navigable point")
                    continue

            # Navigate to the point
            success = self.navigate_to_point(target_point, verbose=verbose)

            if success:
                points_visited += 1
                # Update current position
                current_pos = self.locobot.translation

            # Small pause between navigation targets
            import time
            time.sleep(0.5)

        if verbose:
            print(f"\nExploration completed. Visited {points_visited}/{num_points} points")

        return points_visited > 0
    
    def physics_explore(self):
        """Execute physics-based exploration to a random point."""
        try:
            # Get a valid random point with correct height coordinate
            target_point = self.pathfinder.get_random_navigable_point()
            print(f"Navigating to: {target_point}")

            # Use the direct wheel control navigation method
            success = self.test_wheel_motors(duration=3.0, verbose=True)

            if success:
                print("Basic wheel test succeeded!")
                # Now try actual navigation
                success = self.fixed_navigate_to_point(target_point, verbose=True)
                print(f"Navigation {'succeeded' if success else 'failed'}")

        except Exception as e:
            print(f"Error during exploration: {e}")
            import traceback
            traceback.print_exc()

        return True
    
        # Add to rl_navigator.py:
    def direct_wheel_test(self):
        """Direct wheel control test using motor joints"""
        print("\n=== DIRECT WHEEL TEST ===")
        print(f"Initial position: {self.locobot.translation}")

        # Get wheel joint IDs
        if "wheel_left_joint" not in self.dof_map or "wheel_right_joint" not in self.dof_map:
            print("ERROR: Wheel joints not found!")
            return False

        left_id = self.dof_map["wheel_left_joint"]
        right_id = self.dof_map["wheel_right_joint"]

        print(f"Left wheel joint ID: {left_id}, Right wheel joint ID: {right_id}")

        # Use very low velocity and impulse for stability
        wheel_velocity = 0.5
        impulse = 100.0  # Much lower than the 400.0 in initialization

        # Set wheel velocities directly through motor settings
        self.motor_settings[left_id] = habitat_sim.physics.JointMotorSettings(
            position_target=0.0,
            position_gain=0.0,  # Pure velocity control
            velocity_target=wheel_velocity,
            velocity_gain=5.0,  # Lower gain
            max_impulse=impulse
        )

        self.motor_settings[right_id] = habitat_sim.physics.JointMotorSettings(
            position_target=0.0,
            position_gain=0.0,  # Pure velocity control
            velocity_target=wheel_velocity,
            velocity_gain=5.0,
            max_impulse=impulse
        )

        # Apply motor settings
        print(f"Setting wheel velocities: left={wheel_velocity}, right={wheel_velocity}")
        self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
        self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

        # Step physics with frequent logging
        duration = 5.0  # Run for 5 seconds
        steps = int(duration / self.dt)
        initial_pos = self.locobot.translation

        for i in range(steps):
            # Step physics with smaller time steps for stability
            self.sim.step_physics(self.dt/2)

            # Print position every 10 steps
            if i % 10 == 0:
                current_pos = self.locobot.translation
                movement = (current_pos - initial_pos).length()
                print(f"Step {i}: Position={current_pos}, Movement={movement:.4f}m")

                # Update agent to match robot
                self.agent.scene_node.translation = current_pos
                self.agent.scene_node.rotation = self.locobot.rotation

        # Stop wheels
        self.motor_settings[left_id].velocity_target = 0.0
        self.motor_settings[right_id].velocity_target = 0.0
        self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
        self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

        # Final report
        final_pos = self.locobot.translation
        total_movement = (final_pos - initial_pos).length()
        print(f"\nTest complete: Total movement = {total_movement:.4f}m")
    
        return True
    
    def fixed_navigate_to_point(self, target_position, verbose=True):
        """Navigate to target with synchronized agent-robot movement."""
        # Get wheel joint IDs
        left_id = self.dof_map["wheel_left_joint"]
        right_id = self.dof_map["wheel_right_joint"]

        # Plan path to target
        start_pos = self.locobot.translation
        path = habitat_sim.nav.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = target_position

        if not self.pathfinder.find_path(path):
            print("No path found")
            return False

        # Follow path with physical robot
        waypoints = path.points
        current_waypoint = 0

        while current_waypoint < len(waypoints):
            target = waypoints[current_waypoint]

            # Calculate direction to target
            robot_pos = self.locobot.translation
            direction = target - robot_pos
            direction.y = 0  # Ignore height
            distance = direction.length()

            # If close enough to waypoint, move to next
            if distance < 0.5:
                current_waypoint += 1
                continue

            # Calculate forward direction of robot
            robot_forward = self.locobot.rotation.transform_vector(mn.Vector3(0, 0, -1))
            robot_forward.y = 0

            # Calculate angle between forward and target
            dot = robot_forward.dot(direction.normalized())
            angle = np.arccos(np.clip(dot, -1.0, 1.0))

            # Determine turn direction using cross product
            cross = robot_forward.cross(direction)
            turn_dir = 1 if cross.y > 0 else -1

            # Apply wheel velocities based on direction
            speed = 0.3  # Lower speed for stability
            turn_amount = min(0.5, angle * turn_dir)

            # Set motor parameters for velocity control
            left_vel = speed - turn_amount * 0.5
            right_vel = speed + turn_amount * 0.5

            self.motor_settings[left_id] = habitat_sim.physics.JointMotorSettings(
                position_target=0.0,
                position_gain=0.0,  # Pure velocity control
                velocity_target=left_vel,
                velocity_gain=5.0,  # Lower gain for stability
                max_impulse=200.0   # Lower impulse prevents overflow
            )

            self.motor_settings[right_id] = habitat_sim.physics.JointMotorSettings(
                position_target=0.0,
                position_gain=0.0,
                velocity_target=right_vel,
                velocity_gain=5.0,
                max_impulse=200.0
            )

            # Apply motor settings
            self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
            self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

            # Run physics simulation
            self.sim.step_physics(self.dt)

            # CRITICAL: Sync agent position with robot
            self.agent.scene_node.translation = self.locobot.translation
            self.agent.scene_node.rotation = self.locobot.rotation

            # Update progress periodically
            if verbose and np.random.random() < 0.05:
                print(f"Distance to waypoint: {distance:.2f}m")

        # Stop wheels after reaching target
        self.motor_settings[left_id].velocity_target = 0.0
        self.motor_settings[right_id].velocity_target = 0.0
        self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
        self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

        return True

        # Updated navigate_to_point method for rl_navigator.py
    def navigate_to_point(self, target_position, verbose=True):
        """
        Navigate to a target position using continuous physics-based wheel control.

        Args:
            target_position: 3D target position to navigate to
            verbose: Whether to print navigation progress

        Returns:
            bool: Success or failure of navigation
        """
        if self.pathfinder is None or not self.pathfinder.is_loaded:
            print("ERROR: Pathfinder not initialized. Cannot navigate.")
            return False

        if verbose:
            print(f"Starting physics-based navigation to: {target_position}")

        # Convert target to Vector3 if needed
        if isinstance(target_position, (list, tuple, np.ndarray)):
            goal_pos = mn.Vector3(target_position[0], target_position[1], target_position[2])
        else:
            goal_pos = target_position

        # Check if target is navigable
        if not self.pathfinder.is_navigable(goal_pos):
            if verbose:
                print(f"Target position {goal_pos} is not navigable")
            # Try to find closest navigable point
            closest_point = self.pathfinder.snap_point(goal_pos)
            if (closest_point - goal_pos).length() > 2.0:
                if verbose:
                    print(f"No navigable point found near target")
                return False
            if verbose:
                print(f"Using closest navigable point: {closest_point}")
            goal_pos = closest_point

        # Plan path to target
        start_pos = self.locobot.translation
        path = ShortestPath()
        path.requested_start = start_pos
        path.requested_end = goal_pos

        found_path = self.pathfinder.find_path(path)
        if not found_path:
            if verbose:
                print(f"No path found from {start_pos} to {goal_pos}")
            return False

        if verbose:
            print(f"Path found with length: {path.geodesic_distance}m")

        # Use the geodesic path for navigation
        path_points = path.points
        if len(path_points) == 0:
            if verbose:
                print("Path has no points")
            return False

        # Navigation parameters
        target_distance_threshold = 0.5  # Success distance threshold
        waypoint_threshold = 0.8         # Distance to consider waypoint reached
        max_steps = 1000                 # Maximum physics steps for safety
        steps = 0
        current_waypoint_idx = 0

        # Main navigation loop
        while steps < max_steps:
            steps += 1

            # Get current position and next waypoint
            current_pos = self.locobot.translation

            # Check if we've reached the final target
            dist_to_goal = (current_pos - goal_pos).length()
            if dist_to_goal < target_distance_threshold:
                if verbose:
                    print(f"Reached target (distance: {dist_to_goal:.2f}m)")
                self._stop_motors()
                return True

            # Determine current waypoint to follow
            if current_waypoint_idx < len(path_points):
                current_waypoint = path_points[current_waypoint_idx]
                dist_to_waypoint = (current_pos - current_waypoint).length()

                # If reached current waypoint, move to next one
                if dist_to_waypoint < waypoint_threshold:
                    current_waypoint_idx += 1
                    if current_waypoint_idx < len(path_points):
                        if verbose and steps % 20 == 0:
                            print(f"Moving to waypoint {current_waypoint_idx}/{len(path_points)}")
                        current_waypoint = path_points[current_waypoint_idx]
                    else:
                        # All waypoints visited, target final position
                        current_waypoint = goal_pos
            else:
                # All waypoints processed, target final position
                current_waypoint = goal_pos

            # Calculate direction and control robot
            direction = current_waypoint - current_pos
            direction.y = 0  # Ignore height differences

            # Skip if no meaningful movement needed
            if direction.length() < 0.001:
                continue

            # Calculate 2D angle between robot's forward direction and target
            robot_forward = self.locobot.rotation.transform_vector(mn.Vector3(0, 0, -1))
            robot_forward.y = 0
            robot_forward = robot_forward.normalized()

            target_dir = direction.normalized()

            # Calculate signed angle between vectors (positive = turn right, negative = turn left)
            dot = robot_forward.x * target_dir.x + robot_forward.z * target_dir.z
            det = robot_forward.x * target_dir.z - robot_forward.z * target_dir.x
            angle = np.arctan2(det, dot)

            # Determine if we need to turn or go forward
            turn_threshold = 0.3  # radians, ~17 degrees

            if abs(angle) > turn_threshold:
                # Need to turn in place first
                if angle > 0:
                    # Turn right
                    self._apply_wheel_velocities(self.turn_speed, -self.turn_speed)
                    if verbose and steps % 30 == 0:
                        print(f"Turning right: {np.degrees(angle):.1f}°")
                else:
                    # Turn left
                    self._apply_wheel_velocities(-self.turn_speed, self.turn_speed)
                    if verbose and steps % 30 == 0:
                        print(f"Turning left: {np.degrees(angle):.1f}°")
            else:
                # Mostly aligned, drive forward with minor adjustment
                # Scale the wheel speeds for smoother turning while moving
                left_scale = 1.0 - angle * 0.5
                right_scale = 1.0 + angle * 0.5
                self._apply_wheel_velocities(
                    self.drive_speed * left_scale, 
                    self.drive_speed * right_scale
                )
                if verbose and steps % 30 == 0:
                    print(f"Moving forward: distance={direction.length():.2f}m")

            # Step physics
            for _ in range(3):  # Take multiple small physics steps
                self.sim.step_physics(self.dt)

            # Update agent position to match robot
            self.agent.scene_node.translation = self.locobot.translation
            self.agent.scene_node.rotation = self.locobot.rotation

            # Optional: slow down simulation for debugging/visualization
            if verbose and steps % 100 == 0:
                print(f"Navigation progress: {dist_to_goal:.2f}m to target")

        # If we got here, we hit max_steps without reaching the target
        if verbose:
            print(f"Navigation timeout after {max_steps} steps")
            print(f"Final distance to target: {(self.locobot.translation - goal_pos).length():.2f}m")

        # Stop the robot
        self._stop_motors()
        return False

    def _apply_wheel_velocities(self, left_velocity, right_velocity):
        """
        Apply velocities to the left and right wheel joints.

        Args:
            left_velocity: Velocity for left wheel
            right_velocity: Velocity for right wheel
        """
        if "wheel_left_joint" not in self.dof_map or "wheel_right_joint" not in self.dof_map:
            return

        # Set wheel velocities
        left_id = self.dof_map["wheel_left_joint"]
        right_id = self.dof_map["wheel_right_joint"]

        # Update motor settings
        self.motor_settings[left_id].velocity_target = left_velocity
        self.motor_settings[right_id].velocity_target = right_velocity

        # Apply settings to motors
        self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
        self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

        # Add to rl_navigator.py

    # Direct wheel test with minimal approach based on tutorials
    def minimal_wheel_test(self):
        """Bare minimum wheel test based on official examples"""
        print("\n=== MINIMAL WHEEL TEST ===")
        print(f"Starting position: {self.locobot.translation}")

        # Get wheel joints
        if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
            left_id = self.dof_map["wheel_left_joint"]
            right_id = self.dof_map["wheel_right_joint"]
        else:
            print("ERROR: Wheel joint IDs not found!")
            return

        # Use built-in velocity control of robot
        vel_control = self.locobot.velocity_control

        # Configure velocity control properly
        vel_control.controlling_lin_vel = True
        vel_control.lin_vel_is_local = True
        vel_control.controlling_ang_vel = False
        vel_control.linear_velocity = mn.Vector3(0.0, 0.0, -0.3)  # Forward in local space

        # Log setup
        print(f"Using robot's built-in velocity_control")
        print(f"Velocity set to: {vel_control.linear_velocity}")

        # Step physics in small increments
        for i in range(60):  # 2 seconds at 30Hz
            # Step physics
            self.sim.step_physics(self.dt)

            # Log position every 10 steps
            if i % 10 == 0:
                pos = self.locobot.translation
                print(f"Step {i}: Position={pos}")

                # Update agent to match robot
                self.agent.scene_node.translation = pos
                self.agent.scene_node.rotation = self.locobot.rotation

        # Final position
        print(f"Final position: {self.locobot.translation}")

        # Stop movement
        vel_control.linear_velocity = mn.Vector3(0.0, 0.0, 0.0)

        return True

    def test_wheel_motors(self, duration=5.0, verbose=True):
        """Simple test to verify wheel motors work"""
        if verbose:
            print("TESTING WHEEL MOTORS - Robot should move forward")
            print(f"Initial position: {self.locobot.translation}")

        # Set constant wheel velocities
        left_vel = 0.5  # Moderate speed
        right_vel = 0.5

        # Clear any previous velocities
        for lid in self.motor_settings:
            self.motor_settings[lid].velocity_target = 0.0

        # Print wheel joint IDs for verification
        if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
            left_id = self.dof_map["wheel_left_joint"]
            right_id = self.dof_map["wheel_right_joint"]
            print(f"Left wheel joint ID: {left_id}, Right wheel joint ID: {right_id}")
        else:
            print("ERROR: Wheel joint IDs not found in dof_map!")
            return False

        # Set wheel velocities
        self.motor_settings[left_id].velocity_target = left_vel
        self.motor_settings[right_id].velocity_target = right_vel

        # Print motor settings before applying
        print(f"Setting left wheel velocity: {left_vel}, right wheel velocity: {right_vel}")
        print(f"Left motor max_impulse: {self.motor_settings[left_id].max_impulse}")
        print(f"Right motor max_impulse: {self.motor_settings[right_id].max_impulse}")

        # Apply settings to motors
        self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
        self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

        # Step physics in small increments
        steps = int(duration / self.dt)
        for i in range(steps):
            # Step physics
            self.sim.step_physics(self.dt)

            # Print debug info every 10 steps
            if i % 10 == 0:
                pos = self.locobot.translation
                vel = self.locobot.root_linear_velocity
                print(f"Step {i}: Position={pos}, Velocity={vel}")

                # Update agent position to match robot
                self.agent.scene_node.translation = pos
                self.agent.scene_node.rotation = self.locobot.rotation

        # Stop the robot
        self.motor_settings[left_id].velocity_target = 0.0
        self.motor_settings[right_id].velocity_target = 0.0
        self.locobot.update_joint_motor(self.motor_ids[left_id], self.motor_settings[left_id])
        self.locobot.update_joint_motor(self.motor_ids[right_id], self.motor_settings[right_id])

        # Final position report
        final_pos = self.locobot.translation
        print(f"Final position: {final_pos}")
        print(f"Movement delta: {(final_pos - self.locobot.translation).length()}m")

        return True

    def physics_explore(self):
        """Execute physics-based exploration to a random point."""
        # Get a random point
        try:
            target_point = self.pathfinder.get_random_navigable_point()
            print(f"Navigating to: {target_point}")
            
            # Use the fixed navigation method
            success = self.fixed_navigate_to_point(target_point, verbose=True)
            
            if success:
                print("Navigation succeeded!")
            else:
                print("Navigation failed")
        except Exception as e:
            print(f"Error during exploration: {e}")
        
        return True

    def _check_progress(self):
        """Check if robot is making progress or stuck."""
        # Implementation would track position history and detect if stuck
        pass
    
    def _pause_for_stability(self, frames=2):
        """Pause for a few frames to let physics stabilize."""
        for _ in range(frames):
            self.sim.step_physics(self.dt)
            
    def _apply_action_to_motors(self, action):
        """Apply a navigation action to the robot's motors with improved velocity control."""
        # Reset all velocities first
        for lid in self.motor_settings:
            self.motor_settings[lid].velocity_target = 0.0
        
        # Apply appropriate motor commands based on action
        if action == "move_forward":
            if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
                # Use a ramped approach - start slower and increase
                speed = self.drive_speed * 0.8  # Start at 80% of max speed
                self.motor_settings[self.dof_map["wheel_left_joint"]].velocity_target = speed
                self.motor_settings[self.dof_map["wheel_right_joint"]].velocity_target = speed
        elif action == "turn_left":
            if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
                # Slower and more controlled turns
                speed = self.turn_speed * 0.7
                self.motor_settings[self.dof_map["wheel_left_joint"]].velocity_target = -speed
                self.motor_settings[self.dof_map["wheel_right_joint"]].velocity_target = speed
        elif action == "turn_right":
            if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
                # Slower and more controlled turns
                speed = self.turn_speed * 0.7
                self.motor_settings[self.dof_map["wheel_left_joint"]].velocity_target = speed
                self.motor_settings[self.dof_map["wheel_right_joint"]].velocity_target = -speed
                
        # Update all motors
        for lid, mid in self.motor_ids.items():
            self.locobot.update_joint_motor(mid, self.motor_settings[lid])
    
    def _stop_motors(self):
        """Stop all motors smoothly."""
        # First reduce speed gradually
        for reduction in [0.5, 0.2, 0.0]:
            # Get current speeds and reduce
            for lid in self.motor_settings:
                current_speed = self.motor_settings[lid].velocity_target
                self.motor_settings[lid].velocity_target = current_speed * reduction
                
            # Apply reduced speeds
            for lid, mid in self.motor_ids.items():
                self.locobot.update_joint_motor(mid, self.motor_settings[lid])
                
            # Small physics step to apply changes
            self.sim.step_physics(self.dt)
            
        # Final zero for all motors
        for lid in self.motor_settings:
            self.motor_settings[lid].velocity_target = 0.0
            
        for lid, mid in self.motor_ids.items():
            self.locobot.update_joint_motor(mid, self.motor_settings[lid])
    
    def navigate_to_object(self, object_data, verbose=True):
        """Navigate to an object based on perception data."""
        if not object_data or "bbox" not in object_data:
            if verbose:
                print("Invalid object data for navigation")
            return False
        
        try:
            # Get object position from bounding box
            bbox = object_data["bbox"]
            
            # If we have world coordinates from perception
            if "world_position" in object_data:
                target_pos = object_data["world_position"]
                return self.navigate_to_point(target_pos, verbose)
            
            # Otherwise try to use the robot's depth camera and ray casting
            # This would be implemented by getting depth from the camera
            # and projecting a ray through the center of the bbox
            
            if verbose:
                print("Object position not available. Need world coordinates.")
            return False
        except Exception as e:
            if verbose:
                print(f"Error navigating to object: {e}")
            return False