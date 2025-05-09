#!/usr/bin/env python3
import os
import time
import numpy as np
import magnum as mn
import habitat_sim
from habitat_sim.nav import GreedyGeodesicFollower, ShortestPath

class RLAgent:
    """
    Class for reinforcement learning agent navigation using Habitat's PathFinder.
    This implements the approach shown in the rigid object tutorial:
    https://aihabitat.org/docs/habitat-sim/rigid-object-tutorial.html#continuous-control-on-navmesh
    """
    
    def __init__(self, sim, agent, locobot, motor_ids, motor_settings, dof_map, 
                 drive_speed=1.0, turn_speed=0.5, dt=1/30.0):
        """Initialize the RL Agent."""
        self.sim = sim
        self.agent = agent
        self.locobot = locobot
        self.motor_ids = motor_ids
        self.motor_settings = motor_settings
        self.dof_map = dof_map
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed
        self.dt = dt
        
        # Initialize path follower
        self.pathfinder = sim.pathfinder
        self.path_follower = None
        
        if self.pathfinder is not None and self.pathfinder.is_loaded:
            print("Initializing GreedyGeodesicFollower for navigation...")
            try:
                # Create the GreedyGeodesicFollower using the agent's action space
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
                self.path_follower = None
        else:
            print("WARNING: NavMesh not loaded. Path follower cannot be initialized.")
            
    def navigate_to_point(self, target_position, verbose=True):
        """
        Navigate to a target position using the GreedyGeodesicFollower.
        
        Args:
            target_position: 3D target position to navigate to
            verbose: Whether to print navigation progress
            
        Returns:
            bool: Success or failure of navigation
        """
        if self.path_follower is None:
            print("ERROR: Path follower not initialized. Cannot navigate.")
            return False
            
        if verbose:
            print(f"Navigating to position: {target_position}")
            
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
        
        # Check if path exists
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
            
        try:
            # Get the sequence of actions to follow the path
            action_list = self.path_follower.find_path(goal_pos)
            
            if len(action_list) == 0:
                if verbose:
                    print("Path follower returned empty action list")
                return False
                
            if verbose:
                print(f"Generated {len(action_list)} actions to follow path")
                
            # Execute each action in the sequence
            for i, action in enumerate(action_list):
                if verbose and i % 5 == 0:  # Print progress every 5 actions
                    print(f"Executing action {i+1}/{len(action_list)}: {action}")
                    
                # Convert action to motor commands
                self._apply_action_to_motors(action)
                
                # Step physics
                for _ in range(3):  # Take multiple physics steps per action
                    self.sim.step_physics(self.dt)
                    
                # Update agent position to match robot
                self.agent.scene_node.translation = self.locobot.translation
                self.agent.scene_node.rotation = self.locobot.rotation
                
                # Check if we've reached the goal
                current_pos = self.locobot.translation
                dist = (current_pos - goal_pos).length()
                
                if dist < 0.5:  # Within 0.5m of target
                    if verbose:
                        print(f"Reached target (distance: {dist:.2f}m)")
                    return True
                    
            # Stop the robot
            self._stop_motors()
            
            # Final distance check
            current_pos = self.locobot.translation
            dist = (current_pos - goal_pos).length()
            success = dist < 0.5
            
            if verbose:
                print(f"Navigation {'succeeded' if success else 'failed'}")
                print(f"Final distance to target: {dist:.2f}m")
                
            return success
            
        except habitat_sim.errors.GreedyFollowerError as e:
            if verbose:
                print(f"GreedyFollower error: {e}")
            return False
        except Exception as e:
            if verbose:
                print(f"Error during navigation: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    def explore_environment(self, num_points=3, max_distance=3.0, verbose=True):
        """
        Explore the environment by navigating to random navigable points.
        
        Args:
            num_points: Number of random points to visit
            max_distance: Maximum distance for random points
            verbose: Whether to print progress
            
        Returns:
            bool: Success or failure of exploration
        """
        if self.pathfinder is None or not self.pathfinder.is_loaded:
            if verbose:
                print("Pathfinder not available. Cannot explore.")
            return False
            
        if verbose:
            print(f"Starting exploration with {num_points} random points")
            
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
            time.sleep(0.5)
            
        if verbose:
            print(f"\nExploration completed. Visited {points_visited}/{num_points} points")
            
        return points_visited > 0
    
    def _apply_action_to_motors(self, action):
        """
        Apply a navigation action to the robot's motors.
        
        Args:
            action: Action name (move_forward, turn_left, turn_right)
        """
        # Clear previous motor velocities
        for lid in self.motor_settings:
            self.motor_settings[lid].velocity_target = 0.0
        
        # Apply appropriate motor commands based on action
        if action == "move_forward":
            if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
                self.motor_settings[self.dof_map["wheel_left_joint"]].velocity_target = self.drive_speed
                self.motor_settings[self.dof_map["wheel_right_joint"]].velocity_target = self.drive_speed
        elif action == "turn_left":
            if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
                self.motor_settings[self.dof_map["wheel_left_joint"]].velocity_target = -self.turn_speed
                self.motor_settings[self.dof_map["wheel_right_joint"]].velocity_target = self.turn_speed
        elif action == "turn_right":
            if "wheel_left_joint" in self.dof_map and "wheel_right_joint" in self.dof_map:
                self.motor_settings[self.dof_map["wheel_left_joint"]].velocity_target = self.turn_speed
                self.motor_settings[self.dof_map["wheel_right_joint"]].velocity_target = -self.turn_speed
                
        # Update all motors
        for lid, mid in self.motor_ids.items():
            self.locobot.update_joint_motor(mid, self.motor_settings[lid])
    
    def _stop_motors(self):
        """Stop all motors."""
        for lid in self.motor_settings:
            self.motor_settings[lid].velocity_target = 0.0
            
        for lid, mid in self.motor_ids.items():
            self.locobot.update_joint_motor(mid, self.motor_settings[lid])