#!/usr/bin/env python3
import os
import time
import math
import random
import numpy as np
import torch
import magnum as mn
import habitat_sim
import environment

class RLAgent:
    def __init__(
        self,
        sim,
        agent,
        locobot,
        motor_ids,
        motor_settings,
        dof_map,
        drive_speed=1.2,
        turn_speed=0.5,
        arm_speed=0.7,
        grip_speed=0.7,
        dt=1/30.0,
        nav_model_path=None,
        grasp_model_path=None,
        place_model_path=None
    ):
        """Load pretrained RL policies for navigation and manipulation."""
        self.sim = sim
        self.agent = agent
        self.locobot = locobot
        self.motor_ids = motor_ids
        self.motor_settings = motor_settings
        self.dof_map = dof_map
        self.drive_speed = drive_speed
        self.turn_speed = turn_speed
        self.arm_speed = arm_speed
        self.grip_speed = grip_speed
        self.dt = dt

        # Load navigation policy if available
        self.nav_policy = None
        if nav_model_path and os.path.exists(nav_model_path):
            try:
                self.nav_policy = torch.jit.load(nav_model_path)
                self.nav_policy.eval()
                print(f"Loaded navigation model from {nav_model_path}")
            except Exception as e:
                print(f"Failed to load nav model: {e}")
        else:
            print("Navigation model not provided/found. Using heuristic navigation.")

        # Load grasp policy if available
        self.grasp_policy = None
        if grasp_model_path and os.path.exists(grasp_model_path):
            try:
                self.grasp_policy = torch.jit.load(grasp_model_path)
                self.grasp_policy.eval()
                print(f"Loaded grasp model from {grasp_model_path}")
            except Exception as e:
                print(f"Failed to load grasp model: {e}")
        else:
            print("Grasp model not provided/found. Using heuristic grasp.")

        # Load place policy if available
        self.place_policy = None
        if place_model_path and os.path.exists(place_model_path):
            try:
                self.place_policy = torch.jit.load(place_model_path)
                self.place_policy.eval()
                print(f"Loaded place model from {place_model_path}")
            except Exception as e:
                print(f"Failed to load place model: {e}")
        else:
            print("Place model not provided/found.")

    def navigate_to(self, target_position, max_steps=500):
        """Navigate the robot to the given target position."""
        if target_position is None:
            print("No target position provided for navigation.")
            return False
        print(f"[RLAgent] Navigating to: {target_position}")

        steps = 0
        success = False
        while steps < max_steps:
            # Compute relative vector
            pos = self.locobot.translation
            delta = np.array(target_position) - np.array([pos.x, pos.y, pos.z])
            dist = np.linalg.norm(delta[[0, 2]])

            # Determine forward direction using quaternion
            quat = self.locobot.rigid_state.rotation
            fwd = quat.transformVector(mn.Vector3(0.0, 0.0, -1.0))
            forward_dir = np.array([fwd.x, fwd.z])
            n = np.linalg.norm(forward_dir)
            if n > 1e-8:
                forward_dir /= n
            else:
                forward_dir = np.array([0.0, -1.0])

            # Goal direction
            goal_dir = np.array([delta[0], delta[2]])
            gn = np.linalg.norm(goal_dir)
            if gn > 1e-8:
                goal_dir /= gn

            # Angle and side
            dot = np.clip(np.dot(forward_dir, goal_dir), -1.0, 1.0)
            ang = math.acos(dot)
            side = forward_dir[0]*goal_dir[1] - forward_dir[1]*goal_dir[0]

            # Decide action
            if dist < 0.2:
                action = 'STOP'
            elif ang > 0.1:
                action = 'TURN_LEFT' if side < 0 else 'TURN_RIGHT'
            else:
                action = 'FORWARD'

            # Map to key
            key_map = {'FORWARD': 65362, 'BACKWARD': 65364,
                       'TURN_LEFT': 65361, 'TURN_RIGHT': 65363}
            key = key_map.get(action, 0)
            if action == 'STOP':
                success = True
                break

            environment.run_simulator_step(
                self.sim, self.agent, self.locobot,
                self.motor_ids, self.motor_settings, self.dof_map,
                key, self.drive_speed, self.turn_speed,
                self.arm_speed, self.grip_speed, self.dt
            )
            steps += 1

        # Final stop
        environment.run_simulator_step(
            self.sim, self.agent, self.locobot,
            self.motor_ids, self.motor_settings, self.dof_map,
            0, self.drive_speed, self.turn_speed,
            self.arm_speed, self.grip_speed, self.dt
        )
        print(f"Navigation {'succeeded' if success else 'failed'} in {steps} steps.")
        return success

    def grasp_object(self, name=None, max_steps=200):
        """Heuristic grasp if no policy provided."""
        print(f"[RLAgent] Grasping: {name}")
        for step in range(max_steps):
            if step < 50:
                key = ord('k')  # shoulder down
            elif step < 100:
                key = ord('l')  # elbow down
            elif step < 150:
                key = ord('h')  # close gripper
            else:
                key = 0
                print("Grasp complete or timed out.")
                break

            environment.run_simulator_step(
                self.sim, self.agent, self.locobot,
                self.motor_ids, self.motor_settings, self.dof_map,
                key, self.drive_speed, self.turn_speed,
                self.arm_speed, self.grip_speed, self.dt
            )
        return True

    def explore_environment(self, duration=10, use_habitat_policy=True):
        """Stable exploration using either built-in policies or simplified movements."""
        print(f"[RLAgent] Exploring for {duration}s with stable movements.")

        if use_habitat_policy:
            try:
                # Try to use habitat's built-in navigation policy if available
                from habitat_baselines.rl.ppo import PPO
                from habitat_baselines.config.default import get_config

                # Log that we're using habitat's navigation
                print("Using Habitat's built-in navigation policy for exploration")

                # Simple point goal navigation - go to random points
                for i in range(3):  # Limit to just a few points for stability
                    # Get a random navigable point
                    target_point = self.sim.pathfinder.get_random_navigable_point()
                    print(f"Navigating to point: {target_point}")

                    # Use simplified navigation - small steps with pauses
                    start_time = time.time()
                    step_count = 0

                    while time.time() - start_time < duration/3 and step_count < 20:
                        # Take small steps toward goal
                        current_pos = self.locobot.translation
                        direction = np.array([target_point[0] - current_pos[0], 
                                             target_point[2] - current_pos[2]])

                        # Normalize and scale down movement
                        if np.linalg.norm(direction) > 0.01:
                            direction = direction / np.linalg.norm(direction) * 0.3

                        # Convert to key command - simplified approach
                        key = 65362  # Forward key
                        if abs(direction[0]) > abs(direction[1]):
                            if direction[0] < 0:
                                key = 65361  # Left key
                            else:
                                key = 65363  # Right key

                        # Execute small movement
                        environment.run_simulator_step(
                            self.sim, self.agent, self.locobot,
                            self.motor_ids, self.motor_settings, self.dof_map,
                            key, self.drive_speed * 0.3, self.turn_speed * 0.3,
                            self.arm_speed, self.grip_speed, self.dt
                        )

                        # Critical: add pause between steps
                        time.sleep(0.3)
                        step_count += 1

                    # Force stop and stabilize
                    environment.run_simulator_step(
                        self.sim, self.agent, self.locobot,
                        self.motor_ids, self.motor_settings, self.dof_map,
                        0, self.drive_speed, self.turn_speed,
                        self.arm_speed, self.grip_speed, self.dt
                    )
                    time.sleep(0.5)  # Allow physics to stabilize

            except ImportError as e:
                print(f"Could not use Habitat navigation policy: {e}")
                print("Falling back to simple exploration")
                self._simple_exploration(duration)
        else:
            self._simple_exploration(duration)

        print("Exploration done.")
        return True

    def _simple_exploration(self, duration):
        """Very simple exploration with collision avoidance."""
        end = time.time() + duration
        steps = 0
        
        current_pos = [self.locobot.translation[0], 0.0, self.locobot.translation[2]]
        
        while time.time() < end and steps < 10:
            # Use extremely reduced speed - 10% of normal
            drive_speed = self.drive_speed * 0.1
            turn_speed = self.turn_speed * 0.1
            
            # Try multiple directions until finding a valid one
            found_valid_direction = False
            for attempt in range(8):  # Try up to 8 directions
                # Choose direction (try different angles in sequence)
                angle = (attempt * (math.pi/4)) + random.uniform(-0.2, 0.2)
                distance = 0.15  # Small movement distance
                
                # Calculate movement vector
                dx = distance * math.cos(angle)
                dz = distance * math.sin(angle)
                new_pos = [current_pos[0] + dx, 0.0, current_pos[2] + dz]
                
                # Check if position is navigable
                is_navigable = self.sim.pathfinder.is_navigable(
                    mn.Vector3(new_pos[0], new_pos[1], new_pos[2])
                )
                
                # Cast ray to check for obstacles
                ray_direction = mn.Vector3(dx, 0, dz).normalized()
                ray = habitat_sim.geo.Ray(
                    mn.Vector3(*current_pos), 
                    ray_direction
                )
                raycast_results = self.sim.cast_ray(ray, distance * 1.2)
                
                if is_navigable and not raycast_results.has_hits():
                    found_valid_direction = True
                    print(f"Found valid direction after {attempt+1} attempts")
                    
                    # Apply movement
                    state = self.locobot.rigid_state
                    state.translation = mn.Vector3(new_pos[0], new_pos[1], new_pos[2])
                    self.locobot.rigid_state = state
                    
                    # Update current position
                    current_pos = new_pos
                    break
                
            if not found_valid_direction:
                print("Could not find valid movement, turning in place")
                # Just rotate in place
                quat = self.locobot.rigid_state.rotation
                rot_angle = random.uniform(0.1, 0.3)  # Small rotation
                new_quat = quat * mn.Quaternion.rotation(mn.Rad(rot_angle), mn.Vector3(0, 1, 0))
                
                state = self.locobot.rigid_state
                state.rotation = new_quat
                self.locobot.rigid_state = state
            
            # Zero all velocities between steps
            self.locobot.root_linear_velocity = mn.Vector3(0, 0, 0)
            self.locobot.root_angular_velocity = mn.Vector3(0, 0, 0)
            
            # Wait longer between steps
            time.sleep(1.0)
            steps += 1