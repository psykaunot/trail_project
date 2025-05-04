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

    def explore_environment(self, duration=10):
        """Random exploration for a set duration."""
        print(f"[RLAgent] Exploring for {duration}s.")
        end = time.time() + duration
        while time.time() < end:
            action = random.choice(['FORWARD', 'TURN_LEFT', 'TURN_RIGHT'])
            key = {'FORWARD':65362,'TURN_LEFT':65361,'TURN_RIGHT':65363}[action]
            environment.run_simulator_step(
                self.sim, self.agent, self.locobot,
                self.motor_ids, self.motor_settings, self.dof_map,
                key, self.drive_speed, self.turn_speed,
                self.arm_speed, self.grip_speed, self.dt
            )
            time.sleep(0.1)
        # Stop at end
        environment.run_simulator_step(
            self.sim, self.agent, self.locobot,
            self.motor_ids, self.motor_settings, self.dof_map,
            0, self.drive_speed, self.turn_speed,
            self.arm_speed, self.grip_speed, self.dt
        )
        print("Exploration done.")
        return True
