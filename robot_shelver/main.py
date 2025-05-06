#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import time
import json
import magnum as mn
import math
from environment import setup_simulator, run_simulator_step, is_camera_movement_command, visualize_gripper_state
from perception import Perception
from controller import Controller, ControlMode

def ensure_same_channels(images):
    """Ensure all images have 3 channels (convert grayscale or RGBA to BGR)."""
    processed_images = []
    for img in images:
        if len(img.shape) == 2:  # Grayscale image
            processed_images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        elif img.shape[2] == 4:  # RGBA image
            processed_images.append(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
        else:  # Already 3-channel
            processed_images.append(img)
    return processed_images

def main():
    # Initialize simulator, agent, and robot
    sim, agent, locobot, motor_ids, motor_settings, dof_map, camera_controller = setup_simulator()

    # Debug: print initial state
    print(f"DEBUG: Initial robot position: {locobot.translation}")
    print(f"DEBUG: Initial robot rotation: {locobot.rotation}")

    # Ensure all joint motors start with zero velocity to prevent initial jitter
    for lid, mid in motor_ids.items():
        joint_name = locobot.get_link_joint_name(lid)
        motor_settings[lid].velocity_target = 0.0
        locobot.update_joint_motor(mid, motor_settings[lid])
        print(f"DEBUG: Initial velocity for joint '{joint_name}' set to 0.0")

    # Take a small physics step to let the robot settle
    sim.step_physics(0.01)

    # Initialize perception module (Visual Language Model, etc.)
    perception = Perception(model_name="llava-phi3")
    perception.set_camera_params(sim, "robot_rgb")  # Use front RGB camera

    # Define motion speed constants
    DRIVE_SPEED = 1.2
    TURN_SPEED  = 0.5
    ARM_SPEED   = 0.7
    GRIP_SPEED  = 0.7
    dt = 1.0 / 30.0  # simulation timestep (30 FPS)

    # Initialize controller (for LLM/VLM-based autonomous control)
    controller = Controller(llm_model="qwen2.5:7b")
    # Provide controller with access to robot control interfaces and speed parameters
    controller.set_robot_controls(
        locobot, motor_ids, motor_settings, dof_map, sim=sim,
        drive_speed=DRIVE_SPEED, turn_speed=TURN_SPEED,
        arm_speed=ARM_SPEED, grip_speed=GRIP_SPEED
    )
    controller.start()  # Start controller thread (for asynchronous commands)

    # Create an OpenCV window for displaying the robot's view
    cv2.namedWindow("Robot View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Robot View", 1280, 960)

    # Print control instructions for the user
    print("\n--- Control Guide ---")
    print("Navigation:")
    print("  Arrow keys = Drive robot forward/backward")
    print("  Q/W = Turn left/right")
    print("\nArm Control:")
    print("  U/J = Waist joint up/down")
    print("  I/K = Shoulder joint up/down")
    print("  O/L = Elbow joint up/down")
    print("  P/; = Forearm roll (rotate wrist base)")
    print("  [/] = Wrist angle up/down")
    print("  N/M = Wrist rotate left/right")
    print("\nGripper:")
    print("  G/H = Open/close gripper")
    print("\nPerception Commands:")
    print("  V = Describe scene (Vision-Language Model)")
    print("  B = Detect objects (bounding boxes)")
    print("  S = Analyze scene structure")
    print("\nAutomated Control:")
    print("  T = Target and navigate to an object")
    print("  Y = Grasp the targeted object")
    print("  E = Explore environment autonomously")
    print("  Z = Return to manual control")
    print("  ESC = Quit\n")

    # Initialize perception state
    last_description = ""         # Last VLM description text
    objects_detected = []         # List of objects detected in current view
    show_object_detections = False

    # Initialize control state
    control_mode = "manual"       # Current control mode (manual or various auto modes)
    target_object = None          # Currently targeted object name (if any)

    # Main control loop
    while True:
        key = cv2.waitKeyEx(1)  # capture keyboard input (non-blocking, 1ms wait)
        if key == 27:  # ESC key pressed
            break

        if key is not None and is_camera_movement_command(key):
        # Block camera movement in manual mode unless controller specifically requested it
            if not (controller.mode != ControlMode.MANUAL or key in controller.command_queue.queue):
                print("Camera movement blocked - only LLM can control camera")
                key = 0  # Nullify the key to prevent processing

        # Check controller (LLM) for any commands and handle them
        controller_command = None
        try:
            if not controller.command_queue.empty():
                controller_command = controller.command_queue.get_nowait()
                # Case 1: Direct motor control command from the LLM (dictionary of action)
                if isinstance(controller_command, dict):
                    action_type = controller_command.get("type")
                    # Reset all motor velocity targets first to avoid unintended motion carry-over
                    for s in motor_settings.values():
                        s.velocity_target = 0.0
                    # Apply the LLM command to motors
                    if action_type == "FORWARD":
                        # Move forward (both wheels positive speed)
                        speed = controller_command.get("value", DRIVE_SPEED)
                        motor_settings[dof_map["wheel_left_joint"]].velocity_target  = speed
                        motor_settings[dof_map["wheel_right_joint"]].velocity_target = speed
                        print(f"Direct control: FORWARD (speed: {speed:.2f})")
                    elif action_type == "BACKWARD":
                        # Move backward (both wheels negative speed)
                        speed = controller_command.get("value", DRIVE_SPEED)
                        motor_settings[dof_map["wheel_left_joint"]].velocity_target  = -speed
                        motor_settings[dof_map["wheel_right_joint"]].velocity_target = -speed
                        print(f"Direct control: BACKWARD (speed: {speed:.2f})")
                    elif action_type == "TURN_LEFT":
                        # Turn left (left wheel backward, right wheel forward)
                        speed = controller_command.get("value", TURN_SPEED)
                        motor_settings[dof_map["wheel_left_joint"]].velocity_target  = -speed
                        motor_settings[dof_map["wheel_right_joint"]].velocity_target =  speed
                        print(f"Direct control: TURN LEFT (speed: {speed:.2f})")
                    elif action_type == "TURN_RIGHT":
                        # Turn right (left wheel forward, right wheel backward)
                        speed = controller_command.get("value", TURN_SPEED)
                        motor_settings[dof_map["wheel_left_joint"]].velocity_target  =  speed
                        motor_settings[dof_map["wheel_right_joint"]].velocity_target = -speed
                        print(f"Direct control: TURN RIGHT (speed: {speed:.2f})")
                    elif action_type == "ARM_JOINT":
                        # Move a specific arm joint at given velocity value
                        joint_name = controller_command.get("joint")
                        value = controller_command.get("value", 0.0)
                        if joint_name in dof_map:
                            motor_settings[dof_map[joint_name]].velocity_target = value
                            print(f"Direct control: ARM_JOINT '{joint_name}' = {value}")
                    elif action_type == "GRIPPER":
                        # Open/close gripper (positive value -> open, negative -> close)
                        value = controller_command.get("value", 0.0)
                        direction = 1 if value > 0 else -1
                        motor_settings[dof_map["left_finger"]].velocity_target = direction * GRIP_SPEED
                        print(f"Direct control: GRIPPER {'OPEN' if value > 0 else 'CLOSE'}")
                    # Update all joint motors with the new targets
                    for lid, mid in motor_ids.items():
                        locobot.update_joint_motor(mid, motor_settings[lid])
                    # Step physics a few times to enact the direct motor command immediately
                    for _ in range(3):
                        sim.step_physics(dt)
                    # Print updated robot position for debugging
                    print(f"Robot position: {locobot.translation}")
                    # Set key to None to prevent also processing it as a manual key
                    key = None
                elif controller_command == "EXPLORE":  # Add this specific check
                    # Use collision-aware exploration from environment module
                    import environment
                    environment.explore_with_collision_avoidance(sim, locobot, 0.2)
                    key = None  # Prevent double processing
                elif controller_command is not None:
                    # Case 2: The controller provided a key press
                    key = controller_command
                    print(f"Executing controller-generated key command: {key}")
        except Exception as e:
            print(f"Error processing controller command: {e}")

        # Step the simulation either with a manual key command or idle (no command)
        if key is not None:
            # **Manual or controller key input path**:
            # Reset all velocity targets to 0 before applying new key action to avoid residual motion
            for s in motor_settings.values():
                s.velocity_target = 0.0
            # Run one simulation step with the given key command
            obs, combined = run_simulator_step(
                sim, agent, locobot,
                motor_ids, motor_settings, dof_map,
                key, DRIVE_SPEED, TURN_SPEED,
                ARM_SPEED, GRIP_SPEED, dt, camera_controller
            )
        else:
            # **Idle or direct motor command path** (no discrete key input):
            # If we are truly idle (no controller dict command in this frame),
            # ensure the base wheels are fully stopped to avoid drifting
            if not isinstance(controller_command, dict):
                # Explicitly set wheel joint targets to zero
                if "wheel_left_joint" in dof_map and "wheel_right_joint" in dof_map:
                    motor_settings[dof_map["wheel_left_joint"]].velocity_target  = 0.0
                    motor_settings[dof_map["wheel_right_joint"]].velocity_target = 0.0
                for lid, mid in motor_ids.items():
                    locobot.update_joint_motor(mid, motor_settings[lid])
            # Advance physics a few larger timesteps to let the robot come to rest (prevents shaking)
            for _ in range(3):
                sim.step_physics(dt)
            # Synchronize the agent's camera with the robot's updated state
            rs = locobot.rigid_state
            agent.scene_node.translation = rs.translation
            agent.scene_node.rotation    = rs.rotation
            # Capture sensor observations for this frame
            obs = sim.get_sensor_observations()
            top_img  = obs['top_rgb']
            rob_img  = obs['robot_rgb']    # front RGB camera image
            back_img = obs['back_rgb']
            # Label the views (for visualization purposes)
            cv2.putText(top_img,  'TOP',   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(rob_img,  'FRONT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(back_img, 'BACK',  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # Combine the three camera views side by side
            combined = np.hstack((top_img, rob_img, back_img))

        # Ensure the combined image is in BGR color (convert from RGBA if needed)
        if combined.shape[2] == 4:
            combined = cv2.cvtColor(combined, cv2.COLOR_RGBA2BGR)

        # Extract the front camera image for perception tasks (make a copy for safety)
        front_img = obs['robot_rgb'].copy()
        if front_img.shape[2] == 4:
            front_img = cv2.cvtColor(front_img, cv2.COLOR_RGBA2BGR)

        # Record current robot state (position and orientation) for context
        robot_position = locobot.translation
        robot_rotation = locobot.rotation
        # Update controller with the latest robot state (for path planning, etc.)
        controller.update_robot_state(robot_position, robot_rotation)

        # --- Perception (VLM/Vision) Commands ---
        if key == ord('v') or key == ord('V'):
            # Describe the scene using the Vision-Language Model
            print("\nProcessing front view with VLM...")
            description = perception.process_image(
                front_img, prompt="Describe in detail what you see in this image from the robot's perspective."
            )
            print(f"\nVLM Description: {description}\n")
            last_description = description  # store the description for display
        elif key == ord('b') or key == ord('B'):
            # Detect objects in the front view
            print("\nDetecting objects in view...")
            objects_detected = perception.detect_objects(front_img)
            show_object_detections = True  # trigger display of detection results
            print(f"Detected {len(objects_detected)} objects:")
            for obj in objects_detected:
                name = obj.get('name', 'Unknown')
                desc = obj.get('description', '')
                print(f" - {name}: {desc}")
            # Update controller with perception results (for potential autonomous actions)
            controller.update_perception(front_img, objects_detected)
        elif key == ord('s') or key == ord('S'):
            # Analyze overall scene structure/relationships
            print("\nAnalyzing scene structure...")
            scene_data = perception.analyze_scene(front_img)
            print(json.dumps(scene_data, indent=2))
            # Update controller with scene analysis data (if needed for decision making)
            controller.update_perception(scene_analysis=scene_data)

        # --- Automated Control Commands (triggering LLM/VLM behaviors) ---
        elif key == ord('t') or key == ord('T'):
            # Target an object for navigation
            if objects_detected:
                # List detected objects for user selection
                print("\nSelect an object to target (enter its number):")
                for i, obj in enumerate(objects_detected):
                    print(f"  {i}: {obj.get('name', 'Unknown')}")
                target_idx = input("Object number: ")
                try:
                    target_idx = int(target_idx)
                    if 0 <= target_idx < len(objects_detected):
                        target_object = objects_detected[target_idx].get('name', 'Unknown')
                        print(f"Targeting object: {target_object}")
                        controller.navigate_to_object(target_object)  # instruct controller to navigate
                    else:
                        print("Invalid selection.")
                except ValueError:
                    print("Invalid input (please enter a number).")
            else:
                print("No objects detected to target.")
        elif key == ord('y') or key == ord('Y'):
            # Attempt to grasp the currently targeted object
            if target_object:
                print(f"Attempting to grasp: {target_object}")
                controller.grasp_object(target_object)
            else:
                print("No target object selected to grasp.")
        elif key == ord('e') or key == ord('E'):
            # Start autonomous exploration of the environment
            print("\n=== Starting autonomous exploration ===")
            result = controller.start_exploration()
            print(f"Exploration started: {result}")
            # Immediately plan and perform the first exploration action (for demo purposes)
            next_action = controller._plan_exploration_action()
            if next_action:
                print(f"Initial exploration action: {next_action}")
                if isinstance(next_action, dict):
                    # If an action dict is returned (e.g., move forward or turn), apply it directly to motors
                    action_type = next_action.get("type")
                    if action_type == "FORWARD":
                        motor_settings[dof_map["wheel_left_joint"]].velocity_target  = DRIVE_SPEED
                        motor_settings[dof_map["wheel_right_joint"]].velocity_target = DRIVE_SPEED
                        print(f"Applied initial FORWARD command (speed: {DRIVE_SPEED})")
                    elif action_type == "TURN_LEFT":
                        motor_settings[dof_map["wheel_left_joint"]].velocity_target  = -TURN_SPEED
                        motor_settings[dof_map["wheel_right_joint"]].velocity_target =  TURN_SPEED
                        print(f"Applied initial TURN_LEFT command (speed: {TURN_SPEED})")
                    # Update motors immediately for the exploration action
                    for lid, mid in motor_ids.items():
                        locobot.update_joint_motor(mid, motor_settings[lid])
        elif key == ord('z') or key == ord('Z'):
            # Return to manual control mode, stopping any autonomous behavior
            print("Switching back to manual control.")
            controller.stop_autonomous_control()

        # --- Build the display view with overlays ---
        # 1. Draw robot coordinates (position & orientation) on the combined view
        display_view = perception.draw_robot_coordinates(combined, robot_position, robot_rotation)

        display_view = visualize_gripper_state(display_view, locobot, dof_map)
        # 2. If object detections are to be shown, create a side-by-side detection panel
        if show_object_detections:
            # Copy the front image and draw detections (if any)
            detection_img = front_img.copy()
            if objects_detected:
                detection_img = perception.visualize_detections(detection_img, objects_detected)
            # Create a header for the detection panel
            header = np.zeros((40, detection_img.shape[1], 3), dtype=np.uint8)
            status_text = f"Object Detection: {len(objects_detected)} object(s)"
            cv2.putText(header, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            # Ensure header and detection_img both have 3 channels
            header, detection_img = ensure_same_channels([header, detection_img])
            # Stack header above the detection image
            detection_view = np.vstack([header, detection_img])
            # Resize detection view to match the main display width
            target_width = display_view.shape[1]
            scale = target_width / detection_view.shape[1]
            target_height = int(detection_view.shape[0] * scale)
            detection_view_resized = cv2.resize(detection_view, (target_width, target_height))
            # Append the detection panel below the main view
            display_view = np.vstack([display_view, detection_view_resized])
        # 3. If a description from the VLM is available, overlay it at the bottom
        if last_description:
            desc_height = 40
            desc_bg = np.zeros((desc_height, display_view.shape[1], 3), dtype=np.uint8)
            # Truncate the description text if it's too long for one line
            short_desc = (last_description[:80] + "...") if len(last_description) > 80 else last_description
            cv2.putText(desc_bg, f"VLM: {short_desc}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            # Stack the description background at the very bottom of the display
            display_view = np.vstack([display_view, desc_bg])
        # 4. Add controller status (mode, target, etc.) at the bottom as well
        status_height = 40
        status_bg = np.zeros((status_height, display_view.shape[1], 3), dtype=np.uint8)
        controller_status = controller.get_status()  # retrieve current mode/target/progress
        status_text = f"Mode: {controller_status['mode']}"
        if controller_status.get('target_object'):
            status_text += f" | Target: {controller_status['target_object']}"
        if controller_status.get('task_description'):
            # Show a truncated task description if present
            task_desc = controller_status['task_description'][:30]
            status_text += f" | Task: {task_desc}"
        if controller_status.get('progress') and controller_status['progress'] != "N/A":
            status_text += f" | Progress: {controller_status['progress']}"
        cv2.putText(status_bg, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 1)
        display_view = np.vstack([display_view, status_bg])

        # Finally, show the assembled display in the OpenCV window
        cv2.imshow("Robot View", display_view)
    # End of main loop

    # Cleanup: stop the controller thread and close the simulator
    controller.stop()
    cv2.destroyAllWindows()
    sim.close()

if __name__ == '__main__':
    main()