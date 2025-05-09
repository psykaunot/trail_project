#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import time
import json
import magnum as mn
import math
import threading

from environment import setup_simulator, run_simulator_step, is_camera_movement_command, visualize_gripper_state
from controller import Controller, ControlMode
from perception.perception import Perception
from navigation.rl_navigator import RLNavigator
from navigation.navmesh_navigator import NavMeshNavigator
from manipulation.pick_place_demo import PickAndPlaceTask
from utils.object_spawner import ObjectSpawner
from utils.link_monitor import start_link_monitor
from utils.video_recorder import VideoRecorder

# Initialize video recorder
video_recorder = VideoRecorder(output_path="book_mission.mp4", fps=30)


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

    # After drawing the display view:
    if hasattr(video_recorder, 'is_recording') and video_recorder.is_recording:
        video_recorder.add_frame(display_view)

    #monitor_thread = start_link_monitor(locobot) #Use to print all therobot links' position and rotation vectors every 5 sec.
    object_spawner = ObjectSpawner(sim)
    book_object = object_spawner.spawn_book()
    print(f"Book object spawned: {book_object is not None}")
    if book_object is not None:
        print(f"Book position: {book_object.translation}")
        print(f"Book motion type: {book_object.motion_type}")

    # Ensure all joint motors start with zero velocity to prevent initial jitter
    for lid, mid in motor_ids.items():
        motor_settings[lid].velocity_target = 0.0
        locobot.update_joint_motor(mid, motor_settings[lid])

    # Take a small physics step to let the robot settle
    sim.step_physics(0.01)
    
    # Initialize perception module (Visual Language Model, etc.)
    perception = Perception(model_name="llava-phi")
    perception.set_camera_params(sim, "robot_rgb")  # Use front RGB camera

    # Define motion speed constants
    DRIVE_SPEED = 1.2
    TURN_SPEED  = 0.5
    ARM_SPEED   = 0.7
    GRIP_SPEED  = 0.7
    dt = 1.0 / 30.0  # simulation timestep (30 FPS)

    # Initialize controller (for LLM/VLM-based autonomous control)
    controller = Controller(llm_model="qwen3:8b")
    # Provide controller with access to robot control interfaces and speed parameters
    controller.set_robot_controls(
        locobot, motor_ids, motor_settings, dof_map, sim=sim,
        drive_speed=DRIVE_SPEED, turn_speed=TURN_SPEED,
        arm_speed=ARM_SPEED, grip_speed=GRIP_SPEED
    )
    controller.start()  # Start controller thread (for asynchronous commands)
    
    # Initialize RL navigator for path planning
    rl_navigator = RLNavigator(
        sim=sim,
        agent=agent,
        locobot=locobot, 
        motor_ids=motor_ids,
        motor_settings=motor_settings,
        dof_map=dof_map,
        drive_speed=DRIVE_SPEED,
        turn_speed=TURN_SPEED,
        dt=dt
    )

    navmesh_navigator = NavMeshNavigator(
    sim=sim,
    locobot=locobot,
    motor_ids=motor_ids,
    motor_settings=motor_settings,
    dof_map=dof_map,
    drive_speed=1.0,
    turn_speed=0.5,
    dt=dt
    )



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
            if key in [ord('r'), ord('R')]:
                if video_recorder.is_recording:
                    video_recorder.stop_recording()
                else:
                    video_recorder.start_recording()
            
            elif not (controller.mode != ControlMode.MANUAL or key in controller.command_queue.queue):
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
                    # Set key to None to prevent also processing it as a manual key
                    key = None
                elif controller_command == "EXPLORE":
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
            front_facing_img = obs['front_facing_rgb']

            # Label the views (for visualization purposes)
            cv2.putText(top_img,  'TOP',   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(rob_img,  'FRONT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(back_img, 'BACK',  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(front_facing_img, 'FRONT VIEW', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # 2x2 grid layout
            top_row = np.hstack((top_img, rob_img))
            bottom_row = np.hstack((back_img, front_facing_img))
            # Combine the three camera views side by side
            combined = np.vstack((top_row, bottom_row))

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
            print("🚀 Starting Book Finding Mission!")
            controller.find_book_mission()
            controller.update_perception(front_img, objects_detected)
        elif key == ord('s') or key == ord('S'):
            # Analyze overall scene structure/relationships
            print("\nAnalyzing scene structure...")
            scene_data = perception.analyze_scene(front_img)
            print(json.dumps(scene_data, indent=2))
            # Update controller with scene analysis data (if needed for decision making)
            controller.update_perception(scene_analysis=scene_data)

        # --- RL Navigation Commands ---
        elif key == ord('x') or key == ord('X'):
            # Test wheel movement
            print("\n=== TESTING MINIMAL WHEEL MOVEMENT ===")
            rl_navigator.direct_wheel_test()

        elif key == ord('p'):
            pick_place = PickAndPlaceTask(sim, locobot, motor_ids, motor_settings, dof_map, book_object)
            pick_place.execute_pick_and_place()

        elif key == ord('r') or key == ord('R'):
             controller.camera_guided_book_search()

        elif key == ord('t') or key == ord('T'):
            print("Testing camera movement...")
            camera_controller.test_direct_camera_control()

        elif key == ord('y') or key == ord('Y'):
            # Attempt to grasp the currently targeted object
            if target_object:
                print(f"Attempting to grasp: {target_object}")
                controller.grasp_object(target_object)
            else:
                print("No target object selected to grasp.")

        elif key == ord('l') or key == ord('L'):
            controller.llm_guided_book_mission()

        elif key == ord('e') or key == ord('E'):
            # Start physics-based exploration with fixed navigation
            print("\n=== Starting physics-based exploration with robot-agent sync ===")
            rl_navigator.physics_explore()

        elif key == ord('n'):
            current_pos = locobot.translation
            forward = locobot.rotation.transform_vector(mn.Vector3(0, 0, -1))
            # Use floor level y-coordinate (important!)
            target_point = mn.Vector3(
                current_pos.x + forward.x * 2.0,
                0.159348,  # Use correct floor height from NavMesh
                current_pos.z + forward.z * 2.0
            )
            print(f"Navigating to point: {target_point}")
            navmesh_navigator.navigate_to(target_point)

        elif key == ord('z') or key == ord('Z'):
            # Return to manual control mode, stopping any autonomous behavior
            print("Switching back to manual control.")
            controller.stop_autonomous_control()

        # --- Build the display view with overlays ---
        # 1. Draw robot coordinates (position & orientation) on the combined view
        display_view = perception.draw_robot_coordinates(combined, robot_position, robot_rotation)

        display_view = visualize_gripper_state(display_view, locobot, dof_map)

        # After drawing the display view in the main loop (around line ~350):
        if video_recorder.is_recording:
            # Make sure to add the current frame to the recording
            video_recorder.add_frame(display_view)
            # Add a small "REC" indicator in the corner
            cv2.circle(display_view, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(display_view, "REC", (45, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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