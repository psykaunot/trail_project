#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
import time
import json
import magnum as mn
import threading
import queue
import matplotlib.pyplot as plt
from collections import deque

from environment import setup_simulator, run_simulator_step, visualize_gripper_state
from controller import Controller, ControlMode, AutonomousBookSearchAgent
from perception.perception import Perception
from utils.object_spawner import ObjectSpawner
from utils.video_recorder import VideoRecorder
from tools import initialize_tools, TOOL_REGISTRY, get_tool_descriptions

tool_descriptions = get_tool_descriptions()
video_recorder = VideoRecorder(output_path="autonomous_book_search.mp4", fps=30)

def main():
    camera_history = {
        'pan': deque(maxlen=200),
        'tilt': deque(maxlen=200),
        'pan_target': deque(maxlen=200),
        'tilt_target': deque(maxlen=200),
        'time': deque(maxlen=200)
    }
    plot_counter = 0

    sim, agent, locobot, motor_ids, motor_settings, dof_map, camera_controller = setup_simulator()

    object_spawner = ObjectSpawner(sim)
    book_object = object_spawner.spawn_book()
    print(f"Book object spawned: {book_object is not None}")

    perception = Perception(model_name="gemma3:4b")
    perception.set_camera_params(sim, "robot_rgb")

    initialize_tools(camera_controller, sim, perception)
    tools = TOOL_REGISTRY

    DRIVE_SPEED = 1.2
    TURN_SPEED  = 0.5
    ARM_SPEED   = 0.7
    GRIP_SPEED  = 0.7
    dt = 1.0 / 30.0

    frame_counter = 0
    controller = Controller(llm_model="qwen3:8b")
    controller.set_robot_controls(
        locobot, motor_ids, motor_settings, dof_map, sim=sim,
        drive_speed=DRIVE_SPEED, turn_speed=TURN_SPEED,
        arm_speed=ARM_SPEED, grip_speed=GRIP_SPEED
    )
    controller.start()

    cv2.namedWindow("Autonomous Book Search", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Autonomous Book Search", 1280, 960)

    print("Initializing camera to floor view...")
    if camera_controller:
        camera_controller.move_camera(0.0, -0.6, absolute=True)
        time.sleep(1.0)

    book_search_agent = AutonomousBookSearchAgent(
        camera_controller=camera_controller,
        perception=perception,
        sim=sim,
        llm_model="qwen3:8b",
        prompts_file="prompt.yaml",
        tool_descriptions=tool_descriptions
    )
    book_search_agent.found_books_lock = threading.Lock()
    video_recorder.start_recording()

    mission_thread = threading.Thread(
        target=book_search_agent.start_mission,
        daemon=True
    )
    mission_thread.start()

    while True:
        try:
            try:
                command = book_search_agent.command_queue.get_nowait()
                try:
                    if command['tool'] == '_get_camera_state':
                        result = camera_controller.get_current_state()
                    elif command['tool'] == '_check_books':
                        obs = sim.get_sensor_observations()
                        current_image = obs['robot_rgb'].copy()
                        books = perception.find_books_in_image(current_image)
                        
                        for book in books:
                            if "bbox" in book:
                                world_pos = perception.get_book_position(book["bbox"])
                                if world_pos is not None:
                                    book["world_position"] = [world_pos[0], world_pos[1], world_pos[2]]
                        
                        result = {
                            'books': books,
                            'camera_state': camera_controller.get_current_state(),
                            'num_objects': 0,
                            'num_books': len(books)
                        }
                    elif command['tool'] == '_check_books_at_scan':
                        obs = sim.get_sensor_observations()
                        current_image = obs['robot_rgb'].copy()
                        books = perception.find_books_in_image(current_image)

                        for book in books:
                            if "bbox" in book:
                                world_pos = perception.get_book_position(book["bbox"])
                                if world_pos is not None:
                                    book["world_position"] = [world_pos[0], world_pos[1], world_pos[2]]

                        # Process through memory
                        unique_books = []
                        if hasattr(book_search_agent, 'book_memory'):
                            camera_state = camera_controller.get_current_state()
                            for book in books:
                                book_id, is_novel = book_search_agent.book_memory.add_observation(book, camera_state)
                                if is_novel:
                                    unique_books.append(book)
                        else:
                            unique_books = books

                        result = {
                            'books': unique_books,
                            'books_found': len(unique_books),
                            'position_index': command['params'].get('position_index', -1)
                        }
                    elif command['tool'] == '_update_book_count':
                        book_count = command['params'].get('count', 0)
                        result = {'status': 'ok', 'count': book_count}
                    elif command['tool'] == '_execute_pick':
                        book_pos = command['params'].get('book_position')
                        if book_pos:
                            from manipulation.pick_place_demo import PickAndPlaceTask
                            task = PickAndPlaceTask(
                                sim=sim, locobot=locobot,
                                motor_ids=motor_ids, motor_settings=motor_settings,
                                dof_map=dof_map,
                                book_object=book_object
                            )
                            success = task.execute_pick_and_place()
                            result = {'success': success}
                        else:
                            result = {'success': False, 'error': 'No book position provided'}
                    else:
                        tool_name = command['tool']
                        params = command.get('params', {})

                        if tool_name in TOOL_REGISTRY:
                            tool = TOOL_REGISTRY[tool_name]
                            try:
                                if tool_name == "move_camera":
                                    result_json = tool(
                                        pan=params.get('pan', 0.0),
                                        tilt=params.get('tilt', 0.0),
                                        duration=params.get('duration', 2.0)
                                    )
                                else:
                                    result_json = tool(**params)

                                try:
                                    result = json.loads(result_json)
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing tool result JSON: {e}")
                                    result = {'status': 'error', 'message': f'JSON parse error: {str(e)}'}

                                result['camera_state'] = camera_controller.get_current_state()
                            except Exception as e:
                                print(f"Error executing tool {tool_name}: {e}")
                                result = {'status': 'error', 'message': str(e)}
                        else:
                            result = {'status': 'error', 'message': f'Unknown tool: {tool_name}'}

                    book_search_agent.result_queue.put(result)

                except Exception as e:
                    print(f"Error processing command {command}: {e}")
                    book_search_agent.result_queue.put({'status': 'error', 'message': str(e)})

            except queue.Empty:
                pass
            except Exception as e:
                print(f"Unexpected error in command processing: {e}")
                import traceback
                traceback.print_exc()

            # Camera command processing
            if camera_controller:
                # Process book checking from camera
                try:
                    cmd = camera_controller.command_queue.get_nowait()
                    if cmd['action'] == 'check_books_at_position':
                        obs = sim.get_sensor_observations()
                        current_image = obs['robot_rgb'].copy()

                        print(f"\n--- CHECKING POSITION {cmd['position'] + 1} ---")
                        print(f"Camera at pan={cmd['pan']:.3f}, tilt={cmd['tilt']:.3f}")

                        # Try general object detection first
                        print("Attempting general object detection...")
                        all_objects = perception.detect_objects(current_image)
                        print(f"Found {len(all_objects)} objects total:")
                        for i, obj in enumerate(all_objects):
                            print(f"  {i+1}. {obj.get('name', 'Unknown')}: {obj.get('description', 'No description')}")
                            if 'bbox' in obj:
                                print(f"     Bbox: {obj['bbox']}")

                        # Then specifically check for books
                        print("\nAttempting book-specific detection...")
                        books = perception.find_books_in_image(current_image)
                        print(f"Found {len(books)} books")

                        # Add world positions to books
                        for book in books:
                            if "bbox" in book:
                                world_pos = perception.get_book_position(book["bbox"])
                                if world_pos is not None:
                                    book["world_position"] = [world_pos[0], world_pos[1], world_pos[2]]
                                    print(f"  Book position: {world_pos}")

                        print("----------------------------\n")

                        if books:
                            # Update book search agent
                            with book_search_agent.found_books_lock:
                                book_search_agent.found_books.extend(books)

                            # Process through memory if available
                            if hasattr(book_search_agent, 'book_memory'):
                                camera_state = camera_controller.get_current_state()
                                for book in books:
                                    book_id, is_novel = book_search_agent.book_memory.add_observation(book, camera_state)
                                    if is_novel:
                                        print(f"New unique book {book_id} found!")
                except queue.Empty:
                    pass

            key = cv2.waitKeyEx(1)
            if key == 27:
                break

            frame_counter += 1

            obs, combined = run_simulator_step(
                sim, agent, locobot,
                motor_ids, motor_settings, dof_map,
                key=None,
                DRIVE_SPEED=DRIVE_SPEED,
                TURN_SPEED=TURN_SPEED,
                ARM_SPEED=ARM_SPEED,
                GRIP_SPEED=GRIP_SPEED,
                dt=dt,
                camera_controller=camera_controller
            )

            if combined.shape[2] == 4:
                combined = cv2.cvtColor(combined, cv2.COLOR_RGBA2BGR)

            robot_position = locobot.translation
            robot_rotation = locobot.rotation

            controller.update_robot_state(robot_position, robot_rotation)

            if camera_controller:
                camera_state = camera_controller.get_current_state()
                camera_history['pan'].append(camera_state['pan'])
                camera_history['tilt'].append(camera_state['tilt'])
                camera_history['pan_target'].append(camera_state['pan_target'])
                camera_history['tilt_target'].append(camera_state['tilt_target'])
                camera_history['time'].append(plot_counter)
                plot_counter += 1

                if plot_counter % 30 == 0:
                    plt.clf()
                    plt.subplot(2,1,1)
                    plt.plot(camera_history['time'], camera_history['pan'], 'b-', label='Pan Current')
                    plt.plot(camera_history['time'], camera_history['pan_target'], 'b--', label='Pan Target')
                    plt.ylabel('Pan (rad)')
                    plt.legend()
                    plt.grid(True)

                    plt.subplot(2,1,2)
                    plt.plot(camera_history['time'], camera_history['tilt'], 'r-', label='Tilt Current')
                    plt.plot(camera_history['time'], camera_history['tilt_target'], 'r--', label='Tilt Target')
                    plt.ylabel('Tilt (rad)')
                    plt.xlabel('Frame')
                    plt.legend()
                    plt.grid(True)

                    plt.suptitle('Camera Movement')
                    plt.tight_layout()
                    plt.pause(0.001)

            display_view = perception.draw_robot_coordinates(combined, robot_position, robot_rotation)
            display_view = visualize_gripper_state(display_view, locobot, dof_map)

            status_height = 60
            status_bg = np.zeros((status_height, display_view.shape[1], 3), dtype=np.uint8)
            
            if mission_thread.is_alive():
                with book_search_agent.found_books_lock:
                    count = len(book_search_agent.found_books)
                status_text = f"AUTONOMOUS MISSION ACTIVE | Books Found: {count} | "
                status_text += f"Positions Scanned: {len(book_search_agent.scanned_positions)}"
            else:
                status_text = "AUTONOMOUS MISSION COMPLETE | "
                status_text += f"Total Books Found: {len(book_search_agent.found_books)}"
            
            cv2.putText(status_bg, status_text, (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            display_view = np.vstack([display_view, status_bg])

            if video_recorder.is_recording:
                video_recorder.add_frame(display_view)
                cv2.circle(display_view, (30, 30), 10, (0, 0, 255), -1)
                cv2.putText(display_view, "REC", (45, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("Autonomous Book Search", display_view)

            if not mission_thread.is_alive():
                with book_search_agent.found_books_lock:
                    count = len(book_search_agent.found_books)
                if count > 0:
                    time.sleep(5.0)
                    break

        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
            break

    if mission_thread.is_alive():
        book_search_agent.stop_mission()
        mission_thread.join(timeout=2.0)

    controller.stop()
    video_recorder.stop_recording()

    print("\n===== FINAL RESULTS =====")
    with book_search_agent.found_books_lock:
        total = len(book_search_agent.found_books)
        print(f"Total books found: {total}")
    for idx, book in enumerate(book_search_agent.found_books):
        print(f"  Book {idx+1}: {book.get('description', 'No description')}")
    print("========================\n")

    cv2.destroyAllWindows()
    sim.close()

if __name__ == '__main__':
    main()