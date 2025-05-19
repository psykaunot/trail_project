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
import yaml

from environment import setup_simulator, run_simulator_step, visualize_gripper_state
from controller import Controller, ControlMode, AutonomousBookSearchAgent
from perception.perception import Perception
from utils.object_spawner import ObjectSpawner
from utils.video_recorder import VideoRecorder
from tools import initialize_tools, TOOL_REGISTRY, get_tool_descriptions, query_semantic_forest, add_spatial_context, visualize_semantic_forest
from manipulation.pick_place_demo import PickAndPlaceTask
import traceback

from memory.semantic_memory import SemanticBookMemory
from perception.yolo_sam_perception import YoloSamPerception

sys.path.append('Embodied_RAG')

tool_descriptions = get_tool_descriptions()
video_recorder = VideoRecorder(output_path="autonomous_book_search.mp4", fps=30)
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)


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
    # Initialize semantic memory
    semantic_memory = SemanticBookMemory(config)
    
    # Initialize perception system based on configuration
    perception_mode = config['perception']['mode']
    print(f"Initializing perception system in mode: {perception_mode}")
    
    if perception_mode == 'vlm':
        # Use VLM-based perception
        # Perception already imported at the top
        vlm_model = config['perception']['vlm']['model']
        perception = Perception(model_name=vlm_model)
        print(f"Using VLM perception with model: {vlm_model}")
    elif perception_mode == 'yolo_sam':
        # Use YOLO-SAM based perception
        perception = YoloSamPerception(
            use_yolo_sam=True,
            confidence_threshold=config['perception']['yolo_sam']['confidence_threshold']
        )
        print("Using YOLO-SAM perception")
    else:
        # Fallback to VLM perception
        # Perception already imported at the top
        perception = Perception(model_name="qwen3:8b")
        print("Invalid perception mode, falling back to VLM perception")
    
    # Set camera parameters for the selected perception system
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

    #book_search_agent = AutonomousBookSearchAgent(
    #    camera_controller=camera_controller,
    #    perception=perception,
    #    sim=sim,
    #    llm_model="qwen3:8b",
    #    prompts_file="prompt.yaml",
    #    tool_descriptions=tool_descriptions
    #)

    book_search_agent = AutonomousBookSearchAgent(
        camera_controller=camera_controller,
        perception=perception,
        sim=sim,
        llm_model=config['ollama']['models']['llm'],
        prompts_file="prompt.yaml",
        tool_descriptions=tool_descriptions,
        semantic_memory=semantic_memory,  # Pass the semantic memory
        config=config
    )

    # Make sure the forest references are properly initialized
    if hasattr(book_search_agent, 'book_memory') and hasattr(book_search_agent.book_memory, 'semantic_forest'):
        # Use safe attribute assignment
        if 'query_semantic_forest' in globals():
            query_semantic_forest.forest = book_search_agent.book_memory.semantic_forest
            print("Initialized query_semantic_forest tool with book memory")
            
        if 'add_spatial_context' in globals():
            add_spatial_context.forest = book_search_agent.book_memory.semantic_forest
            print("Initialized add_spatial_context tool with book memory")


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
                        
                        for idx, book in enumerate(books):
                            if "bbox" in book:
                                # get_book_position now always returns a tuple of (position, object_reference)
                                world_pos, book_obj = perception.get_book_position(book["bbox"])
                                
                                if world_pos is not None:
                                    # Successfully got position
                                    book["world_position"] = [world_pos[0], world_pos[1], world_pos[2]]
                                    
                                    if book_obj is not None:
                                        # Got object reference from perception
                                        book["object_ref"] = book_obj
                                        print(f"Book {idx} has position AND object reference from perception")
                                    elif book_object is not None:
                                        # Use global book object as fallback
                                        book["object_ref"] = book_object
                                        print(f"Book {idx} has position but using global fallback object reference")
                                    else:
                                        print(f"Book {idx} has position but NO object reference available")
                                else:
                                    print(f"Book {idx} position calculation failed")

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

                        for book_idx, book in enumerate(books):
                            if "bbox" in book:
                                # get_book_position now always returns a tuple of (position, object_reference)
                                world_pos, book_obj = perception.get_book_position(book["bbox"])
                                
                                if world_pos is not None:
                                    # Successfully got position
                                    book["world_position"] = [world_pos[0], world_pos[1], world_pos[2]]
                                    
                                    if book_obj is not None:
                                        # Got object reference from perception
                                        book["object_ref"] = book_obj
                                        print(f"Book at scan {book_idx} has position AND object reference from perception")
                                    elif book_object is not None:
                                        # Use global book object as fallback
                                        book["object_ref"] = book_object
                                        print(f"Book at scan {book_idx} has position but using global fallback object reference")
                                    else:
                                        print(f"Book at scan {book_idx} has position but NO object reference available")
                                else:
                                    print(f"Book at scan {book_idx} position calculation failed")

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
                        target_book_obj = command['params'].get('book_object')
                        book_id = command['params'].get('book_id')
                        
                        # Enhanced logging
                        print(f"\n===== EXECUTING BOOK PICK =====")
                        print(f"Book ID: {book_id}")
                        print(f"Book Position: {book_pos}")
                        print(f"Has book object: {target_book_obj is not None}")
                        
                        # Check if the book is within reach
                        in_reach = True  # Default to True
                        try:
                            # Already imported at the top of the file, don't re-import
                            robot_pos = locobot.translation
                            book_position = np.array(book_pos)
                            robot_position = np.array([robot_pos[0], robot_pos[1], robot_pos[2]])
                            distance = np.linalg.norm(book_position - robot_position)
                            print(f"Distance to book: {distance:.3f}m")
                            
                            # Check if too far
                            if distance > 1.5:  # 1.5 meters is roughly the maximum reach
                                in_reach = False
                                print(f"Book is too far to reach ({distance:.3f}m > 1.5m)")
                            else:
                                print(f"Book is within reach ({distance:.3f}m <= 1.5m)")
                        except Exception as e:
                            print(f"Error calculating distance: {e}")
                        
                        # Only proceed if book is in reach
                        if in_reach:
                            # Log what we're using
                            if book_pos:
                                print(f"Executing pick with position: {book_pos}")
                                
                                # Determine which book object to use
                                pick_book_obj = None
                                if target_book_obj is not None:
                                    print("Using provided book object reference")
                                    pick_book_obj = target_book_obj
                                elif book_object is not None:
                                    print("Using fallback global book object")
                                    pick_book_obj = book_object
                                else:
                                    print("Warning: No book object available - searching for any book in simulator")
                                    
                                    # Last resort - search for any book in simulator
                                    try:
                                        rigid_obj_mgr = sim.get_rigid_object_manager()
                                        for obj_handle in rigid_obj_mgr.get_object_handles():
                                            if "book" in obj_handle.lower():
                                                pick_book_obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
                                                print(f"Found book object: {obj_handle}")
                                                break
                                    except Exception as e:
                                        print(f"Error searching for book objects: {e}")
                                
                                if pick_book_obj is not None:
                                    print(f"Book object position: {pick_book_obj.translation}")
                                    print(f"Using book object ID: {pick_book_obj.object_id}")
                                
                                try:
                                    # PickAndPlaceTask already imported at the top
                                    task = PickAndPlaceTask(
                                        sim=sim, locobot=locobot,
                                        motor_ids=motor_ids, motor_settings=motor_settings,
                                        dof_map=dof_map,
                                        book_object=pick_book_obj
                                    )
                                    
                                    # Execute picking with enhanced camera guidance
                                    print("Starting camera-guided book picking...")
                                    # Actually execute the pick operation and get the result
                                    success = task.camera_guided_grasp()
                                    
                                    # Update memory if available
                                    if hasattr(book_search_agent, 'book_memory'):
                                        try:
                                            memory = book_search_agent.book_memory
                                            if hasattr(memory, 'record_pick_action') and callable(memory.record_pick_action):
                                                memory.record_pick_action(book_pos)
                                                print(f"Recorded successful pick in memory system")
                                        except Exception as e:
                                            print(f"Error updating memory: {e}")
                                    
                                    result = {'success': success, 'book_id': book_id}
                                    print(f"Pick operation completed: {success}")
                                except Exception as e:
                                    print(f"Error executing pick operation: {e}")
                                    traceback.print_exc()
                                    result = {'success': False, 'error': str(e)}
                            else:
                                result = {'success': False, 'error': 'No book position provided'}
                        else:
                            result = {'success': False, 'error': 'Book is too far to reach', 'distance': float(distance)}
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
                        for book_idx, book in enumerate(books):
                            if "bbox" in book:
                                # get_book_position now always returns a tuple of (position, object_reference)
                                world_pos, book_obj = perception.get_book_position(book["bbox"])
                                
                                if world_pos is not None:
                                    # Successfully got position
                                    book["world_position"] = [world_pos[0], world_pos[1], world_pos[2]]
                                    
                                    if book_obj is not None:
                                        # Got object reference from perception
                                        book["object_ref"] = book_obj
                                        print(f"  Book {book_idx} position: {world_pos} WITH object reference from perception")
                                    elif book_object is not None:
                                        # Use global book object as fallback
                                        book["object_ref"] = book_object
                                        print(f"  Book {book_idx} position: {world_pos} (using global fallback object reference)")
                                    else:
                                        print(f"  Book {book_idx} position: {world_pos} but NO object reference available")
                                else:
                                    print(f"  Book {book_idx} position calculation failed")

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
                                        
                                        # Automatically attempt to pick the book if it has position and object reference
                                        if "world_position" in book:
                                            # Check if the book is within reach before attempting pick
                                            try:
                                                # numpy already imported at top of file, don't re-import
                                                robot_pos = locobot.translation
                                                book_position = np.array(book["world_position"])
                                                robot_position = np.array([robot_pos[0], robot_pos[1], robot_pos[2]])
                                                distance = np.linalg.norm(book_position - robot_position)
                                                
                                                # Print debug information
                                                print(f"\n===== BOOK PROXIMITY CHECK =====")
                                                print(f"Robot position: {robot_position}")
                                                print(f"Book position: {book_position}")
                                                print(f"Distance to book: {distance:.3f}m")
                                                print(f"Book within reach: {distance <= 1.5}")
                                                
                                                # Only attempt to pick if the book is within reach
                                                if distance <= 1.5:  # 1.5 meters is roughly the maximum reach
                                                    print(f"Attempting to automatically pick book {book_id}")
                                                    
                                                    # Create pick command
                                                    pick_params = {
                                                        'book_position': book["world_position"],
                                                        'book_id': book_id
                                                    }
                                                    
                                                    # Add book object reference if available
                                                    if "object_ref" in book and book["object_ref"] is not None:
                                                        pick_params['book_object'] = book["object_ref"]
                                                        print(f"Using detected book object reference for picking")
                                                    else:
                                                        # Fallback to global book object
                                                        pick_params['book_object'] = book_object
                                                        print(f"Using global book object for picking")
                                                    
                                                    # Queue the pick command
                                                    book_search_agent.command_queue.put({
                                                        'tool': '_execute_pick',
                                                        'params': pick_params
                                                    })
                                                else:
                                                    print(f"Book {book_id} is too far to reach ({distance:.3f}m > 1.5m)")
                                            except Exception as e:
                                                print(f"Error checking if book is in reach: {e}")
                                                traceback.print_exc()
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

            if hasattr(book_search_agent, 'book_memory') and hasattr(book_search_agent.book_memory, 'semantic_forest'):
                # Update robot position in visualization tool
                # visualize_semantic_forest already imported at the top
                visualize_semantic_forest.robot_position = locobot.translation

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

    if hasattr(book_search_agent, 'book_memory'):
        try:
            # Make sure we have a valid memory save path
            memory_save_path = config.get('paths', {}).get('memory_save', 'semantic_forest_save.json')
            book_search_agent.book_memory.save(memory_save_path)
            print(f"Semantic forest saved to {memory_save_path}")
        except Exception as e:
            print(f"Error saving semantic forest: {e}")
            # Fallback to default path
            book_search_agent.book_memory.save("semantic_forest_save.json")
            print("Semantic forest saved to default path: semantic_forest_save.json")

    cv2.destroyAllWindows()
    sim.close()

if __name__ == '__main__':
    main()