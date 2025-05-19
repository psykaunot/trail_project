#!/usr/bin/env python3
import json
import inspect
import numpy as np
from typing import Callable, Dict, Any
from dataclasses import dataclass
import sys
import os

# Add Embodied_RAG to path if it exists
embodied_rag_path = os.path.join(os.path.dirname(__file__), 'Embodied_RAG')
if os.path.exists(embodied_rag_path):
    sys.path.append(embodied_rag_path)

try:
    from embodied_nav.embodied_retriever import EmbodiedRetriever
    from embodied_nav.spatial_relationship_extractor import SpatialRelationshipExtractor
    print("Imported EmbodiedRetriever and SpatialRelationshipExtractor from Embodied_RAG")
except ImportError:
    # Fall back to stub versions
    from embodied_rag_stubs import EmbodiedRetriever, SpatialRelationshipExtractor
    print("Using stub versions of EmbodiedRetriever and SpatialRelationshipExtractor")

# Simple @tool decorator implementation
@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: Dict[str, str]
    
    def __call__(self, **kwargs):
        return self.func(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format for prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


def tool(func: Callable) -> Tool:
    """Decorator to mark a function as a tool for the agent."""
    # Extract function signature
    sig = inspect.signature(func)
    parameters = {}
    
    for param_name, param in sig.parameters.items():
        if param.annotation != param.empty:
            parameters[param_name] = str(param.annotation).replace("<class '", "").replace("'>", "")
        else:
            parameters[param_name] = "any"
    
    # Create tool instance
    return Tool(
        name=func.__name__,
        description=func.__doc__ or "",
        func=func,
        parameters=parameters
    )


# Camera control tools
@tool
def start_scan(pattern: str = "book_search", pause_time: float = 3.0) -> str:
    """Begin systematic environment scanning with specified pattern."""
    global camera_controller
    
    valid_patterns = ["book_search", "wide", "detailed", "horizontal", "vertical", "continuous"]
    if pattern not in valid_patterns:
        return json.dumps({
            "status": "error",
            "message": f"Invalid pattern '{pattern}'. Use: {', '.join(valid_patterns)}"
        })
    
    try:
        success = camera_controller.start_scanning(pattern_name=pattern, pause_time=pause_time)
        if success:
            print(f"Started scanning with pattern '{pattern}'")
        return json.dumps({
            "status": "started" if success else "failed",
            "pattern": pattern,
            "pause_time": pause_time
        })
    except Exception as e:
        print(f"Error in start_scan: {e}")
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def move_camera(pan: float, tilt: float, duration: float = 2.0) -> str:
    """Move camera to specific pan/tilt position within defined limits."""
    global camera_controller
    
    try:
        # Move to absolute position
        success = camera_controller.move_camera(pan, tilt, absolute=True)
        
        # Wait for movement to complete
        import time
        start_time = time.time()
        while camera_controller.is_moving and time.time() - start_time < duration:
            time.sleep(0.1)
        
        # Get final position
        state = camera_controller.get_current_state()
        
        return json.dumps({
            "status": "completed" if success else "failed",
            "final_position": {
                "pan": state["pan"],
                "tilt": state["tilt"]
            },
            "duration": time.time() - start_time
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def examine_area(duration: float = 5.0, focus_on: str = None) -> str:
    """Carefully examine current view for books and objects."""
    global sim, perception
    
    try:
        import time
        
        # Get current camera view
        obs = sim.get_sensor_observations()
        current_image = obs['robot_rgb'].copy()
        
        # Perform detailed analysis
        analysis_prompt = f"Examine this view carefully for {duration} seconds."
        if focus_on:
            analysis_prompt += f" Focus particularly on: {focus_on}"
        
        result = perception.process_image(current_image, analysis_prompt)
        
        # Check specifically for books
        books = perception.find_books_in_image(current_image)
        
        # Wait for the specified duration
        time.sleep(duration)
        
        return json.dumps({
            "status": "completed",
            "books_found": len(books),
            "analysis": result[:200] + "..." if len(result) > 200 else result,
            "focus": focus_on
        })
    except Exception as e:
        return json.dumps({
            "status": "error", 
            "message": str(e)
        })


@tool
def analyze_view(analysis_type: str = "general") -> str:
    """Request detailed analysis of current camera view."""
    global sim, perception
    
    try:
        # Get current camera view
        obs = sim.get_sensor_observations()
        current_image = obs['robot_rgb'].copy()
        
        if analysis_type == "books":
            books = perception.find_books_in_image(current_image)
            result = {
                "type": "books",
                "count": len(books),
                "books": books
            }
        elif analysis_type == "objects":
            objects = perception.detect_objects(current_image)
            result = {
                "type": "objects",
                "count": len(objects),
                "objects": objects
            }
        else:
            scene_analysis = perception.analyze_scene(current_image)
            result = {
                "type": "general",
                "analysis": scene_analysis
            }
        
        return json.dumps({
            "status": "completed",
            "analysis_type": analysis_type,
            "result": result
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })


@tool
def query_semantic_forest(query: str, top_k: int = 5, use_spatial: bool = True) -> str:
    """Query semantic forest for book information with improved spatial context.
    
    Args:
        query: Natural language query about books
        top_k: Number of results to return
        use_spatial: Whether to include spatial relationships in results
        
    Returns:
        JSON string with retrieved book information
    """
    if not hasattr(query_semantic_forest, 'forest'):
        return json.dumps({"status": "error", "message": "Semantic forest not initialized"})
    
    try:
        # Create retriever with improved matching
        retriever = EmbodiedRetriever(query_semantic_forest.forest)
        results = retriever.retrieve(query, top_k=top_k)
        
        formatted_results = []
        for node in results:
            result = {
                "book_id": node.node_id,
                "description": node.description,
                "position": node.attributes.get("world_position"),
                "last_seen": node.attributes.get("timestamp")
            }
            
            # Add spatial relationships if requested
            if use_spatial and "spatial_context" in node.attributes:
                result["spatial_relationships"] = node.attributes["spatial_context"]
                
            formatted_results.append(result)
        
        return json.dumps({
            "status": "success",
            "results": formatted_results,
            "count": len(formatted_results),
            "query": query
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

@tool
def add_spatial_context(book_id: str, nearby_books: list) -> str:
    """Add spatial relationships between books.
    
    Args:
        book_id: ID of target book
        nearby_books: List of nearby book IDs with positions
        
    Returns:
        JSON string with updated spatial context
    """
    if not hasattr(add_spatial_context, 'forest'):
        return json.dumps({"status": "error", "message": "Semantic forest not initialized"})
    
    try:
        extractor = SpatialRelationshipExtractor()
        target_node = add_spatial_context.forest.get_node(book_id)
        
        if not target_node:
            return json.dumps({"status": "error", "message": "Book not found"})
        
        spatial_relations = []
        for nearby in nearby_books:
            relation = extractor.compute_relation(
                target_node.attributes["world_position"],
                nearby["position"]
            )
            spatial_relations.append({
                "book_id": nearby["id"],
                "relation": relation
            })
        
        target_node.attributes["spatial_context"] = spatial_relations
        add_spatial_context.forest.update_node(target_node)
        
        return json.dumps({
            "status": "success",
            "book_id": book_id,
            "spatial_relations": spatial_relations
        })
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
    

@tool
def visualize_semantic_forest(width: int = 800, height: int = 600) -> str:
    """Create a visual representation of the semantic forest.
    
    Args:
        width: Width of the visualization
        height: Height of the visualization
        
    Returns:
        JSON string with visualization data
    """
    if not hasattr(visualize_semantic_forest, 'forest'):
        return json.dumps({"status": "error", "message": "Semantic forest not initialized"})
        
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import io
        import base64
        
        # Create a 3D plot
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot book nodes
        book_positions = []
        book_labels = []
        
        for node in visualize_semantic_forest.forest.get_all_nodes():
            if node.node_type == "book_instance" and "world_position" in node.attributes:
                pos = node.attributes["world_position"]
                if pos and len(pos) == 3:
                    book_positions.append(pos)
                    book_labels.append(node.description[:20])  # Truncate long descriptions
        
        if book_positions:
            book_positions = np.array(book_positions)
            ax.scatter(book_positions[:,0], book_positions[:,2], book_positions[:,1], 
                      c='blue', marker='o', s=100, label='Books')
            
            # Add labels for books
            for i, (pos, label) in enumerate(zip(book_positions, book_labels)):
                ax.text(pos[0], pos[2], pos[1], f"{i+1}", fontsize=10)
        
        # Add robot position if available - with enhanced display
        if hasattr(visualize_semantic_forest, 'robot_position'):
            robot_pos = visualize_semantic_forest.robot_position
            
            # Plot robot position with larger marker and different color for visibility
            ax.scatter([robot_pos[0]], [robot_pos[2]], [robot_pos[1]], 
                      c='red', marker='^', s=300, label='Robot')
            
            # Add text label "ROBOT" near the position
            ax.text(robot_pos[0], robot_pos[2], robot_pos[1] + 0.2, 
                   "ROBOT", color='red', fontweight='bold', fontsize=12)
            
            # Plot a vertical line from robot to ground for better spatial understanding
            ax.plot([robot_pos[0], robot_pos[0]], 
                   [robot_pos[2], robot_pos[2]], 
                   [robot_pos[1], 0], 'r--', linewidth=2)
            
            # Print position information
            print(f"Robot position in visualization: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})")
        
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title('Semantic Forest Visualization')
        ax.legend()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode as base64
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        
        # Create legend mapping indices to descriptions
        legend = {str(i+1): label for i, label in enumerate(book_labels)}
        
        return json.dumps({
            "status": "success",
            "visualization": img_data,
            "format": "base64_png",
            "legend": legend,
            "book_count": len(book_positions)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "message": str(e)})

# Global references (set by main application)
camera_controller = None
sim = None
perception = None


def initialize_tools(controller, simulator, perception_module):
    """Initialize tool references."""
    global camera_controller, sim, perception
    camera_controller = controller
    sim = simulator
    perception = perception_module


def get_tool_descriptions() -> str:
    """Get formatted tool descriptions for prompt."""
    tools = [start_scan, move_camera, examine_area, analyze_view]
    descriptions = []
    
    for idx, tool in enumerate(tools, 1):
        desc = f"{idx}. {tool.name}\n"
        desc += f"   Description: {tool.description}\n"
        desc += f"   Parameters:\n"
        for param, ptype in tool.parameters.items():
            desc += f"   - {param}: {ptype}\n"
        
        descriptions.append(desc)
    
    return "\n".join(descriptions)

def get_tool_descriptions() -> str:
    """Get formatted tool descriptions for prompt."""
    print("DEBUG: get_tool_descriptions() called")
    tools = [
        start_scan, 
        move_camera, 
        examine_area, 
        analyze_view, 
        query_semantic_forest, 
        add_spatial_context, 
        visualize_semantic_forest,
        pick_object,
        camera_guided_pick
    ]
    descriptions = []
    
    for idx, tool in enumerate(tools, 1):
        print(f"DEBUG: Processing tool {idx}: {tool.name}")
        desc = f"{idx}. {tool.name}\n"
        desc += f"   Description: {tool.description}\n"
        desc += f"   Parameters:\n"
        for param, ptype in tool.parameters.items():
            desc += f"   - {param}: {ptype}\n"
        
        descriptions.append(desc)
    
    print("DEBUG: Tool descriptions complete")
    return "\n".join(descriptions)

# Create picking tools
@tool
def pick_object(book_id: str = None, position: list = None):
    """Pick an object by ID or position.
    
    Args:
        book_id: ID of the book to pick (from semantic forest)
        position: Position [x, y, z] to pick from
    
    Returns:
        JSON string with pick operation result
    """
    if sim is None:
        return json.dumps({"status": "error", "message": "Simulator not initialized"})
        
    try:
        # Validate input
        if book_id is None and position is None:
            return json.dumps({"status": "error", "message": "Either book_id or position must be provided"})
            
        # If book_id is provided, try to get book object and position from semantic forest
        book_object = None
        if book_id is not None and hasattr(query_semantic_forest, 'forest'):
            forest = query_semantic_forest.forest
            
            # Try to find the node
            node = forest.get_node(book_id) if hasattr(forest, 'get_node') else None
            
            # If node was found and has position, extract it
            if node and hasattr(node, 'attributes'):
                # Get position if available
                if 'world_position' in node.attributes:
                    position = node.attributes['world_position']
                
                # Check if node has object reference
                if 'object_ref' in node.attributes and node.attributes['object_ref'] is not None:
                    book_object = node.attributes['object_ref']
                    print(f"Got book object reference from semantic forest")
            else:
                # Fallback to old-style dictionary lookup
                nodes = forest.get('nodes', {})
                if book_id in nodes:
                    node_data = nodes[book_id]
                    if 'attributes' in node_data:
                        if 'world_position' in node_data['attributes']:
                            position = node_data['attributes']['world_position']
                        if 'object_ref' in node_data['attributes'] and node_data['attributes']['object_ref'] is not None:
                            book_object = node_data['attributes']['object_ref']
                            print(f"Got book object reference from semantic forest dictionary")
                    
        # If position is still None, fail
        if position is None:
            return json.dumps({"status": "error", "message": "Could not determine position for picking"})
            
        # Check if book location is within reach
        robot_pos = None
        if hasattr(visualize_semantic_forest, 'robot_position'):
            robot_pos = visualize_semantic_forest.robot_position
        
        if robot_pos:
            import numpy as np
            distance = np.linalg.norm(np.array(position) - np.array(robot_pos))
            if distance > 1.5:  # Too far to reach
                return json.dumps({
                    "status": "error", 
                    "message": f"Target position too far to reach: {distance:.2f}m (max: 1.5m)",
                    "distance": distance,
                    "book_id": book_id,
                    "position": position
                })
        
        # If we don't have a book object yet, try to find one in the simulator
        if book_object is None:
            try:
                print("No book object reference found in memory, searching in simulator...")
                rigid_obj_mgr = sim.get_rigid_object_manager()
                all_objects = rigid_obj_mgr.get_object_handles()
                
                print(f"Available object handles: {all_objects}")
                
                # First try to find a book near the target position
                closest_book = None
                closest_distance = float('inf')
                
                for obj_handle in all_objects:
                    if "book" in obj_handle.lower():
                        obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
                        obj_pos = obj.translation
                        dist = np.linalg.norm(np.array(position) - np.array([obj_pos[0], obj_pos[1], obj_pos[2]]))
                        print(f"Book {obj_handle} at distance {dist:.3f}m from target position")
                        
                        if dist < closest_distance:
                            closest_distance = dist
                            closest_book = obj
                
                # Use the closest book if it's reasonably close
                if closest_book is not None and closest_distance < 0.5:
                    book_object = closest_book
                    print(f"Using closest book object at distance {closest_distance:.3f}m")
                # Fallback to any book if no close match
                elif len(all_objects) > 0:
                    for obj_handle in all_objects:
                        if "book" in obj_handle.lower():
                            book_object = rigid_obj_mgr.get_object_by_handle(obj_handle)
                            print(f"Using fallback book object: {obj_handle}")
                            break
            except Exception as e:
                print(f"Warning: Could not get book object: {e}")
                import traceback
                traceback.print_exc()
        
        # Execute pick operation as a command to the main loop
        from main import book_search_agent
        if hasattr(book_search_agent, 'command_queue'):
            # Include both position and object reference
            params = {
                'book_position': position,
                'book_id': book_id
            }
            
            # Add object reference if available
            if book_object is not None:
                params['book_object'] = book_object
                print(f"Sending pick command with book object reference")
            else:
                print(f"WARNING: Sending pick command WITHOUT book object reference")
                
            book_search_agent.command_queue.put({
                'tool': '_execute_pick',
                'params': params
            })
            
            # Wait for the result
            try:
                result = book_search_agent.result_queue.get(timeout=5.0)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({
                    "status": "error",
                    "message": f"Error waiting for pick command result: {e}"
                })
        else:
            # Direct execution fallback
            from manipulation.pick_place_demo import PickAndPlaceTask
            task = PickAndPlaceTask(
                sim=sim,
                locobot=None,  # Will be updated in main.py
                motor_ids=None,  # Will be updated in main.py
                motor_settings=None,  # Will be updated in main.py
                dof_map=None,  # Will be updated in main.py
                book_object=book_object
            )
            success = task.execute_pick_and_place()
            
            return json.dumps({
                "status": "success" if success else "error",
                "message": "Pick operation completed" if success else "Failed to pick object",
                "book_id": book_id,
                "position": position
            })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({"status": "error", "message": f"Error picking object: {str(e)}"})

@tool
def camera_guided_pick(confidence_threshold: float = 0.5):
    """Use camera guidance to pick the most visible book.
    
    Args:
        confidence_threshold: Minimum confidence for book detection
    
    Returns:
        JSON string with guided pick operation result
    """
    if sim is None or perception is None:
        return json.dumps({"status": "error", "message": "Simulator or perception system not initialized"})
        
    try:
        print("\n===== STARTING ENHANCED CAMERA-GUIDED PICK =====")
        
        # Mark robot arm as active and in picking mode in environment
        try:
            import time
            import sys
            sys.path.append(".")
            from environment import _is_arm_moving, _is_gripper_moving, _exploration_active, _last_command_time
            import environment
            # Force all relevant flags for arm movement
            environment._is_arm_moving = True
            environment._is_gripper_moving = True
            environment._exploration_active = False
            environment._last_command_time = time.time()
            print("Successfully set environment flags to enable arm movement")
            print(f"DEBUG: _is_arm_moving={environment._is_arm_moving}, _is_gripper_moving={environment._is_gripper_moving}")
        except Exception as e:
            print(f"WARNING: Could not set environment flags: {e}")
            import traceback
            traceback.print_exc()
        
        # Set camera controller to picking mode
        from main import camera_controller
        if camera_controller:
            camera_controller.is_picking = True
            print("Set camera controller to picking mode")
            # Also try to directly modify environment flags through camera controller
            if hasattr(camera_controller, '_camera_env'):
                if hasattr(camera_controller._camera_env, '_is_arm_moving'):
                    camera_controller._camera_env._is_arm_moving = True
                if hasattr(camera_controller._camera_env, '_is_gripper_moving'):
                    camera_controller._camera_env._is_gripper_moving = True
                print("Updated environment flags through camera controller")
        
        # Get current observation
        obs = sim.get_sensor_observations()
        current_image = obs['robot_rgb'].copy()
        
        # Find books in the image with proper threshold
        books = perception.find_books_in_image(current_image, confidence_threshold=confidence_threshold)
        
        if not books:
            print("No books detected in current camera view.")
            camera_controller.is_picking = False if camera_controller else None
            return json.dumps({"status": "error", "message": "No books visible for picking"})
        
        # Get the most confident book
        best_book = max(books, key=lambda b: b.get('confidence', 0))
        print(f"Selected best book with confidence: {best_book.get('confidence', 0)}")
        
        # Get the book position and object reference (now always returns a tuple)
        position = None
        book_obj = None
        
        if "bbox" in best_book:
            # get_book_position now always returns (position, object_reference)
            world_pos, book_obj = perception.get_book_position(best_book["bbox"])
            
            if world_pos is not None:
                position = [world_pos[0], world_pos[1], world_pos[2]]
                print(f"Detected book position: {position}")
                
                if book_obj is not None:
                    print(f"Camera-guided pick found position AND object reference")
                else:
                    print(f"Camera-guided pick found position but NO object reference from perception")
                    
                    # Try to find a book object in the simulator as fallback
                    try:
                        rigid_obj_mgr = sim.get_rigid_object_manager()
                        all_objects = rigid_obj_mgr.get_object_handles()
                        
                        # Look for a book object near position
                        book_objects = []
                        for obj_handle in all_objects:
                            if "book" in obj_handle.lower():
                                book_objects.append(obj_handle)
                        
                        if book_objects:
                            # Find closest book to detected position
                            closest_book = None
                            closest_distance = float('inf')
                            for obj_handle in book_objects:
                                obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
                                obj_pos = obj.translation
                                dist = ((obj_pos[0] - position[0])**2 + 
                                        (obj_pos[1] - position[1])**2 + 
                                        (obj_pos[2] - position[2])**2)**0.5
                                if dist < closest_distance:
                                    closest_distance = dist
                                    closest_book = obj
                            
                            if closest_book and closest_distance < 0.5:
                                book_obj = closest_book
                                print(f"Found closest book object at distance {closest_distance:.3f}m")
                            else:
                                # Fallback to first book
                                book_handle = book_objects[0]
                                book_obj = rigid_obj_mgr.get_object_by_handle(book_handle)
                                print(f"Found fallback book object: {book_handle}")
                    except Exception as e:
                        print(f"Warning: Could not get fallback book object: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                camera_controller.is_picking = False if camera_controller else None
                print("ERROR: Could not determine 3D book position")
                return json.dumps({"status": "error", "message": "Could not determine book position"})
        else:
            camera_controller.is_picking = False if camera_controller else None
            print("ERROR: No bounding box found for book")
            return json.dumps({"status": "error", "message": "No bounding box found for book"})
        
        # Unlock arm joints if needed
        print("UNLOCKING ALL ARM JOINTS...")
        from main import locobot, motor_ids, motor_settings, dof_map
        import habitat_sim
        
        try:
            # Apply multi-method approach to unlock joints with retries
            print("\n======= CRITICAL JOINT UNLOCKING PROCESS =======")
            
            # Apply several methods in sequence for all arm joints
            arm_joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
            
            # First print all joint states for debugging
            print("Initial Joint State Check:")
            for joint_name in arm_joints:
                if joint_name in dof_map:
                    joint_id = dof_map[joint_name]
                    joint_type = locobot.get_link_joint_type(joint_id)
                    pos_offset = locobot.get_link_joint_pos_offset(joint_id)
                    current_pos = locobot.joint_positions[pos_offset] if pos_offset >= 0 else "unknown"
                    print(f"  {joint_name}: type={joint_type}, position={current_pos}")
            
            # First, attempt to unlock joints directly
            print("\nMethod 1: Direct Joint Type Change")
            for joint_name in arm_joints:
                if joint_name in dof_map:
                    joint_id = dof_map[joint_name]
                    
                    # Always try to force the joint type regardless of current state
                    try:
                        if hasattr(locobot, 'set_link_joint_type'):
                            print(f"  Forcing {joint_name} to REVOLUTE type...")
                            locobot.set_link_joint_type(joint_id, habitat_sim.physics.JointType.Revolute)
                            
                            # Verify the change
                            new_type = locobot.get_link_joint_type(joint_id)
                            if new_type == habitat_sim.physics.JointType.Revolute:
                                print(f"  ✓ Successfully set {joint_name} to REVOLUTE")
                            else:
                                print(f"  ✗ Failed to change type: still {new_type}")
                    except Exception as e:
                        print(f"  ✗ Error changing {joint_name} type: {e}")
            
            # Method 2: Force position change
            print("\nMethod 2: Force Position Change")
            for joint_name in arm_joints:
                if joint_name in dof_map:
                    joint_id = dof_map[joint_name]
                    pos_offset = locobot.get_link_joint_pos_offset(joint_id)
                    
                    if pos_offset >= 0:
                        try:
                            # Get current position
                            joint_positions = locobot.joint_positions
                            current_pos = joint_positions[pos_offset]
                            
                            # Apply an alternating wiggle to shake the joint loose
                            print(f"  Forcing position change for {joint_name}...")
                            for i in range(3):  # Try 3 small movements
                                delta = 0.02 * (-1)**i  # Alternate between +0.02 and -0.02
                                joint_positions[pos_offset] = current_pos + delta
                                locobot.joint_positions = joint_positions
                                
                                # Let physics apply the change
                                for _ in range(3):
                                    sim.step_physics(1/60.0)
                            
                            print(f"  ✓ Applied position wiggle to {joint_name}")
                        except Exception as e:
                            print(f"  ✗ Error forcing position change: {e}")
            
            # Method 3: Set Motion Type and apply extreme settings
            print("\nMethod 3: Apply Extreme Motor Settings")
            for joint_name in arm_joints:
                if joint_name in dof_map:
                    joint_id = dof_map[joint_name]
                    
                    # Try to force object motion type
                    try:
                        link_obj = locobot.get_link_object(joint_id)
                        if link_obj and hasattr(link_obj, 'motion_type'):
                            print(f"  Setting {joint_name} motion type to KINEMATIC")
                            link_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                    except Exception as e:
                        print(f"  Error setting motion type: {e}")
                    
                    # Apply extreme motor settings
                    print(f"  Applying extreme motor settings to {joint_name}")
                    motor_settings[joint_id] = habitat_sim.physics.JointMotorSettings(
                        position_target=0.0,
                        position_gain=15000.0,  # Extremely high gain
                        velocity_target=0.0,
                        velocity_gain=1500.0,   # Very high damping
                        max_impulse=150000.0    # Maximum possible force
                    )
                    
                    if joint_id in motor_ids:
                        try:
                            # Apply the settings multiple times to ensure they take effect
                            for _ in range(5):
                                locobot.update_joint_motor(motor_ids[joint_id], motor_settings[joint_id])
                                sim.step_physics(1/100.0)  # Small physics step to apply settings
                            print(f"  ✓ Applied extreme settings to {joint_name}")
                        except Exception as e:
                            print(f"  ✗ Error applying motor settings: {e}")
            
            # Final verification
            print("\nVerifying Joint Status After Unlocking:")
            for joint_name in arm_joints:
                if joint_name in dof_map:
                    joint_id = dof_map[joint_name]
                    joint_type = locobot.get_link_joint_type(joint_id)
                    print(f"  {joint_name}: {'STILL LOCKED' if joint_type == habitat_sim.physics.JointType.Fixed else 'UNLOCKED'} (type: {joint_type})")
            
            print("\n==================================================\n")
        except Exception as e:
            print(f"Error unlocking joints: {e}")
            import traceback
            traceback.print_exc()
        
        # Execute the camera-guided pick with more precise control
        from manipulation.pick_place_demo import PickAndPlaceTask
        
        # Get locobot and controls from the main simulation - redundant but kept for clarity
        
        # Make sure we have the best possible book object reference
        from main import book_object as global_book_object
        
        # Use provided book object, or fall back to global if necessary
        pick_book_obj = book_obj
        if pick_book_obj is None and global_book_object is not None:
            pick_book_obj = global_book_object
            print("Using global fallback book object reference")
        
        if pick_book_obj is None:
            print("WARNING: No book object reference found for camera-guided pick")
            # Final attempt - look for any book
            try:
                rigid_obj_mgr = sim.get_rigid_object_manager()
                for obj_handle in rigid_obj_mgr.get_object_handles():
                    if "book" in obj_handle.lower():
                        pick_book_obj = rigid_obj_mgr.get_object_by_handle(obj_handle)
                        print(f"Using last-resort book object: {obj_handle}")
                        break
            except Exception as e:
                print(f"Warning: Last resort book object search failed: {e}")
        
        # Create the task with the best book object reference
        task = PickAndPlaceTask(
            sim=sim,
            locobot=locobot,
            motor_ids=motor_ids,
            motor_settings=motor_settings,
            dof_map=dof_map,
            book_object=pick_book_obj
        )
        
        print("EXECUTING CAMERA-GUIDED GRASP WITH ENHANCED DEBUGGING...")
        
        # Use camera guidance for more precise picking
        success = task.camera_guided_grasp()
        
        print(f"Camera-guided grasp {'SUCCEEDED' if success else 'FAILED'}")
        
        # Record successful pick in memory system if available
        if success:
            try:
                from main import book_search_agent
                if hasattr(book_search_agent, 'book_memory'):
                    memory = book_search_agent.book_memory
                    if hasattr(memory, 'record_pick_action') and callable(memory.record_pick_action):
                        # If we have a position, record it
                        memory.record_pick_action(position)
                        print("Successfully recorded pick action in memory system")
            except Exception as e:
                print(f"Warning: Could not record pick in memory: {e}")
        
        # If successful, update memory system with the pick action
        if success and hasattr(visualize_semantic_forest, 'forest'):
            forest = visualize_semantic_forest.forest
            # Create a unique ID for the book if we don't already have one
            book_id = best_book.get('id', f"book_{int(time.time())}")
            
            # If we have a method to update the book state in the forest, use it
            if hasattr(forest, 'update_book_state') and callable(forest.update_book_state):
                forest.update_book_state(book_id, 'picked', position)
            
        # Reset picking flag
        if camera_controller:
            camera_controller.is_picking = False
            print("Reset camera controller picking mode")
            
        print("===== ENHANCED CAMERA-GUIDED PICK COMPLETE =====\n")
            
        return json.dumps({
            "status": "success" if success else "error",
            "message": "Camera-guided pick completed" if success else "Failed to pick object",
            "book": best_book,
            "position": position,
            "confidence": best_book.get('confidence', 0)
        })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        if camera_controller:
            camera_controller.is_picking = False
        return json.dumps({"status": "error", "message": f"Error in camera-guided pick: {str(e)}"})

# Tool registry for easy lookup
TOOL_REGISTRY = {
    "start_scan": start_scan,
    "move_camera": move_camera,
    "examine_area": examine_area,
    "analyze_view": analyze_view,
    "query_semantic_forest": query_semantic_forest,
    "add_spatial_context": add_spatial_context,
    "visualize_semantic_forest": visualize_semantic_forest,
    "pick_object": pick_object,
    "camera_guided_pick": camera_guided_pick
}