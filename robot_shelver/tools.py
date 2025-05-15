#!/usr/bin/env python3
import json
import inspect
from typing import Callable, Dict, Any
from dataclasses import dataclass


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
    tools = [start_scan, move_camera, examine_area, analyze_view]
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

# Tool registry for easy lookup
TOOL_REGISTRY = {
    "start_scan": start_scan,
    "move_camera": move_camera,
    "examine_area": examine_area,
    "analyze_view": analyze_view
}