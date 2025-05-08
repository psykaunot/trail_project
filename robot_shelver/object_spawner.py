#!/usr/bin/env python3
import os
import numpy as np
import magnum as mn
import habitat_sim

class ObjectSpawner:
    """Utility class for spawning objects in the Habitat simulator."""
    
    def __init__(self, sim):
        """Initialize the object spawner with a simulator instance."""
        self.sim = sim
        self.obj_template_mgr = sim.get_object_template_manager()
        self.rigid_obj_mgr = sim.get_rigid_object_manager()
        self.spawned_objects = {}
        
    def spawn_book(self, position=None, rotation=None):
        """Spawn a book object in the scene at the specified position."""
        # Check if book object already exists
        if "book" in self.spawned_objects:
            print("Book already exists in the scene")
            return self.spawned_objects["book"]

        try:
            # Load book template
            book_template_path = "data/objects/book"
            template_ids = self.obj_template_mgr.load_configs(book_template_path)

            if not template_ids:
                print("ERROR: Failed to load book template")
                return None

            # Add book to scene
            book_obj = self.rigid_obj_mgr.add_object_by_template_id(template_ids[0])

            if book_obj is None:
                print("ERROR: Failed to add book object to scene")
                return None

            # Determine position with improved visibility
            if position is None:
                # Use fixed position for book
                position = mn.Vector3(-0.02, 0.0, -2.0)
                print(f"Using fixed position for book: {position}")
            else:
                # Convert numpy array to Vector3 if needed
                if isinstance(position, np.ndarray):
                    position = mn.Vector3(position[0], 0.2, position[2])
                elif not isinstance(position, mn.Vector3):
                    position = mn.Vector3(-0.03, 0.0, -2.0)

            # Set book position and rotation
            state = book_obj.rigid_state
            state.translation = position
            if rotation is not None:
                state.rotation = rotation
            book_obj.rigid_state = state

            # Make book KINEMATIC to prevent falling through floor
            book_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC

            # Store reference to the spawned object
            self.spawned_objects["book"] = book_obj

            print(f"Book spawned successfully at {position}")
            return book_obj

        except Exception as e:
            print(f"ERROR spawning book: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def spawn_cube(self, size=0.1, position=None, color=None):
        """Spawn a simple cube in the scene."""
        try:
            # Create cube primitive
            prim_template_mgr = self.sim.get_asset_template_manager()
            cube_template = prim_template_mgr.get_default_cube_template()
            cube_template.scale = mn.Vector3(size, size, size)
            
            if color is not None:
                # Convert color to Magnum format if provided as RGB tuple
                if isinstance(color, (list, tuple)):
                    color = mn.Color4(color[0], color[1], color[2], 1.0)
                cube_template.primitive_color_texture = color
                
            # Register the template
            template_id = prim_template_mgr.register_template(cube_template, "cube")
            
            # Add cube to scene
            cube_obj = self.rigid_obj_mgr.add_object_by_template_id(template_id)
            
            if cube_obj is None:
                print("ERROR: Failed to add cube object to scene")
                return None
                
            # Determine position
            if position is None:
                # Default position
                position = mn.Vector3(0.0, 0.5, -1.0)
            
            # Set cube position
            state = cube_obj.rigid_state
            state.translation = position
            cube_obj.rigid_state = state
            
            # Generate unique handle
            cube_handle = f"cube_{len(self.spawned_objects)}"
            self.spawned_objects[cube_handle] = cube_obj
            
            print(f"Cube spawned successfully at {position}")
            return cube_obj
            
        except Exception as e:
            print(f"ERROR spawning cube: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def remove_object(self, name_or_object):
        """Remove an object from the scene by name or object reference."""
        try:
            # Handle object reference or name
            if isinstance(name_or_object, str):
                if name_or_object in self.spawned_objects:
                    obj = self.spawned_objects[name_or_object]
                else:
                    print(f"Object '{name_or_object}' not found in spawned objects")
                    return False
            else:
                obj = name_or_object
                
            # Remove from the simulator
            self.rigid_obj_mgr.remove_object_by_id(obj.object_id)
            
            # Remove from our tracking dict
            for name, tracked_obj in list(self.spawned_objects.items()):
                if tracked_obj.object_id == obj.object_id:
                    del self.spawned_objects[name]
                    break
                    
            print(f"Object removed successfully")
            return True
            
        except Exception as e:
            print(f"ERROR removing object: {e}")
            return False
            
    def get_spawned_object(self, name):
        """Get a spawned object by name."""
        return self.spawned_objects.get(name, None)