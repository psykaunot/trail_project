import os
import logging
import numpy as np
import trimesh
import collada
from PIL import Image
from urdf_parser_py.urdf import URDF

logging.basicConfig(level=logging.INFO)

# Override arm link poses to lay the arm out straight
ARM_POSE_OVERRIDES = {
    'locobot/arm_base_link':      {'rpy': [0, 0, 0]},
    'locobot/shoulder_link':      {'rpy': [0, 0, 0]},
    'locobot/upper_arm_link':     {'rpy': [0, 0, 0]},
    'locobot/upper_forearm_link': {'rpy': [0, 0, 0]},
    'locobot/lower_forearm_link': {'rpy': [0, 0, 0]},
    'locobot/wrist_link':         {'rpy': [0, 0, 0]},
}

# Create primitive shapes for URDF box/cylinder/sphere
from trimesh.creation import box as create_box, cylinder as create_cylinder, icosphere

def primitive_from_geometry(geom):
    if geom.box:
        size = np.array(geom.box.size, dtype=np.float64)
        return create_box(extents=size)
    elif geom.cylinder:
        radius = float(geom.cylinder.radius)
        length = float(geom.cylinder.length)
        # axis along Z
        return create_cylinder(radius=radius, height=length, sections=32)
    elif geom.sphere:
        radius = float(geom.sphere.radius)
        return icosphere(radius=radius)
    return None


def load_dae_with_texture(dae_path):
    try:
        coll_doc = collada.Collada(dae_path)
        scene = trimesh.Scene()
        for geom in coll_doc.scene.objects('geometry'):
            for prim in geom.primitives():
                mesh = trimesh.Trimesh(vertices=prim.vertex, faces=prim.vertex_index, process=False)
                # default grey fill
                if mesh.visual is None or (hasattr(mesh.visual,'vertex_colors') and len(mesh.visual.vertex_colors)==0):
                    grey = np.array([0.7,0.7,0.7])
                    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.tile(grey,(len(mesh.vertices),1)))
                scene.add_geometry(mesh)
        return scene
    except Exception as e:
        logging.error(f"DAE load error {dae_path}: {e}")
        return trimesh.load(dae_path, force='scene')


def apply_origin(geo, origin):
    xyz = origin.xyz if origin and origin.xyz else [0,0,0]
    rpy = origin.rpy if origin and origin.rpy else [0,0,0]
    mat = trimesh.transformations.compose_matrix(translate=xyz, angles=rpy)
    if isinstance(geo, trimesh.Scene):
        for g in geo.geometry.values():
            g.apply_transform(mat)
    else:
        geo.apply_transform(mat)
    return geo


def load_geometry(vis, base_dir, mats, loaded_links, scene, is_collision=False):
    geom = vis.geometry
    origin = vis.origin
    # Try filename mesh
    if geom.filename:
        mesh_path = geom.filename
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(base_dir, mesh_path)
        if os.path.exists(mesh_path):
            ext = os.path.splitext(mesh_path)[1].lower()
            try:
                if ext == '.dae':
                    loaded = load_dae_with_texture(mesh_path)
                else:
                    loaded = trimesh.load(mesh_path, force='scene') if ext != '.stl' else trimesh.load(mesh_path)
                # apply STL mats
                if ext == '.stl' and hasattr(vis,'material') and vis.material and vis.material.name in mats:
                    entry = mats[vis.material.name]
                    mesh_g = loaded.geometry[next(iter(loaded.geometry))] if isinstance(loaded,trimesh.Scene) else loaded
                    if 'color' in entry:
                        col = np.array(entry['color'][:3])
                        mesh_g.visual = trimesh.visual.ColorVisuals(mesh=mesh_g, vertex_colors=np.tile(col,(len(mesh_g.vertices),1)))
                        loaded = mesh_g
                loaded = apply_origin(loaded, origin)
                # add to scene
                if isinstance(loaded,trimesh.Scene):
                    for g in loaded.geometry.values():
                        scene.add_geometry(g, node_name=vis.parent)
                else:
                    scene.add_geometry(loaded, node_name=vis.parent)
                loaded_links.add(vis.parent)
                return True
            except Exception as e:
                logging.error(f"Error loading mesh {mesh_path}: {e}")
    # Try primitive
    prim_mesh = primitive_from_geometry(geom)
    if prim_mesh:
        prim_mesh.visual = trimesh.visual.ColorVisuals(mesh=prim_mesh, vertex_colors=np.tile([0.5,0.5,0.5],(len(prim_mesh.vertices),1)))
        prim_mesh = apply_origin(prim_mesh, origin)
        scene.add_geometry(prim_mesh, node_name=vis.parent)
        loaded_links.add(vis.parent)
        return True
    return False


def urdf_to_glb(urdf_file, output_glb):
    robot = URDF.from_xml_file(urdf_file)
    scene = trimesh.Scene()
    base_dir = os.path.dirname(os.path.abspath(urdf_file))
    # collect mats
    mats = {m.name:{'color':tuple(m.color.rgba) if m.color else None,'texture':m.texture.filename if m.texture else None} for m in robot.materials}
    loaded_links = set()
    # visuals
    for link in robot.links:
        for vis in link.visuals or []:
            vis.parent = link.name
            load_geometry(vis, base_dir, mats, loaded_links, scene)
    # collisions fallback
    for link in robot.links:
        if link.name in loaded_links: continue
        for col in link.collisions or []:
            col.parent = link.name
            if load_geometry(col, base_dir, mats, loaded_links, scene): break
    missing = set(l.name for l in robot.links) - loaded_links
    if missing:
        logging.warning(f"Missing: {missing}")
    scene.export(output_glb, file_type='glb', include_normals=True)
    logging.info(f"Exported GLB to {output_glb}")

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('urdf');p.add_argument('output')
    a=p.parse_args(); urdf_to_glb(a.urdf, a.output)
