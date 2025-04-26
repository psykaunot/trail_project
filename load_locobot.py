#!/usr/bin/env python3
import os
import sys
import numpy as np
import magnum as mn
from magnum import Rad
import cv2

import habitat_sim
import habitat_sim.gfx
import habitat_sim.physics as phys
from habitat_sim._ext.habitat_sim_bindings import RigidState


def main():
    # Paths
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")

    # ─── Load HSSD-HAB scene via dataset config ────────────────────────────────
    ds_cfg = os.path.join(
        data_dir, "scene_datasets", "hssd-hab",
        "hssd-hab.scene_dataset_config.json"
    )
    if not os.path.isfile(ds_cfg):
        print(f"ERROR: dataset config not found: {ds_cfg}")
        sys.exit(1)

    # Simulator config
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file     = ds_cfg
    sim_cfg.scene_id                      = "102344049.scene_instance.json"
    sim_cfg.enable_physics                = True
    sim_cfg.load_semantic_mesh            = False
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup             = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

    # ─── Agent & Cameras ──────────────────────────────────────────────────────
    def make_cam(name, pos, ori):
        cam = habitat_sim.CameraSensorSpec()
        cam.uuid        = name
        cam.sensor_type = habitat_sim.SensorType.COLOR
        cam.resolution  = [480, 640]
        cam.position    = mn.Vector3(*pos)
        cam.orientation = mn.Vector3(*ori)
        return cam
    
    # Front camera - dynamically attached to robot's camera link
    CAM_POS = (0.0, 0.0, 0.0)  # Will be updated to actual link position
    
    CHASE_POS = (1.0, 0.0, 1.0)  # 1.5m behind robot
    CHASE_PITCH = -0.25  # Look down
    
    TOP_POS = (0.0, 0.0, 1.2)  # 1.2m above

    YAW_180 = np.pi
    
    cams = [
        make_cam("front_rgb", CAM_POS, (0.0, 0.0, 0.0)),  
        make_cam("top_rgb", TOP_POS, (0.0, 0.0, -YAW_180/2)),  
        make_cam("chase_rgb", CHASE_POS, (YAW_180/2, 0.0, -YAW_180)),  
    ]
    
    agent_cfg = habitat_sim.agent.AgentConfiguration() 
    agent_cfg.sensor_specifications = cams

    # Create simulator and agent
    sim   = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    agent = sim.get_agent(0)

    # ─── Load LoCoBot URDF ───────────────────────────────────────────────────
    urdf_fp = os.path.join(data_dir, "robots", "locobot_wx250s.urdf")
    if not os.path.isfile(urdf_fp):
        print(f"ERROR: URDF not found: {urdf_fp}")
        sys.exit(1)

    ao_mgr  = sim.get_articulated_object_manager()
    locobot = ao_mgr.add_articulated_object_from_urdf(
        urdf_fp,
        True,   # fixed_base
        1.0,    # global_scale
        1.0,    # mass_scale
        False,  # force_reload
        False,  # maintain_link_order
        False,  # inertia_from_urdf
        habitat_sim.gfx.DEFAULT_LIGHTING_KEY
    )
    if locobot is None:
        print("ERROR: failed to instantiate LoCoBot")
        sys.exit(1)
    
    # ─── Initial robot pose ──────────────────────────────────────────────────
    state = locobot.rigid_state
    state.translation = mn.Vector3(1.0, 0.2, 0.8)
    upright_q = mn.Quaternion.rotation(Rad(-np.pi/2), mn.Vector3(1,0,0))
    yaw_q     = mn.Quaternion.rotation(Rad(np.pi), mn.Vector3(0,1,0))
    state.rotation = yaw_q * upright_q
    locobot.rigid_state = state
    
    # Update physics to ensure all links are properly positioned
    sim.step_physics(0.0)
    
    # Initialize camera position
    camera_link_id = -1
    for lid in locobot.get_link_ids():
        if locobot.get_link_name(lid) == "camera_locobot_link":
            camera_link_id = lid
            break
    
    if camera_link_id >= 0:
        # Get the camera link transform
        camera_transform = locobot.get_link_scene_node(camera_link_id).absolute_transformation()
        
        # Identity matrix works for our setup
        urdf_to_camera = mn.Matrix4()
        final_transform = camera_transform @ urdf_to_camera
        
        # Set initial front camera position
        front_sensor = agent._sensors["front_rgb"]
        front_sensor.node.translation = final_transform.translation
        front_sensor.node.rotation = mn.Quaternion.from_matrix(final_transform.rotation())
        
        print(f"Initial camera position: {final_transform.translation}")
        print(f"Initial camera rotation: {mn.Quaternion.from_matrix(final_transform.rotation())}")
    else:
        print("Warning: Could not find camera_locobot_link during initialization!")

    # ─── Joint Motors Setup ─────────────────────────────────────────────────
    link_map = {
        locobot.get_link_joint_name(lid): lid
        for lid in locobot.get_link_ids()
        if locobot.get_link_num_dofs(lid) > 0
    }
    motor_ids      = {}
    motor_settings = {}

    def mk_motor(name, impulse):
        lid = link_map[name]
        s   = phys.JointMotorSettings(
            position_target=0.0,
            position_gain  = 0.0,
            velocity_target=0.0,
            velocity_gain  = 1.0,
            max_impulse    = impulse
        )
        mid = locobot.create_joint_motor(lid, s)
        motor_ids[lid]      = mid
        motor_settings[lid] = s

    # wheels
    mk_motor("wheel_left_joint",  100.0)
    mk_motor("wheel_right_joint", 100.0)
    # arms
    for nm in ["waist","shoulder","elbow","forearm_roll","wrist_angle","wrist_rotate"]:
        mk_motor(nm, 50.0)
    # gripper
    mk_motor("gripper", 10.0)

    # ─── Load Kinematic Book ─────────────────────────────────────────────────
    tpl_mgr   = sim.get_object_template_manager()
    rigid_mgr = sim.get_rigid_object_manager()
    book_cfg  = os.path.join(data_dir, "objects", "book", "book.object_config.json")
    book_ids  = tpl_mgr.load_configs(book_cfg)
    book      = rigid_mgr.add_object_by_template_id(book_ids[0], None)
    book.motion_type = phys.MotionType.KINEMATIC
    book.translation = state.translation + mn.Vector3(0.5, 0.0, -0.2)

    book_vel = phys.VelocityControl()
    book_vel.controlling_lin_vel  = True
    book_vel.controlling_ang_vel  = True
    book_vel.lin_vel_is_local     = True
    book_vel.ang_vel_is_local     = True

    # ─── Teleop Loop ─────────────────────────────────────────────────────────
    DRIVE_SPEED, TURN_SPEED = 5.0, 3.0
    ARM_SPEED,   GRIP_SPEED = 1.0, 0.5
    LIN_B,       ANG_B      = 0.5, 1.0

    arm_map = {
        ord('u'): ("waist", +ARM_SPEED), ord('j'): ("waist", -ARM_SPEED),
        ord('i'): ("shoulder", +ARM_SPEED), ord('k'): ("shoulder", -ARM_SPEED),
        ord('o'): ("elbow", +ARM_SPEED),   ord('l'): ("elbow", -ARM_SPEED),
        ord('p'): ("forearm_roll", +ARM_SPEED), ord(';'): ("forearm_roll", -ARM_SPEED),
        ord('['): ("wrist_angle", +ARM_SPEED),   ord(']'): ("wrist_angle", -ARM_SPEED),
        ord('n'): ("wrist_rotate", +ARM_SPEED),  ord('m'): ("wrist_rotate", -ARM_SPEED),
    }

    win = "Habitat Views"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("""
Controls:
  Arrow keys: drive base
  Q/W       : spin base
  U/J...N/M : arm joints +/−
  G/H       : gripper open/close
  SPACE     : stop all
  ESC       : quit
"""
    )

    dt = 1.0 / 60.0
    while True:
        key = cv2.waitKeyEx(1)
        if key == 27:  # ESC
            break

        # Reset all motor targets
        for s in motor_settings.values():
            s.velocity_target = 0.0

        # Base movement
        if   key == 65362:  # UP
            motor_settings[link_map["wheel_left_joint"]].velocity_target  = DRIVE_SPEED
            motor_settings[link_map["wheel_right_joint"]].velocity_target = DRIVE_SPEED
        elif key == 65364:  # DOWN
            motor_settings[link_map["wheel_left_joint"]].velocity_target  = -DRIVE_SPEED
            motor_settings[link_map["wheel_right_joint"]].velocity_target = -DRIVE_SPEED
        elif key in (65361, ord('q')):  # LEFT
            motor_settings[link_map["wheel_left_joint"]].velocity_target  = -TURN_SPEED
            motor_settings[link_map["wheel_right_joint"]].velocity_target =  TURN_SPEED
        elif key in (65363, ord('w')):  # RIGHT
            motor_settings[link_map["wheel_left_joint"]].velocity_target  =  TURN_SPEED
            motor_settings[link_map["wheel_right_joint"]].velocity_target = -TURN_SPEED

        # Arm & gripper
        if key in arm_map:
            nm, vel = arm_map[key]
            motor_settings[link_map[nm]].velocity_target = vel
        if key == ord('g'):
            motor_settings[link_map["gripper"]].velocity_target = +GRIP_SPEED
        elif key == ord('h'):
            motor_settings[link_map["gripper"]].velocity_target = -GRIP_SPEED

        # Apply motors and step physics
        for lid, mid in motor_ids.items():
            locobot.update_joint_motor(mid, motor_settings[lid])
        sim.step_physics(dt)

        # Sync agent's scene node to the robot's pose
        rs = locobot.rigid_state
        agent.scene_node.translation = rs.translation
        agent.scene_node.rotation    = rs.rotation

        # ─── Overlay camera name & robot pose ─────────────────────────────────
        obs   = sim.get_sensor_observations()
        front, top, chase = obs["front_rgb"], obs["top_rgb"], obs["chase_rgb"]
        pose_str = f"pos={rs.translation}  rot={rs.rotation}"
        for img, name in [(front, "front_rgb"), (top, "top_rgb"), (chase, "chase_rgb")]:
            # Camera name
            cv2.putText(img, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            # Pose
            cv2.putText(img, pose_str, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # Build and show the mosaic
        mosaic = np.hstack((front, top, chase))
        cv2.imshow(win, mosaic)

    cv2.destroyAllWindows()
    sim.close()


if __name__ == "__main__":
    main()