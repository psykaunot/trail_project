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

# Utility to create a camera sensor spec
def make_cam(name, pos, ori):
    cam = habitat_sim.CameraSensorSpec()
    cam.uuid = name
    cam.sensor_type = habitat_sim.SensorType.COLOR
    cam.resolution = [480, 640]
    cam.position = mn.Vector3(*pos)
    cam.orientation = mn.Vector3(*ori)
    return cam


def main():
    # Paths & Simulator Setup
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    ds_cfg = os.path.join(
        data_dir, "scene_datasets", "hssd-hab",
        "hssd-hab.scene_dataset_config.json"
    )
    if not os.path.isfile(ds_cfg):
        print(f"ERROR: dataset config not found: {ds_cfg}")
        sys.exit(1)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = ds_cfg
    sim_cfg.scene_id = "102344049.scene_instance.json"
    sim_cfg.enable_physics = True
    sim_cfg.load_semantic_mesh = False
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = habitat_sim.gfx.DEFAULT_LIGHTING_KEY

    # Define Cameras
    top_cam = make_cam("top_rgb",(0.0, 0.0, 1.2),(0.0, 0.0, -np.pi / 2))
    
    robot_cam = make_cam("robot_rgb",(0.1, 0.0, 0.4),(-np.pi / 2, -np.pi, np.pi / 2))

    # Follow-from-behind camera: 1m behind, elevated, slight downward tilt
    back_cam = make_cam("back_rgb",(-1.0, 0.0, 0.8),(-np.pi/1.5, np.pi, np.pi / 2))

    # Create agent with all cameras
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [top_cam, robot_cam, back_cam]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    agent = sim.get_agent(0)

    # Load LoCoBot URDF
    urdf_fp = os.path.join(data_dir, "robots", "locobot_wx250s.urdf")
    if not os.path.isfile(urdf_fp):
        print(f"ERROR: URDF not found: {urdf_fp}")
        sys.exit(1)
    ao_mgr = sim.get_articulated_object_manager()
    locobot = ao_mgr.add_articulated_object_from_urdf(
        urdf_fp,
        fixed_base=False,
        global_scale=1.0,
        mass_scale=1.0,
        force_reload=False,
        maintain_link_order=False,
        intertia_from_urdf=False,
        light_setup_key=habitat_sim.gfx.DEFAULT_LIGHTING_KEY,
    )
    if locobot is None:
        print("ERROR: failed to instantiate LoCoBot")
        sys.exit(1)

    # Initial Robot Pose
    state = locobot.rigid_state
    state.translation = mn.Vector3(1.0, 0.0, -2.0)
    upright_q = mn.Quaternion.rotation(Rad(-np.pi / 2), mn.Vector3(1, 0, 0))
    yaw_q = mn.Quaternion.rotation(Rad(np.pi), mn.Vector3(0, 1, 0))
    state.rotation = yaw_q * upright_q
    locobot.rigid_state = state
    sim.step_physics(0.0)

    # Find the camera link on the robot
    link_map = {locobot.get_link_joint_name(lid): lid for lid in locobot.get_link_ids()}
    cam_name = "camera"
    if cam_name not in link_map:
        print(f"ERROR: link '{cam_name}' not found on LoCoBot")
        sys.exit(1)
    cam_lid = link_map[cam_name]
    cam_link_node = locobot.get_link_scene_node(cam_lid)
    print(f"Using camera link: {cam_name} (ID {cam_lid})")

    # Joint Motors Setup
    dof_map = {locobot.get_link_joint_name(lid): lid
               for lid in locobot.get_link_ids()
               if locobot.get_link_num_dofs(lid) > 0}
    motor_ids = {}
    motor_settings = {}
    def mk_motor(name, impulse):
        lid = dof_map[name]
        s = phys.JointMotorSettings(
            position_target=0.0,
            position_gain=0.0,
            velocity_target=0.0,
            velocity_gain=1.0,
            max_impulse=impulse
        )
        mid = locobot.create_joint_motor(lid, s)
        motor_ids[lid] = mid
        motor_settings[lid] = s
    for nm, imp in [
        ("wheel_left_joint", 300.0), ("wheel_right_joint", 300.0),
        ("waist", 150.0), ("shoulder", 150.0), ("elbow", 150.0),
        ("forearm_roll", 150.0), ("wrist_angle", 150.0),
        ("wrist_rotate", 150.0), ("gripper", 30.0)
    ]:
        mk_motor(nm, imp)

    # Combined View Window
    cv2.namedWindow("Combined View", cv2.WINDOW_NORMAL)
    print("Controls: arrows=drive, U/J/I/K/O/L/P/;/[/]/]/N/M=arm, G/H=gripper, ESC=quit")

    DRIVE_SPEED = 20.0
    TURN_SPEED = 10.0
    ARM_SPEED = 3.0
    GRIP_SPEED = 1.5
    dt = 1.0 / 30.0

    while True:
        key = cv2.waitKeyEx(1)
        if key == 27:  # ESC
            break

        # Reset motors
        for s in motor_settings.values():
            s.velocity_target = 0.0

        # Drive controls
        if key == 65362:  # Up arrow
            motor_settings[dof_map["wheel_left_joint"]].velocity_target = DRIVE_SPEED
            motor_settings[dof_map["wheel_right_joint"]].velocity_target = DRIVE_SPEED
        elif key == 65364:  # Down arrow
            motor_settings[dof_map["wheel_left_joint"]].velocity_target = -DRIVE_SPEED
            motor_settings[dof_map["wheel_right_joint"]].velocity_target = -DRIVE_SPEED
        elif key in (65361, ord('q')):  # Left arrow or Q
            motor_settings[dof_map["wheel_left_joint"]].velocity_target = -TURN_SPEED
            motor_settings[dof_map["wheel_right_joint"]].velocity_target = TURN_SPEED
        elif key in (65363, ord('w')):  # Right arrow or W
            motor_settings[dof_map["wheel_left_joint"]].velocity_target = TURN_SPEED
            motor_settings[dof_map["wheel_right_joint"]].velocity_target = -TURN_SPEED

        # Arm & gripper controls
        arm_map = {
            ord('u'): ("waist", ARM_SPEED), ord('j'): ("waist", -ARM_SPEED),
            ord('i'): ("shoulder", ARM_SPEED), ord('k'): ("shoulder", -ARM_SPEED),
            ord('o'): ("elbow", ARM_SPEED), ord('l'): ("elbow", -ARM_SPEED),
            ord('p'): ("forearm_roll", ARM_SPEED), ord(';'): ("forearm_roll", -ARM_SPEED),
            ord('['): ("wrist_angle", ARM_SPEED), ord(']'): ("wrist_angle", -ARM_SPEED),
            ord('n'): ("wrist_rotate", ARM_SPEED), ord('m'): ("wrist_rotate", -ARM_SPEED)
        }
        if key in arm_map:
            nm, vel = arm_map[key]
            motor_settings[dof_map[nm]].velocity_target = vel
        if key == ord('g'):
            motor_settings[dof_map['gripper']].velocity_target = GRIP_SPEED
        elif key == ord('h'):
            motor_settings[dof_map['gripper']].velocity_target = -GRIP_SPEED

        # Step physics and sync agent
        for lid, mid in motor_ids.items():
            locobot.update_joint_motor(mid, motor_settings[lid])
        for _ in range(5):
            sim.step_physics(dt / 5.0)
        rs = locobot.rigid_state
        agent.scene_node.translation = rs.translation
        agent.scene_node.rotation = rs.rotation

        # Get images and label
        obs = sim.get_sensor_observations()
        top_img = obs['top_rgb']
        rob_img = obs['robot_rgb']
        back_img = obs['back_rgb']
        cv2.putText(top_img, 'TOP', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(rob_img, 'FRONT', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(back_img, 'BACK', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Combine & display
        combined = np.hstack((top_img, rob_img, back_img))
        cv2.imshow('Combined View', combined)

    cv2.destroyAllWindows()
    sim.close()


if __name__ == '__main__':
    main()
