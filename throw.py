import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# Left arm actuators (tossing hand)
l_pitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pitch_joint")
l_roll  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_roll_joint")
l_yaw   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_yaw_joint")
l_elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_elbow_joint")
l_wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wrist_roll_joint")

# Right arm actuators (catching hand)
r_pitch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pitch_joint")
r_roll  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_roll_joint")
r_yaw   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_yaw_joint")
r_elbow = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_elbow_joint")
r_wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wrist_roll_joint")

# Left and right hand weld IDs
left_weld  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "left_hand_weld")
right_weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "right_hand_weld")

tossed = False
caught = False
prev_t = 0.0
wall_start = time.perf_counter()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Detect Reset / Backspace and re-sync the real-time clock
        if data.time < prev_t:
            tossed = False
            caught = False
            wall_start = time.perf_counter() - data.time
        prev_t = data.time

        # Sync simulation clock 1:1 with real wall-clock seconds
        target_sim_time = time.perf_counter() - wall_start
        if target_sim_time - data.time > 0.05:
            wall_start = time.perf_counter() - data.time
            target_sim_time = data.time

        while data.time < target_sim_time:
            t = data.time

            # PHASE 1 (0.0s - 1.0s): Ready stance
            if t < 1.0:
                data.ctrl[l_pitch] = -0.10
                data.ctrl[l_roll]  =  0.35
                data.ctrl[l_yaw]   =  0.30
                data.ctrl[l_elbow] =  0.65
                data.ctrl[l_wrist] = -1.45

                data.ctrl[r_pitch] = -0.38
                data.ctrl[r_roll]  = -0.16
                data.ctrl[r_yaw]   =  0.00
                data.ctrl[r_elbow] =  0.28
                data.ctrl[r_wrist] =  1.45

            # PHASE 2 (1.0s - 1.15s): Lift left forearm upward and inward to toss
            elif 1.0 <= t < 1.15:
                data.ctrl[l_pitch] = -0.55
                data.ctrl[l_roll]  =  0.12
                data.ctrl[l_yaw]   = -0.17
                data.ctrl[l_elbow] = -0.15

            # PHASE 3 (At 1.15s): Release mid-lift and return left arm smoothly
            elif t >= 1.15:
                if not tossed:
                    data.eq_active[left_weld] = 0
                    tossed = True
                    print(f"[{t:.2f}s] Tossed apple from left palm!")

                data.ctrl[l_pitch] = -0.25
                data.ctrl[l_roll]  =  0.30
                data.ctrl[l_yaw]   =  0.20
                data.ctrl[l_elbow] =  0.40

            # PHASE 4 (After 1.40s): Catch right as the apple lands on the open right palm
            if tossed and not caught and t >= 1.40:
                apple_pos = data.body("apple").xpos
                r_wrist_pos = data.body("right_wrist_yaw_link").xpos

                if np.linalg.norm(apple_pos - r_wrist_pos) < 0.15 and apple_pos[2] <= r_wrist_pos[2] + 0.08:
                    data.eq_active[right_weld] = 1
                    caught = True
                    data.ctrl[r_pitch] = -0.28
                    data.ctrl[r_elbow] =  0.48
                    print(f"[{t:.2f}s] Clean catch in right palm!")

            mujoco.mj_step(model, data)

        viewer.sync()
        time.sleep(0.001)