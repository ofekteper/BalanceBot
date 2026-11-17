import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from ballplate_model import build_xml, ik, joint_gaps, plate_tilt

BALL = "pingpong"   # "pingpong", "steel_20mm", "plastic_20mm", or None for no ball
DEMO = "flat"       # "flat", "ramp", "circle", "manual"
#   flat   : plate level (the ball just sits there)
#   ramp   : smooth tilt about X, 0 -> 8 deg -> 0 over 16 s. Use BALL=None: with a ball it rolls off
#   circle : tilt direction rotates (use BALL=None to watch the legs move)
#   manual : the script sends nothing; drag the servo sliders in the viewer's "Control" panel
#            (values are servo angles in radians, 0 = neutral)

model = mujoco.MjModel.from_xml_string(build_xml(ball=BALL))
data = mujoco.MjData(model)


def plate_command(t):
    """Return (z, rx, ry) of the plate we want at time t."""
    if DEMO == "flat":
        return 0.0, 0.0, 0.0
    if DEMO == "ramp":
        a = math.radians(8)
        return 0.0, a * 0.5 * (1 - math.cos(2 * math.pi * t / 16)), 0.0
    if DEMO == "circle":
        a = math.radians(5)
        return 0.0, a * math.sin(2 * t), a * math.cos(2 * t)
    return None


def ball_in_plate_frame():
    if BALL is None:
        return np.zeros(3)
    plate, ball = data.body("plate"), data.body("ball")
    R = plate.xmat.reshape(3, 3)
    return R.T @ (ball.xpos - plate.xpos)


FRAME_DT = 0.016                            # ~60 fps
steps_per_frame = round(FRAME_DT / model.opt.timestep)
next_print = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()

        cmd = plate_command(data.time)
        if cmd is not None:
            try:
                data.ctrl[:] = ik(*cmd)
            except ValueError as e:
                print("IK:", e)

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        viewer.sync()

        if data.time >= next_print:
            next_print += 0.5
            bx, by, _ = ball_in_plate_frame()
            rx, ry = plate_tilt(data)
            cmd_deg = [math.degrees(c) for c in data.ctrl]
            act_deg = [math.degrees(data.joint(f"servo{i}").qpos[0]) for i in range(3)]
            print(f"t={data.time:5.2f}s  ball x={bx*1000:6.1f} y={by*1000:6.1f} mm | "
                  f"plate tilt rx={math.degrees(rx):5.1f} ry={math.degrees(ry):5.1f} deg | "
                  f"servo cmd {[round(c, 1) for c in cmd_deg]} act {[round(a, 1) for a in act_deg]} | "
                  f"joint gap max {max(joint_gaps(data)) * 1000:.2f} mm")

        time.sleep(max(0.0, FRAME_DT - (time.time() - t0)))
