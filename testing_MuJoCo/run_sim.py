import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from ballplate_model import build_xml, ik

BALL = "pingpong"   # "pingpong", "steel_20mm", "plastic_20mm"
DEMO = "flat"       # "flat", "tilt", "circle"

model = mujoco.MjModel.from_xml_string(build_xml(ball=BALL))
data = mujoco.MjData(model)


def plate_command(t):
    """Return (z, rx, ry) of the plate we want at time t."""
    if DEMO == "flat":
        return 0.0, 0.0, 0.0
    if DEMO == "tilt":                      # alternate 5 deg tilt about X and Y
        a = math.radians(5)
        return 0.0, (a if int(t) % 4 < 2 else -a), 0.0
    if DEMO == "circle":                    # tilt direction rotates -> ball circles
        a = math.radians(5)
        return 0.0, a * math.sin(2 * t), a * math.cos(2 * t)
    raise ValueError(DEMO)


def ball_in_plate_frame():
    plate, ball = data.body("plate"), data.body("ball")
    R = plate.xmat.reshape(3, 3)
    return R.T @ (ball.xpos - plate.xpos)


FRAME_DT = 0.016                            # ~60 fps
steps_per_frame = round(FRAME_DT / model.opt.timestep)
next_print = 0.0

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t0 = time.time()

        z, rx, ry = plate_command(data.time)
        try:
            data.ctrl[:] = ik(z, rx, ry)
        except ValueError as e:
            print("IK:", e)

        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        viewer.sync()

        if data.time >= next_print:
            next_print += 0.5
            bx, by, bz = ball_in_plate_frame()
            print(f"t={data.time:5.2f}s  ball(plate frame) x={bx*1000:6.1f} y={by*1000:6.1f} mm  "
                  f"plate rx={math.degrees(data.joint('plate_rx').qpos[0]):5.1f} deg "
                  f"ry={math.degrees(data.joint('plate_ry').qpos[0]):5.1f} deg "
                  f"z={data.joint('plate_z').qpos[0]*1000:5.1f} mm")

        time.sleep(max(0.0, FRAME_DT - (time.time() - t0)))
