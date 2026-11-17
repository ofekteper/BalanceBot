import time
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("ball_plate.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        # Step 4a: plate flat (ctrl = 0). Later we will change this.
        data.ctrl[0] = 0   # tilt about X
        data.ctrl[1] = 0   # tilt about Y

        mujoco.mj_step(model, data)
        viewer.sync()

        # keep real-time pace
        time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))