"""Interactive BalanceBot simulation: MuJoCo viewer + a control panel window.

    python run_sim.py

Control panel modes
  Plate tilt : sliders for tilt about X / Y (0.1 deg steps) and height; inverse kinematics
               computes the three servo angles for you.
  Servos     : drive each servo directly, in degrees from neutral (0.1 deg steps).
  Circle demo: the tilt direction rotates slowly (amplitude = the "rx" slider).

All commands go through ServoCommander, which limits the speed like a real hobby servo.
The panel also shows the load on each magnetic joint: if it goes into tension beyond
MAGNET_PULL_N, a real magnet would let the ball go, and the panel turns red.
"""
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from ballplate_model import (ARM_LIMIT, MAGNET_PULL_N, ServoCommander, build_xml, joint_gaps,
                             joint_loads, plate_tilt)

BALL = "pingpong"        # "pingpong", "steel_20mm", "plastic_20mm", or None for no ball
SERVO_RESOLUTION = 0.0   # deg; try 0.3 to see what a cheap servo's deadband does to small tilts
MAX_TILT_DEG = 10.0      # range of the tilt sliders
FRAME_DT = 0.016         # ~60 fps

model = mujoco.MjModel.from_xml_string(build_xml(ball=BALL))
data = mujoco.MjData(model)
servos = ServoCommander(resolution_deg=SERVO_RESOLUTION)
steps_per_frame = max(1, round(FRAME_DT / model.opt.timestep))


def ball_in_plate_frame():
    if BALL is None:
        return np.zeros(3)
    plate, ball = data.body("plate"), data.body("ball")
    return plate.xmat.reshape(3, 3).T @ (ball.xpos - plate.xpos)


# ------------------------------------------------------------------ control panel (tkinter)
import tkinter as tk

root = tk.Tk()
root.title("BalanceBot control")
mode = tk.StringVar(value="tilt")
v_rx, v_ry, v_z = tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)
v_servo = [tk.DoubleVar(value=0.0) for _ in range(3)]
status = tk.StringVar()
paused = tk.BooleanVar(value=False)


def slider(parent, text, var, lo, hi, row):
    tk.Label(parent, text=text, width=14, anchor="w").grid(row=row, column=0, sticky="w")
    tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=0.1, orient="horizontal",
             length=320).grid(row=row, column=1)


def on_mode_change():
    if mode.get() == "servo":   # start the servo sliders where the servos are now: no jump
        for i in range(3):
            v_servo[i].set(round(math.degrees(servos.target[i]), 1))


def level():
    for v in (v_rx, v_ry, v_z, *v_servo):
        v.set(0.0)


def reset():
    mujoco.mj_resetData(model, data)
    servos.output[:] = 0.0
    level()


f_mode = tk.LabelFrame(root, text="Mode", padx=6, pady=4)
f_mode.pack(fill="x", padx=6, pady=4)
for val, txt in (("tilt", "Plate tilt (IK)"), ("servo", "Servos directly"), ("circle", "Circle demo")):
    tk.Radiobutton(f_mode, text=txt, variable=mode, value=val,
                   command=on_mode_change).pack(side="left")

f_tilt = tk.LabelFrame(root, text="Plate (deg / mm)", padx=6, pady=4)
f_tilt.pack(fill="x", padx=6, pady=4)
slider(f_tilt, "tilt about X  rx", v_rx, -MAX_TILT_DEG, MAX_TILT_DEG, 0)
slider(f_tilt, "tilt about Y  ry", v_ry, -MAX_TILT_DEG, MAX_TILT_DEG, 1)
slider(f_tilt, "height  z", v_z, -8, 8, 2)

f_srv = tk.LabelFrame(root, text="Servo angles (deg from neutral)", padx=6, pady=4)
f_srv.pack(fill="x", padx=6, pady=4)
lim = round(math.degrees(ARM_LIMIT))
for i in range(3):
    slider(f_srv, f"servo {i}", v_servo[i], -lim, lim, i)

f_btn = tk.Frame(root)
f_btn.pack(fill="x", padx=6, pady=4)
tk.Button(f_btn, text="Level (all 0)", command=level).pack(side="left")
tk.Button(f_btn, text="Reset simulation", command=reset).pack(side="left", padx=6)
tk.Checkbutton(f_btn, text="Pause", variable=paused).pack(side="left")

status_label = tk.Label(root, textvariable=status, justify="left", font=("Consolas", 9), anchor="w")
status_label.pack(fill="x", padx=6, pady=4)

closed = False


def on_close():
    global closed
    closed = True


root.protocol("WM_DELETE_WINDOW", on_close)


# ------------------------------------------------------------------ main loop
def apply_panel():
    m = mode.get()
    if m == "tilt":
        servos.set_tilt_deg(v_rx.get(), v_ry.get(), v_z.get())
    elif m == "servo":
        servos.set_servos_deg([v.get() for v in v_servo])
    elif m == "circle":
        a = v_rx.get() or 4.0
        servos.set_tilt_deg(a * math.sin(1.0 * data.time), a * math.cos(1.0 * data.time), v_z.get())


frame = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and not closed:
        t0 = time.time()
        apply_panel()
        if not paused.get():
            with viewer.lock():
                for _ in range(steps_per_frame):
                    servos.update(data, model.opt.timestep)
                    mujoco.mj_step(model, data)
        viewer.sync()

        frame += 1
        if frame % 6 == 0:   # refresh the readout ~10x per second
            rx, ry = plate_tilt(data)
            act = [math.degrees(data.joint(f"servo{i}").qpos[0]) for i in range(3)]
            loads = joint_loads(model, data)
            bx, by, _ = ball_in_plate_frame()
            pulled = min(loads) < -MAGNET_PULL_N
            lines = [
                f"t = {data.time:6.2f} s",
                "servo target  " + "  ".join(f"{math.degrees(a):6.1f}" for a in servos.target) + "  deg",
                "servo actual  " + "  ".join(f"{a:6.1f}" for a in act) + "  deg",
                f"plate tilt    rx {math.degrees(rx):5.2f}  ry {math.degrees(ry):5.2f}  deg",
                "joint gap     " + "  ".join(f"{g * 1000:6.3f}" for g in joint_gaps(data)) + "  mm",
                "joint load    " + "  ".join(f"{f:6.2f}" for f in loads)
                + f"  N  (+ push, - pull; magnet lets go below -{MAGNET_PULL_N} N)",
            ]
            if BALL is not None:
                lines.append(f"ball on plate x {bx * 1000:6.1f}  y {by * 1000:6.1f}  mm")
            if servos.last_error:
                lines.append("IK: " + servos.last_error + "  (keeping last reachable pose)")
            if pulled:
                lines.append("!! a magnetic joint is being pulled harder than MAGNET_PULL_N")
            status.set("\n".join(lines))
            status_label.config(fg="red" if pulled or servos.last_error else "black")
        root.update()

        time.sleep(max(0.0, FRAME_DT - (time.time() - t0)))

root.destroy()
