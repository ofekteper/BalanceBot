"""Interactive BalanceBot simulation: MuJoCo viewer + a control panel window.

    python run_sim.py

Control panel modes
  Balance    : the BaBot firmware (babot_firmware.py, ported from main.ino) balances the ball:
               IR sensors -> weighted centre -> PID -> movePlatform IK -> servos, at 30 Hz.
               PID gains, setpoint and ON/OFF can be changed live; "Kick ball" pushes the ball.
  Plate tilt : sliders for tilt about X / Y (0.1 deg steps) and height; inverse kinematics
               computes the three servo angles for you.
  Servos     : drive each servo directly, in degrees from neutral (0.1 deg steps).
  Circle demo: the tilt direction rotates slowly (amplitude = the "rx" slider).

All commands go through ServoCommander, which limits the speed like a real hobby servo.

IR sensing PCB (read at 30 Hz, like the firmware):
  * in the viewer the 16 phototransistors light up with their reading, and a green disc on the
    plate shows where the firmware thinks the ball is
  * the panel shows the 4x4 signals (ADC counts) seen from above, servo 0 at the top, with the
    true ball (orange) and the estimate (green)
"""
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from babot_firmware import FirmwareLoop
from ballplate_model import (MAGNET_PULL_N, PLATE_THICK, PLATE_Z_REL, SERVO_MAX, SERVO_MIN,
                             ServoCommander, build_xml, joint_gaps, joint_loads, plate_tilt)
from ir_sensors import ADC_REF_SIGNAL, PT_PITCH, PT_XY, IRSensorArray

BALL = "pingpong"        # "pingpong" (what the firmware is tuned for), "steel_20mm", "plastic_20mm", None
START_MODE = "balance"   # "balance", "tilt", "servo", "circle"
BALL_START = (0.03, 0.02)  # m, where the ball is dropped (plate frame)
SERVO_RESOLUTION = 0.0   # deg; extra servo deadband on top of the firmware's whole degrees
MAX_TILT_DEG = 10.0      # range of the tilt sliders
FRAME_DT = 0.016         # ~60 fps

model = mujoco.MjModel.from_xml_string(build_xml(ball=BALL, ball_xy=BALL_START))
data = mujoco.MjData(model)
servos = ServoCommander(resolution_deg=SERVO_RESOLUTION)
sensors = IRSensorArray(BALL)
loop = FirmwareLoop(sensors, servos)
fw = loop.fw
steps_per_frame = max(1, round(FRAME_DT / model.opt.timestep))

pt_geoms = [model.geom(f"pt{k}").id for k in range(16)]
est_mocap = model.body("ball_estimate").mocapid[0] if BALL is not None else -1
ball_dof = model.jnt_dofadr[model.joint("ball_free").id] if BALL is not None else -1


def ball_in_plate_frame():
    if BALL is None:
        return np.zeros(3)
    return sensors.ball_in_plate_frame(data)


def heat(value):
    """0..1 -> dark -> red -> yellow."""
    v = min(max(value, 0.0), 1.0)
    return (0.1 + 0.9 * min(1.0, 2 * v), 0.1 + 0.9 * max(0.0, 2 * v - 1), 0.1)


def update_markers():
    """After each 30 Hz sample: colour the phototransistors, move the estimate disc."""
    lvl = (sensors.signal - sensors.baseline) / ADC_REF_SIGNAL
    for k, g in enumerate(pt_geoms):
        model.geom_rgba[g, :3] = heat(lvl[k])
    if est_mocap >= 0:
        plate = data.body("plate")
        if loop.estimate is None:
            data.mocap_pos[est_mocap] = (0, 0, -1)          # hide it under the floor
        else:
            local = np.array([*loop.estimate, PLATE_Z_REL + PLATE_THICK / 2 + 0.0007])
            data.mocap_pos[est_mocap] = plate.xpos + plate.xmat.reshape(3, 3) @ local
            data.mocap_quat[est_mocap] = plate.xquat


# ------------------------------------------------------------------ control panel (tkinter)
import tkinter as tk

root = tk.Tk()
root.title("BalanceBot control")
mode = tk.StringVar(value=START_MODE)
v_rx, v_ry, v_z = tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)
v_servo = [tk.DoubleVar(value=0.0) for _ in range(3)]
v_P, v_I, v_D = tk.DoubleVar(value=fw.P), tk.DoubleVar(value=fw.I), tk.DoubleVar(value=fw.D)
v_spx, v_spy = tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)
v_fw_on = tk.BooleanVar(value=True)
status = tk.StringVar()
paused = tk.BooleanVar(value=False)


def slider(parent, text, var, lo, hi, row, res=0.1, length=320):
    tk.Label(parent, text=text, width=14, anchor="w").grid(row=row, column=0, sticky="w")
    tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=res, orient="horizontal",
             length=length).grid(row=row, column=1)


def on_mode_change():
    if mode.get() == "servo":   # start the servo sliders where the servos are now: no jump
        for i in range(3):
            v_servo[i].set(round(math.degrees(servos.target[i]), 1))
    if mode.get() == "balance":
        loop.reset(data.time)


def level():
    for v in (v_rx, v_ry, v_z, *v_servo, v_spx, v_spy):
        v.set(0.0)


def reset():
    mujoco.mj_resetData(model, data)
    servos.output[:] = 0.0
    servos.target[:] = 0.0
    loop.reset(0.0)
    level()


def kick():
    """Give the ball a random push (~0.15 m/s), like flicking it with a finger."""
    if ball_dof >= 0:
        a = np.random.uniform(0, 2 * math.pi)
        data.qvel[ball_dof:ball_dof + 2] += 0.15 * np.array([math.cos(a), math.sin(a)])


def reset_pid():
    v_P.set(3.0), v_I.set(0.04), v_D.set(25.0)
    loop.reset(data.time)


left = tk.Frame(root)
left.pack(side="left", fill="y")
right = tk.Frame(root)
right.pack(side="left", fill="y")

f_mode = tk.LabelFrame(left, text="Mode", padx=6, pady=4)
f_mode.pack(fill="x", padx=6, pady=4)
for val, txt in (("balance", "Balance (firmware)"), ("tilt", "Plate tilt (IK)"),
                 ("servo", "Servos directly"), ("circle", "Circle demo")):
    tk.Radiobutton(f_mode, text=txt, variable=mode, value=val,
                   command=on_mode_change).pack(side="left")

f_fw = tk.LabelFrame(left, text="Firmware PID (main.ino)", padx=6, pady=4)
f_fw.pack(fill="x", padx=6, pady=4)
slider(f_fw, "P gain", v_P, 0, 10, 0, res=0.1)
slider(f_fw, "I gain", v_I, 0, 0.2, 1, res=0.005)
slider(f_fw, "D gain", v_D, 0, 60, 2, res=0.5)
slider(f_fw, "setpoint x mm", v_spx, -30, 30, 3, res=1)
slider(f_fw, "setpoint y mm", v_spy, -30, 30, 4, res=1)
f_fwb = tk.Frame(f_fw)
f_fwb.grid(row=5, column=0, columnspan=2, sticky="w")
tk.Checkbutton(f_fwb, text="Firmware ON (single press)", variable=v_fw_on).pack(side="left")
tk.Button(f_fwb, text="Kick ball", command=kick).pack(side="left", padx=6)
tk.Button(f_fwb, text="Default gains", command=reset_pid).pack(side="left")

f_tilt = tk.LabelFrame(left, text="Plate (deg / mm)", padx=6, pady=4)
f_tilt.pack(fill="x", padx=6, pady=4)
slider(f_tilt, "tilt about X  rx", v_rx, -MAX_TILT_DEG, MAX_TILT_DEG, 0)
slider(f_tilt, "tilt about Y  ry", v_ry, -MAX_TILT_DEG, MAX_TILT_DEG, 1)
slider(f_tilt, "height  z", v_z, -8, 8, 2)

f_srv = tk.LabelFrame(left, text="Servo angles (deg from neutral)", padx=6, pady=4)
f_srv.pack(fill="x", padx=6, pady=4)
for i in range(3):
    slider(f_srv, f"servo {i}", v_servo[i], round(math.degrees(SERVO_MIN)),
           round(math.degrees(SERVO_MAX)), i)

f_btn = tk.Frame(left)
f_btn.pack(fill="x", padx=6, pady=4)
tk.Button(f_btn, text="Level (all 0)", command=level).pack(side="left")
tk.Button(f_btn, text="Reset simulation", command=reset).pack(side="left", padx=6)
tk.Checkbutton(f_btn, text="Pause", variable=paused).pack(side="left")

# IR heat map: plate frame +-55 mm, seen from above, +Y up (servo 0 side)
f_ir = tk.LabelFrame(right, text="IR sensors (top view, servo 0 at top)", padx=6, pady=4)
f_ir.pack(padx=6, pady=4)
CAN, SPAN = 300, 0.055
canvas = tk.Canvas(f_ir, width=CAN, height=CAN, bg="#202020", highlightthickness=0)
canvas.pack()


def to_px(x, y):
    return CAN / 2 + x / SPAN * CAN / 2, CAN / 2 - y / SPAN * CAN / 2


pcb_r = 0.04995 / SPAN * CAN / 2
canvas.create_oval(CAN / 2 - pcb_r, CAN / 2 - pcb_r, CAN / 2 + pcb_r, CAN / 2 + pcb_r, outline="#2e7d32")
cells, labels = [], []
half = PT_PITCH / 2 / SPAN * CAN / 2
for k, (x, y) in enumerate(PT_XY):
    px, py = to_px(x, y)
    cells.append(canvas.create_rectangle(px - half, py - half, px + half, py + half, fill="#000",
                                         outline="#444"))
    labels.append(canvas.create_text(px, py, text="", fill="white", font=("Consolas", 9)))
sp_dot = canvas.create_oval(0, 0, 0, 0, outline="#66aaff", width=2, dash=(3, 2))
true_dot = canvas.create_oval(0, 0, 0, 0, outline="orange", width=2)
est_dot = canvas.create_line(0, 0, 0, 0, fill="#33ff55", width=3)
est_dot2 = canvas.create_line(0, 0, 0, 0, fill="#33ff55", width=3)
ir_text = tk.StringVar()
tk.Label(f_ir, textvariable=ir_text, justify="left", font=("Consolas", 9), anchor="w").pack(fill="x")
status_label = tk.Label(right, textvariable=status, justify="left", font=("Consolas", 9), anchor="w")
status_label.pack(fill="x", padx=6, pady=4)


def draw_ir():
    sig = sensors.signal
    lvl = (sig - sensors.baseline) / ADC_REF_SIGNAL
    for k in range(16):
        r, g, b = heat(lvl[k])
        canvas.itemconfig(cells[k], fill=f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
        canvas.itemconfig(labels[k], text=f"{int(sig[k])}", fill="black" if lvl[k] > 0.6 else "white")
    sx, sy = to_px(v_spx.get() / 1000, v_spy.get() / 1000)
    canvas.coords(sp_dot, sx - 9, sy - 9, sx + 9, sy + 9) if mode.get() == "balance" else \
        canvas.coords(sp_dot, 0, 0, 0, 0)
    lines = ["orange = ball, green + = firmware estimate, blue = setpoint"]
    if BALL is not None:
        bx, by, _ = ball_in_plate_frame()
        px, py = to_px(bx, by)
        canvas.coords(true_dot, px - 7, py - 7, px + 7, py + 7)
        lines.append(f"true      x {bx * 1000:6.1f}  y {by * 1000:6.1f} mm")
    if loop.estimate is not None:
        ex, ey = to_px(*loop.estimate)
        canvas.coords(est_dot, ex - 7, ey, ex + 7, ey)
        canvas.coords(est_dot2, ex, ey - 7, ex, ey + 7)
        line = f"estimate  x {loop.estimate[0] * 1000:6.1f}  y {loop.estimate[1] * 1000:6.1f} mm"
        if BALL is not None:
            line += f"  (err {math.hypot(loop.estimate[0] - bx, loop.estimate[1] - by) * 1000:.1f})"
        lines.append(line)
    else:
        canvas.coords(est_dot, 0, 0, 0, 0)
        canvas.coords(est_dot2, 0, 0, 0, 0)
        lines.append("estimate  no ball (max - min < 350)")
    ir_text.set("\n".join(lines))


closed = False


def on_close():
    global closed
    closed = True


root.protocol("WM_DELETE_WINDOW", on_close)


# ------------------------------------------------------------------ main loop
def apply_panel():
    m = mode.get()
    loop.enabled = (m == "balance")
    if m == "balance":
        fw.P, fw.I, fw.D = v_P.get(), v_I.get(), v_D.get()
        fw.mode = "ON" if v_fw_on.get() else "OFF"
        fw.set_setpoint_mm(v_spx.get(), v_spy.get())
    elif m == "tilt":
        servos.set_tilt_deg(v_rx.get(), v_ry.get(), v_z.get())
    elif m == "servo":
        servos.set_servos_deg([v.get() for v in v_servo])
    elif m == "circle":
        a = v_rx.get() or 4.0
        servos.set_tilt_deg(a * math.sin(1.0 * data.time), a * math.cos(1.0 * data.time), v_z.get())


frame = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    # The viewer window opens on top: bring the control panel in front of it once.
    root.update()
    root.lift()
    root.attributes("-topmost", True)
    root.after(1500, lambda: root.attributes("-topmost", False))
    while viewer.is_running() and not closed:
        t0 = time.time()
        apply_panel()
        if not paused.get():
            with viewer.lock():
                for _ in range(steps_per_frame):
                    servos.update(data, model.opt.timestep)
                    mujoco.mj_step(model, data)
                    if loop.step(model, data):
                        update_markers()
        viewer.sync()

        frame += 1
        if frame % 6 == 0:   # refresh the readout ~10x per second
            rx, ry = plate_tilt(data)
            act = [math.degrees(data.joint(f"servo{i}").qpos[0]) for i in range(3)]
            loads = joint_loads(model, data)
            pulled = min(loads) < -MAGNET_PULL_N
            lines = [f"t = {data.time:6.2f} s"]
            if mode.get() == "balance":
                lines += [
                    f"firmware  {fw.mode:3s} | ball {'on plate' if fw.ball_on_plate else 'NOT seen'}"
                    f" | centre ({fw.centerX:5.2f}, {fw.centerY:5.2f}) pitch units",
                    f"PID out   roll {fw.smoothRoll:6.2f}  pitch {fw.smoothPitch:6.2f} deg"
                    f" | integral ({fw.integralX:5.2f}, {fw.integralY:5.2f})",
                ]
            lines += [
                "servo target  " + "  ".join(f"{math.degrees(a):6.1f}" for a in servos.target) + "  deg",
                "servo actual  " + "  ".join(f"{a:6.1f}" for a in act) + "  deg",
                f"plate tilt    rx {math.degrees(rx):5.2f}  ry {math.degrees(ry):5.2f}  deg",
                "joint gap     " + "  ".join(f"{g * 1000:6.3f}" for g in joint_gaps(data)) + "  mm",
                "joint load    " + "  ".join(f"{f:6.2f}" for f in loads)
                + f"  N  (- pull; magnet lets go below -{MAGNET_PULL_N} N)",
            ]
            if servos.last_error and mode.get() != "balance":
                lines.append("IK: " + servos.last_error + "  (keeping last reachable pose)")
            if pulled:
                lines.append("!! a magnetic joint is being pulled harder than MAGNET_PULL_N")
            status.set("\n".join(lines))
            status_label.config(fg="red" if pulled else "black")
            draw_ir()
        root.update()

        time.sleep(max(0.0, FRAME_DT - (time.time() - t0)))

root.destroy()
