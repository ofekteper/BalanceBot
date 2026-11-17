"""Headless stability tests (no viewer, runs in seconds).

Runs several scenarios and reports if and when the three legs come apart from the plate.
Usage:  python diagnose.py        then paste the whole output back to Claude.
"""
import math

import mujoco
import numpy as np

from ballplate_model import build_xml, ik, joint_gaps, plate_tilt

SECONDS = 40.0
CTRL_DT = 0.016


def command(demo, t):
    if demo == "flat":
        return 0.0, 0.0, 0.0
    if demo == "ramp":
        a = math.radians(8)
        return 0.0, a * 0.5 * (1 - math.cos(2 * math.pi * t / 16)), 0.0
    if demo == "circle":
        a = math.radians(5)
        return 0.0, a * math.sin(2 * t), a * math.cos(2 * t)
    raise ValueError(demo)


def snapshot(data):
    pf = data.joint("plate_free")
    rx, ry = plate_tilt(data)
    return {
        "t": round(float(data.time), 3),
        "gap_mm": [round(g * 1000, 2) for g in joint_gaps(data)],
        "elbow_deg": [round(math.degrees(data.joint(f"elbow{i}").qpos[0]), 1) for i in range(3)],
        "servo_cmd_deg": [round(math.degrees(c), 1) for c in data.ctrl],
        "servo_act_deg": [round(math.degrees(data.joint(f"servo{i}").qpos[0]), 1) for i in range(3)],
        "plate_tilt_deg": (round(math.degrees(rx), 1), round(math.degrees(ry), 1)),
        "plate_z_mm": round(float(data.body("plate").xpos[2]) * 1000, 1),
        "plate_speed_m_s": round(float(np.linalg.norm(pf.qvel[:3])), 3),
        "plate_spin_rad_s": round(float(np.linalg.norm(pf.qvel[3:6])), 2),
        "max_constraint_force_N": round(float(np.max(np.abs(data.efc_force))) if data.nefc else 0.0, 2),
    }


def run(name, demo, ball, timestep=None):
    kwargs = {} if timestep is None else {"timestep": timestep}
    model = mujoco.MjModel.from_xml_string(build_xml(ball=ball, **kwargs))
    data = mujoco.MjData(model)
    n_sub = max(1, round(CTRL_DT / model.opt.timestep))
    before, first_bad, worst_gap = None, None, 0.0
    while data.time < SECONDS:
        data.ctrl[:] = ik(*command(demo, data.time))
        prev = snapshot(data)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            first_bad, before = ("NaN at t=%.3f" % data.time), prev
            break
        gap = max(joint_gaps(data)) * 1000
        worst_gap = max(worst_gap, gap)
        if gap > 1.0:
            first_bad, before = f"gap > 1 mm at t={data.time:.3f}", prev
            break
    print(f"\n=== {name} ===")
    if first_bad is None:
        print(f"OK for {SECONDS:.0f} s, worst joint gap {worst_gap:.3f} mm")
    else:
        print("FAILED:", first_bad)
        print("  16 ms before:", before)
        print("  at failure  :", snapshot(data))
    try:
        for i in range(len(data.warning)):
            if data.warning[i].number > 0:
                print(f"  MuJoCo warning #{i}: count={data.warning[i].number} info={data.warning[i].lastinfo}")
    except Exception:
        print("  (warning counters unavailable)")


if __name__ == "__main__":
    run("A: flat, no ball", "flat", None)
    run("B: ramp, no ball", "ramp", None)
    run("C: ramp, no ball, half timestep", "ramp", None, timestep=0.0005)
    run("D: circle, no ball", "circle", None)
    run("E: flat, ping-pong ball", "flat", "pingpong")
    run("F: ramp, ping-pong ball (ball rolls off)", "ramp", "pingpong")
    print("\nDone.")
