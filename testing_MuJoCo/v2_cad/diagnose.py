"""Headless tests (no viewer, a few seconds each).

For every scenario: does any magnetic joint open (gap > 0.5 mm), does the plate reach the
commanded tilt, and what is the worst pull (tension) on the magnets?

Usage:  python diagnose.py
"""
import math

import mujoco
import numpy as np

from ballplate_model import (MAGNET_PULL_N, ServoCommander, build_xml, joint_gaps, joint_loads,
                             plate_tilt)

SECONDS = 20.0


def command(demo, t, sc):
    if demo == "flat":
        sc.set_tilt_deg(0, 0)
    elif demo == "small_steps":          # 0.5 deg steps, the kind of move a balancing controller makes
        sc.set_tilt_deg(0.5 * (int(t) % 5 - 2), 0.5 * (int(t / 2) % 3 - 1))
    elif demo == "ramp":
        sc.set_tilt_deg(8 * 0.5 * (1 - math.cos(2 * math.pi * t / 8)), 0)
    elif demo == "circle":
        sc.set_tilt_deg(5 * math.sin(2 * t), 5 * math.cos(2 * t))
    elif demo == "servo_jumps":          # random +-45 deg servo jumps (worst case slider abuse)
        rng = np.random.default_rng(int(t))
        sc.set_servos_deg(rng.uniform(-45, 45, 3))
    else:
        raise ValueError(demo)


def run(name, demo, ball):
    model = mujoco.MjModel.from_xml_string(build_xml(ball=ball, meshes=False))
    data = mujoco.MjData(model)
    sc = ServoCommander()
    worst_gap, worst_pull, worst_tilt_err = 0.0, 0.0, 0.0
    failed = None
    while data.time < SECONDS:
        command(demo, data.time, sc)
        for _ in range(10):
            sc.update(data, model.opt.timestep)
            mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            failed = f"NaN at t={data.time:.3f}"
            break
        worst_gap = max(worst_gap, max(joint_gaps(data)))
        worst_pull = max(worst_pull, -min(joint_loads(model, data)))
        if worst_gap > 0.0005:
            failed = f"joint opened ({worst_gap * 1000:.2f} mm) at t={data.time:.3f}"
            break
    rx, ry = plate_tilt(data)
    verdict = "FAILED: " + failed if failed else "OK"
    print(f"{name:34s} {verdict:4s} worst gap {worst_gap * 1000:6.3f} mm | "
          f"worst magnet pull {max(worst_pull, 0):5.2f} N"
          f"{'  <-- above MAGNET_PULL_N' if worst_pull > MAGNET_PULL_N else ''} | "
          f"final tilt ({math.degrees(rx):.2f}, {math.degrees(ry):.2f}) deg")


def tilt_accuracy():
    """Command a small static tilt and check the plate really gets there."""
    model = mujoco.MjModel.from_xml_string(build_xml(ball=None, meshes=False))
    data = mujoco.MjData(model)
    sc = ServoCommander()
    print("\nStatic tilt accuracy (commanded -> measured after 1.5 s):")
    for rx, ry in [(0.2, 0), (0.5, 0), (1, 0), (0, 1), (2, -2), (5, 3), (8, 0)]:
        mujoco.mj_resetData(model, data)
        sc.output[:] = 0
        sc.set_tilt_deg(rx, ry)
        while data.time < 1.5:
            sc.update(data, model.opt.timestep)
            mujoco.mj_step(model, data)
        mrx, mry = plate_tilt(data)
        print(f"  ({rx:4.1f}, {ry:4.1f}) -> ({math.degrees(mrx):5.2f}, {math.degrees(mry):5.2f}) deg   "
              f"servos {np.round(np.degrees(sc.target), 2)}")


if __name__ == "__main__":
    run("flat, no ball", "flat", None)
    run("0.5 deg steps, ping-pong ball", "small_steps", "pingpong")
    run("ramp 0->8 deg, no ball", "ramp", None)
    run("circle 5 deg, no ball", "circle", None)
    run("circle 5 deg, steel ball", "circle", "steel_20mm")
    run("random servo jumps +-45 deg", "servo_jumps", None)
    tilt_accuracy()
