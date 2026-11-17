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


def sensor_test():
    """Put each ball at many places on the plate and compare the IR estimate with the truth."""
    from ir_sensors import IRSensorArray, PT_XY
    import ballplate_model as bm
    print("\nIR sensor array: position estimate error (ball placed on a 5 mm grid)")
    for ball in ("pingpong", "plastic_20mm", "steel_20mm"):
        model = mujoco.MjModel.from_xml_string(build_xml(ball=ball, meshes=False))
        data = mujoco.MjData(model)
        ir = IRSensorArray(ball)
        adr = model.jnt_qposadr[model.joint("ball_free").id]
        r = bm.BALLS[ball][0]
        res = {"in": [], "out": [], "miss_in": 0, "miss_out": 0}
        for x in np.arange(-0.050, 0.0501, 0.005):
            for y in np.arange(-0.050, 0.0501, 0.005):
                if math.hypot(x, y) > 0.050:
                    continue
                data.qpos[adr:adr + 7] = [x, y, bm.PLATE_TOP_Z + r, 1, 0, 0, 0]
                mujoco.mj_forward(model, data)
                est = ir.estimate(ir.read(model, data))
                key = "in" if max(abs(x), abs(y)) <= 0.032 else "out"
                if est is None:
                    res["miss_" + key] += 1
                else:
                    res[key].append(math.hypot(est[0] - x, est[1] - y) * 1000)
        data.qpos[adr:adr + 7] = [*PT_XY[5], bm.PLATE_TOP_Z + r, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        peak = int((ir.read(model, data) - ir.baseline).max())
        fmt = lambda a: f"mean {np.mean(a):4.1f} max {np.max(a):4.1f} mm" if a else "  -  "
        print(f"  {ball:13s} peak signal +{peak:3d} counts | inside sensor grid: {fmt(res['in'])}, "
              f"missed {res['miss_in']} | outside grid: {fmt(res['out'])}, missed {res['miss_out']}")
    ir = IRSensorArray(None)
    false = sum(ir.estimate(ir.read(None, None)) is not None for _ in range(400))
    print(f"  no ball: false detections {false}/400 samples")


def balance_test():
    """Closed loop with the ported firmware: drop the ball, move the setpoint, kick the ball."""
    from babot_firmware import FirmwareLoop
    from ir_sensors import IRSensorArray
    import ballplate_model as bm
    print("\nFirmware balancing (ping-pong ball, PID P=3 I=0.04 D=25 at 30 Hz):")

    def run(start, events, T):
        model = mujoco.MjModel.from_xml_string(build_xml(ball="pingpong", ball_xy=start, meshes=False))
        data = mujoco.MjData(model)
        ir = IRSensorArray("pingpong")
        sc = ServoCommander()
        loop = FirmwareLoop(ir, sc)
        dof = model.jnt_dofadr[model.joint("ball_free").id]
        log, events = [], sorted(events)
        while data.time < T:
            while events and data.time >= events[0][0]:
                _, kind, val = events.pop(0)
                if kind == "setpoint":
                    loop.fw.set_setpoint_mm(*val)
                else:
                    data.qvel[dof:dof + 2] += val
            sc.update(data, model.opt.timestep)
            mujoco.mj_step(model, data)
            if loop.step(model, data):
                p = ir.ball_in_plate_frame(data)
                if p[2] < 0 or math.hypot(p[0], p[1]) > bm.PLATE_RADIUS:
                    return None, data.time
                log.append((data.time, p[0] * 1000, p[1] * 1000))
        return np.array(log), T

    def window(log, t0, t1, target=(0, 0)):
        sel = log[(log[:, 0] >= t0) & (log[:, 0] < t1)]
        d = np.hypot(sel[:, 1] - target[0], sel[:, 2] - target[1])
        return d.mean(), d.max()

    for start in [(0.02, 0.0), (-0.03, -0.015), (0.035, 0.03)]:
        log, t = run(start, [], 8.0)
        if log is None:
            print(f"  drop at ({start[0] * 1000:.0f}, {start[1] * 1000:.0f}) mm: FELL OFF at t={t:.2f}")
            continue
        m, mx = window(log, 5, 8)
        print(f"  drop at ({start[0] * 1000:3.0f}, {start[1] * 1000:3.0f}) mm: after 5 s within "
              f"{mx:4.1f} mm of centre (mean {m:.1f})")
    log, t = run((0.0, 0.0), [(3.0, "setpoint", (15, 10)), (9.0, "kick", (0.15, -0.1))], 14.0)
    if log is None:
        print(f"  setpoint/kick test: FELL OFF at t={t:.2f}")
    else:
        m, mx = window(log, 6, 9, (15, 10))
        print(f"  setpoint moved to (15, 10) mm: 3 s later within {mx:4.1f} mm (mean {m:.1f})")
        sel = log[log[:, 0] >= 9.0]
        far = np.hypot(sel[:, 1] - 15, sel[:, 2] - 10).max()
        m, mx = window(log, 12, 14, (15, 10))
        print(f"  kick of 0.18 m/s: ball went {far:4.1f} mm away, back within {mx:4.1f} mm after 3 s")


if __name__ == "__main__":
    run("flat, no ball", "flat", None)
    run("0.5 deg steps, ping-pong ball", "small_steps", "pingpong")
    run("ramp 0->8 deg, no ball", "ramp", None)
    run("circle 5 deg, no ball", "circle", None)
    run("circle 5 deg, steel ball", "circle", "steel_20mm")
    run("random servo jumps +-45 deg", "servo_jumps", None)
    tilt_accuracy()
    sensor_test()
    balance_test()
