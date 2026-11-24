"""
Gymnasium environment around the BalanceBot MuJoCo simulator (testing_MuJoCo/v2_cad).

One env step = one 30 Hz firmware cycle, with the same signal path as the real robot:

    IR board (16 phototransistors, firmware smoothing + cubic centroid)
        -> observation -> controller -> plate roll/pitch (deg)
        -> firmware movePlatform() IK -> whole-degree servo commands
        -> ~20 ms sensor-to-servo latency -> rate-limited servos -> physics

RL agents and the PID baseline go through exactly this path, so the only difference
between them is the controller.

Action      : [roll, pitch] in [-1, 1]  ->  x MAX_TILT_DEG degrees (firmware frame:
              roll acts on the firmware X error, pitch on the Y error)
Observation : for each of the last HISTORY frames: [ex, ey, detected]
              (error = setpoint - IR estimate, sensor-pitch units / 2.5),
              then the previous action [roll, pitch]. Size 3*HISTORY + 2.
Reward      : exp(-|e|/10 mm) - velocity, action and action-change penalties,
              -10 when the ball falls / is lost > 1 s, -1 per step above MAGNET_PULL_N.
"""
import math
import os
import sys

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from . import config as C

if C.SIM_DIR not in sys.path:
    sys.path.insert(0, C.SIM_DIR)

import mujoco                                   # noqa: E402
import ballplate_model as bm                    # noqa: E402
from ir_sensors import IRSensorArray, firmware_center, PT_PITCH   # noqa: E402
from babot_firmware import BaBotFirmware, COMMAND_LATENCY, LOOP_DT, PLATE_HEIGHT   # noqa: E402

OBS_SCALE = 2.5     # sensor-pitch units -> roughly [-1, 1] over the sensor grid


class BaBotEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, ball=C.BALL, max_tilt_deg=C.MAX_TILT_DEG, history=C.HISTORY,
                 episode_steps=C.EPISODE_STEPS, random_setpoint=C.RANDOM_SETPOINT,
                 sensor_noise=C.SENSOR_NOISE, reward_source=C.REWARD_SOURCE, reward=None,
                 start_radius_mm=C.START_RADIUS_MM, start_speed=C.START_SPEED):
        super().__init__()
        self.ball = ball
        self.max_tilt = float(max_tilt_deg)
        self.history = int(history)
        self.episode_steps = int(episode_steps)
        self.random_setpoint = random_setpoint
        self.reward_source = reward_source
        self.rw = dict(C.REWARD, **(reward or {}))
        self.start_radius_mm = start_radius_mm
        self.start_speed = start_speed

        self.model = mujoco.MjModel.from_xml_string(bm.build_xml(ball=ball, meshes=False))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.n_sub = int(round(LOOP_DT / self.dt))                  # 33 physics steps per cycle
        self.n_latency = min(self.n_sub, int(round(COMMAND_LATENCY / self.dt)))   # 20
        self.ball_qadr = self.model.jnt_qposadr[self.model.joint("ball_free").id]
        self.ball_dof = self.model.jnt_dofadr[self.model.joint("ball_free").id]
        self.ball_r = bm.BALLS[ball][0]

        self.sensors = IRSensorArray(ball, noise=sensor_noise)
        self.servos = bm.ServoCommander()
        self.kin = BaBotFirmware()          # used only for movePlatform()/moveServos()
        self.lost_limit = int(round(C.LOST_LIMIT_S * C.CONTROL_HZ))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        n_obs = 3 * self.history + 2
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(n_obs,), dtype=np.float32)

    # ------------------------------------------------------------------ helpers
    def _ball_plate_frame(self):
        return self.sensors.ball_in_plate_frame(self.data)

    def _sense(self):
        """One firmware sensing cycle -> centre in firmware units or None."""
        sig, _ = self.sensors.update(self.model, self.data)
        self.last_signal = sig.copy()
        self.last_center = firmware_center(sig)
        return self.last_center

    def _frame(self, c):
        if c is None:                        # not seen: repeat the last known error, flag = 0
            ex, ey = self._last_err_fw
            return [ex / OBS_SCALE, ey / OBS_SCALE, 0.0]
        ex, ey = self.setpoint_fw[0] - c[0], self.setpoint_fw[1] - c[1]
        self._last_err_fw = (ex, ey)
        return [ex / OBS_SCALE, ey / OBS_SCALE, 1.0]

    def _obs(self):
        frames = np.concatenate(self._frames)
        return np.clip(np.r_[frames, self.prev_action], -5, 5).astype(np.float32)

    def _physics(self, n):
        for _ in range(n):
            self.servos.update(self.data, self.dt)
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.sensors.rng = np.random.default_rng(int(rng.integers(2 ** 31)))   # reproducible noise
        mujoco.mj_resetData(self.model, self.data)
        self.servos.output[:] = 0.0
        self.servos.target[:] = 0.0

        if self.random_setpoint:
            r, a = C.SETPOINT_RADIUS_MM * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
            self.setpoint_mm = np.array([r * math.cos(a), r * math.sin(a)])
        else:
            self.setpoint_mm = np.zeros(2)
        if options and "setpoint_mm" in options:
            self.setpoint_mm = np.asarray(options["setpoint_mm"], float)
        # firmware units: x = X / pitch, y = -Y / pitch  (row 0 is on the +Y side)
        self.setpoint_fw = (self.setpoint_mm[0] / 1000 / PT_PITCH, -self.setpoint_mm[1] / 1000 / PT_PITCH)

        # random drop position around the setpoint, random small speed
        r0 = rng.uniform(*self.start_radius_mm) / 1000
        a0 = rng.uniform(0, 2 * math.pi)
        xy = self.setpoint_mm / 1000 + r0 * np.array([math.cos(a0), math.sin(a0)])
        if options and "ball_xy_mm" in options:
            xy = np.asarray(options["ball_xy_mm"], float) / 1000
        self.data.qpos[self.ball_qadr:self.ball_qadr + 7] = [xy[0], xy[1],
                                                            bm.PLATE_TOP_Z + self.ball_r + 0.0002,
                                                            1, 0, 0, 0]
        v, av = rng.uniform(0, self.start_speed), rng.uniform(0, 2 * math.pi)
        self.data.qvel[self.ball_dof:self.ball_dof + 2] = [v * math.cos(av), v * math.sin(av)]
        mujoco.mj_forward(self.model, self.data)

        # warm up the firmware IR smoothing with the plate flat (2 cycles = 67 ms)
        self.sensors.signal = self.sensors.baseline.copy()
        self._last_err_fw = (0.0, 0.0)
        self._frames = []
        for _ in range(2):
            self._physics(self.n_sub)
            self._frames.append(self._frame(self._sense()))
        while len(self._frames) < self.history:
            self._frames.insert(0, self._frames[0])
        self._frames = self._frames[-self.history:]

        self.prev_action = np.zeros(2)
        self.steps = 0
        self.lost_steps = 0
        return self._obs(), self._info(0.0, False, False)

    def step(self, action):
        a = np.clip(np.asarray(action, float).reshape(2), -1.0, 1.0)
        roll, pitch = a * self.max_tilt
        cmd = self.kin.movePlatform(roll, pitch, PLATE_HEIGHT)      # firmware IK + whole degrees

        # sensor read happened at the start of this cycle; the command lands ~20 ms later
        self._physics(self.n_latency)
        self.servos.target = cmd.copy()
        self._physics(self.n_sub - self.n_latency)
        self.steps += 1

        c = self._sense()
        self._frames = self._frames[1:] + [self._frame(c)]
        self.lost_steps = 0 if c is not None else self.lost_steps + 1

        p = self._ball_plate_frame()
        finite = np.all(np.isfinite(self.data.qpos))
        off_plate = (not finite) or p[2] < 0 or math.hypot(p[0], p[1]) > bm.PLATE_RADIUS
        lost = self.lost_steps > self.lost_limit
        terminated = bool(off_plate or lost)
        truncated = self.steps >= self.episode_steps and not terminated

        # ---- reward ----
        if self.reward_source == "estimate" and c is not None:
            ex_mm = (self.setpoint_fw[0] - c[0]) * PT_PITCH * 1000
            ey_mm = -(self.setpoint_fw[1] - c[1]) * PT_PITCH * 1000
            err_mm = math.hypot(ex_mm, ey_mm)
        else:
            err_mm = math.hypot(p[0] * 1000 - self.setpoint_mm[0], p[1] * 1000 - self.setpoint_mm[1])
        speed = float(np.linalg.norm(self.data.qvel[self.ball_dof:self.ball_dof + 2])) if finite else 0.0
        rw = self.rw
        r = (math.exp(-err_mm / rw["pos_scale_mm"])
             - rw["w_vel"] * speed / 0.1
             - rw["w_action"] * float(a @ a)
             - rw["w_action_change"] * float((a - self.prev_action) @ (a - self.prev_action)))
        magnet_over = False
        if finite:
            loads = bm.joint_loads(self.model, self.data)
            magnet_over = -min(loads) > bm.MAGNET_PULL_N
            if magnet_over:
                r += rw["magnet_penalty"]
        if terminated:
            r += rw["fall_penalty"]

        self.prev_action = a
        self.last_tilt = (roll, pitch)
        self.last_servo_cmd = cmd
        return self._obs(), float(r), terminated, truncated, self._info(r, terminated, magnet_over)

    def _info(self, r, failed, magnet_over):
        p = self._ball_plate_frame()
        c = getattr(self, "last_center", None)
        est = (np.nan, np.nan) if c is None else (c[0] * PT_PITCH * 1000, -c[1] * PT_PITCH * 1000)
        return dict(
            t=self.steps / C.CONTROL_HZ,
            x_mm=float(p[0] * 1000), y_mm=float(p[1] * 1000),
            est_x_mm=float(est[0]), est_y_mm=float(est[1]),
            sp_x_mm=float(self.setpoint_mm[0]), sp_y_mm=float(self.setpoint_mm[1]),
            detected=c is not None,
            tilt=tuple(getattr(self, "last_tilt", (0.0, 0.0))),
            servo_deg=tuple(np.degrees(getattr(self, "last_servo_cmd", np.zeros(3)))),
            failed=bool(failed), magnet_over=bool(magnet_over),
        )


def make_env(**kw):
    return BaBotEnv(**kw)
