"""
Controllers with one common interface, so the evaluation code treats them all the same:

    policy.reset(env)            at the start of each episode
    action = policy.act(obs, env)  every 30 Hz step, action in [-1, 1]^2

FirmwarePIDPolicy : the BaBot firmware PID (babot_firmware.py, ported from main.ino),
                    including the 2-cycle skip when the ball appears, the output smoothing
                    and the 1 s hold when the ball is lost.
SB3Policy         : a trained Stable-Baselines3 model (DDPG / TD3 / SAC), deterministic.
"""
import numpy as np

from .babot_env import BaBotFirmware


class _TiltFirmware(BaBotFirmware):
    """Same firmware logic, but movePlatform() returns the (roll, pitch) it was asked for,
    so the env can apply it through the shared actuator path."""

    def movePlatform(self, rollDeg, pitchDeg, height):
        return (rollDeg, pitchDeg)


class FirmwarePIDPolicy:
    name = "PID"

    def __init__(self, P=None, I=None, D=None):
        self.fw = _TiltFirmware()
        if P is not None:
            self.fw.P = P
        if I is not None:
            self.fw.I = I
        if D is not None:
            self.fw.D = D

    def reset(self, env):
        self.fw.reset()
        self.prev = np.zeros(2)

    def act(self, obs, env):
        u = env.unwrapped
        fw = self.fw
        fw.setpointX, fw.setpointY = u.setpoint_fw
        c = u.last_center
        fw.ball_on_plate = c is not None
        fw.centerX, fw.centerY = c if c is not None else (0.0, 0.0)
        out = fw.handleOnMode(u.steps / 30.0)
        if out is not None:                     # None = firmware skips this cycle (keep command)
            self.prev = np.array(out) / u.max_tilt
        return self.prev.copy()


class SB3Policy:
    def __init__(self, model, name):
        self.model, self.name = model, name

    def reset(self, env):
        pass

    def act(self, obs, env):
        a, _ = self.model.predict(obs, deterministic=True)
        return a
