"""
Python port of the BaBot balancing firmware (BaBot PID/main.ino), for use in the simulator.

Ported 1:1 (same names, same order of operations):
  handleOnMode()  : ball detection, 2-cycle skip when the ball appears, PID on X and Y,
                    output smoothing, 1 s hold then reset when the ball is lost
  handleOffMode() : plate level
  pidControl()    : integ += I*e;  vel = e - lastErr;  out = P*e + integ + D*|vel|/(|vel|+0.001)*vel
  movePlatform()  : the firmware's own (approximate) inverse kinematics
  moveServos()    : clamp the crank angle to -10..85 deg, Servo.write() takes whole degrees
Not ported: buttons, status LED, EEPROM, assembly/calibration modes, digital pot, serial.
The servo offsets are assumed perfectly calibrated.

Units, as in the firmware:
  ball position / setpoint : sensor-pitch units (1 = 21.34 mm), centre = 0
                             x = column - 1.5, y = row - 1.5 (row 0 is on the servo-0 side)
  roll / pitch             : degrees, roll comes from the X error, pitch from the Y error

Frames. The firmware IK places its legs at 0/120/240 deg in its own frame F. For the firmware's
signs to push the ball the right way with the sensor order assumed in ir_sensors.py
(mux channel i = Q(i+1)), F must be:  X_F = PCB +Y (servo-0 direction),  Y_F = PCB +X.
That frame is mirrored relative to the PCB. On the real robot the mirror comes from the FPC cable
/ mux wiring. It is harmless because each leg works in its own vertical plane and the platform is
symmetric about the servo-0 axis. Firmware leg 0 -> sim leg 0, leg 1 -> sim leg 2,
leg 2 -> sim leg 1 (FW_TO_SIM_LEG).
"""
import math

import numpy as np

import ballplate_model as bm
from ir_sensors import BALL_THRESHOLD, firmware_center

# ---------------- main.ino configuration ----------------
P_GAIN, I_GAIN, D_GAIN = 3.0, 0.04, 25.0
R1, R2 = 50.0, 39.2                       # mm
BASE_R = 32.9 / math.sqrt(3.0)            # 19.0 mm
PLAT_R = 107.9 / math.sqrt(3.0)           # 62.3 mm
PLATE_HEIGHT = 65.0                       # mm, ball-joint plane above the servo shafts
OUTPUT_SMOOTH_ALPHA = 0.9
VEL_SCALE = 0.001
LOOP_DT = 1 / 30                          # every30Hz()
BALL_LOST_HOLD = 1.0                      # s, millis() - ballLostTime < 1000

# ---------------- simulation of the hardware timing ----------------
# measureIR(): 32 x (delayMicroseconds(600) + analogRead ~30 us at prescaler 32) = ~20 ms
# from the start of the sensor read until the servos get their new command.
COMMAND_LATENCY = 32 * 630e-6
WHOLE_DEGREES = True                      # Servo.write(int): commands are truncated to 1 deg

FW_TO_SIM_LEG = (0, 2, 1)


class BaBotFirmware:
    def __init__(self):
        self.P, self.I, self.D = P_GAIN, I_GAIN, D_GAIN
        self.mode = "ON"                  # "ON" or "OFF"
        self.reset()

    def reset(self):
        self.centerX = self.centerY = 0.0
        self.setpointX = self.setpointY = 0.0
        self.lastErrorX = self.lastErrorY = 0.0
        self.integralX = self.integralY = 0.0
        self.plateRoll = self.platePitch = 0.0
        self.smoothRoll = self.smoothPitch = 0.0
        self.ballWasOnPlate = False
        self.ballLostTime = 0.0
        self.skipCounter = 0
        self.ball_on_plate = False

    # ---------------- one 30 Hz cycle ----------------
    def cycle(self, t, irSignal):
        """Run one firmware loop with the smoothed IR signals. Returns the 3 servo angles for the
        sim legs, in rad from the sim's neutral (ready for ServoCommander), or None = no new command."""
        c = firmware_center(irSignal)
        self.ball_on_plate = c is not None
        self.centerX, self.centerY = c if c is not None else (0.0, 0.0)
        if self.mode == "OFF":
            self.plateRoll = self.platePitch = 0.0
            return self.movePlatform(0, 0, PLATE_HEIGHT)
        return self.handleOnMode(t)

    def handleOnMode(self, t):
        if self.ball_on_plate:
            if not self.ballWasOnPlate:
                self.skipCounter = 2
                self.lastErrorX = self.setpointX - self.centerX
                self.lastErrorY = self.setpointY - self.centerY
            self.ballWasOnPlate = True
            self.ballLostTime = t
            if self.skipCounter > 0:
                self.skipCounter -= 1
                return None
            self.plateRoll, self.lastErrorX, self.integralX = self.pidControl(
                self.centerX, self.setpointX, self.lastErrorX, self.integralX)
            self.platePitch, self.lastErrorY, self.integralY = self.pidControl(
                self.centerY, self.setpointY, self.lastErrorY, self.integralY)
            a = OUTPUT_SMOOTH_ALPHA
            self.smoothRoll = a * self.plateRoll + (1 - a) * self.smoothRoll
            self.smoothPitch = a * self.platePitch + (1 - a) * self.smoothPitch
            return self.movePlatform(self.smoothRoll, self.smoothPitch, PLATE_HEIGHT)
        if self.ballWasOnPlate and t - self.ballLostTime < BALL_LOST_HOLD:
            return self.movePlatform(self.plateRoll, self.platePitch, PLATE_HEIGHT)
        self.ballWasOnPlate = False
        self.integralX = self.integralY = 0.0
        self.lastErrorX = self.lastErrorY = 0.0
        self.setpointX = self.setpointY = 0.0
        self.plateRoll = self.platePitch = 0.0
        return self.movePlatform(0, 0, PLATE_HEIGHT)

    def pidControl(self, inp, target, lastErr, integ):
        error = target - inp
        integ += self.I * error
        vel = error - lastErr
        scale = abs(vel) / (abs(vel) + VEL_SCALE)
        out = self.P * error + integ + self.D * scale * vel
        return out, error, integ

    # ---------------- kinematics ----------------
    def movePlatform(self, rollDeg, pitchDeg, height):
        """main.ino movePlatform() -> crank angles (deg above horizontal) for firmware legs 0,1,2,
        then moveServos() -> sim servo commands."""
        roll, pitch = -math.radians(rollDeg), -math.radians(pitchDeg)
        angles = []
        for i in range(3):
            a = math.radians(120 * i)
            px, py, pz = PLAT_R * math.cos(a), PLAT_R * math.sin(a), height
            x1 = px * math.cos(pitch) + pz * math.sin(pitch)
            z1 = -px * math.sin(pitch) + pz * math.cos(pitch)
            y1 = py * math.cos(roll) - z1 * math.sin(roll)
            z2 = py * math.sin(roll) + z1 * math.cos(roll)
            dx, dy, dz = x1 - BASE_R * math.cos(a), y1 - BASE_R * math.sin(a), z2
            dxl = dx * math.cos(a) + dy * math.sin(a)
            dyl = dz
            d = math.hypot(dxl, dyl)
            cosA = min(1.0, max(-1.0, (R1 * R1 + d * d - R2 * R2) / (2 * R1 * d)))
            angles.append(math.degrees(math.atan2(dyl, dxl) - math.acos(cosA)))
        return self.moveServos(angles)

    def moveServos(self, angles_fw):
        out = np.zeros(3)
        for i, a in enumerate(angles_fw):
            a = min(bm.CRANK_MAX_DEG, max(bm.CRANK_MIN_DEG, a))
            if WHOLE_DEGREES:
                # servo.write(offset - a) with an int argument truncates; with offset ~75 > a
                # this rounds a up to the next whole degree
                a = math.ceil(a - 1e-9)
            out[FW_TO_SIM_LEG[i]] = math.radians(a) - bm.CRANK_ANGLE0
        return out

    # ---------------- helpers for the panel ----------------
    def set_setpoint_mm(self, x_mm, y_mm):
        """Setpoint in plate millimetres -> firmware units (note: firmware resets it to 0 when the
        ball is lost for more than 1 s)."""
        from ir_sensors import PT_PITCH
        self.setpointX = x_mm / 1000 / PT_PITCH
        self.setpointY = -y_mm / 1000 / PT_PITCH


class FirmwareLoop:
    """Runs the sensors + firmware at 30 Hz inside the physics loop, with the read-to-servo
    latency. Call step(model, data) after every mj_step."""

    def __init__(self, sensors, servos, firmware=None):
        self.sensors, self.servos = sensors, servos
        self.fw = firmware or BaBotFirmware()
        self.next_cycle = 0.0
        self.pending = []                 # (apply_time, command)
        self.estimate = None
        self.enabled = True

    def reset(self, t=0.0):
        self.fw.reset()
        self.next_cycle = t
        self.pending.clear()

    def step(self, model, data):
        t = data.time
        new_sample = False
        if t >= self.next_cycle:
            self.next_cycle += LOOP_DT
            sig, self.estimate = self.sensors.update(model, data)
            new_sample = True
            if self.enabled:
                cmd = self.fw.cycle(t, sig)
                if cmd is not None:
                    self.pending.append((t + COMMAND_LATENCY, cmd))
        while self.pending and self.pending[0][0] <= t:
            _, cmd = self.pending.pop(0)
            if self.enabled:
                self.servos.target = cmd.copy()
        return new_sample
