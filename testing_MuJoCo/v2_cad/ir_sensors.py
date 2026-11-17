"""
Simulated IR sensing PCB (the board clipped into the three holders, under the PMMA plate),
processed exactly like the BaBot firmware (BaBot PID/main.ino) does it.

Hardware, from the PCB files (Gerber_Plate_pcb_2025-05-26 + BOM_Plate_Plate_pcb_2025-05-26):
    IR1..IR21 : Everlight IR204/H60, 3 mm 940 nm IR LED, 2θ½ = 50° typ. (330 Ω each)
                840 mil (21.34 mm) grid, 5x5 with the corners removed
    Q1..Q16   : Würth 1540031NC6090 (WL-TTRB), 3 mm 940 nm phototransistor, 60° half sensitivity
                4x4 grid in the gaps between the LEDs, Q1 top-left (-X, +Y), numbered row by row
                -> nets PT00..PT15 -> 20-pin FPC connector U1 -> CD74HC4067 mux on the main board
    Q17       : AO3400A MOSFET, switches all 21 LEDs (firmware pin 7)
    Board     : 100 mm circle with three tabs at 90/210/330 deg that go into the holder clips.
PCB axes = model axes: +Y points at leg 0 (servo 0).
ASSUMED: mux channel i = Q(i+1). The FPC cable is "reverse direction", so check the real order.

Optics (plate frame, so plate tilt does not matter):
  * each LED shines upward with a cos^n beam (half-power angle LED_HALF_ANGLE)
  * light crosses the PMMA (Fresnel loss each way) and hits the ball; the ball reflects part
    diffusely (Lambertian) and part like a mirror (glossy Phong lobe), per BALL_OPTICS
  * each phototransistor collects the light coming back within its cos^m acceptance cone
  * the two PMMA surfaces also reflect ~4 % straight back: a constant background per sensor

Firmware pipeline (main.ino, every 33.3 ms):
  measureIR()               : LEDs off -> read 16, LEDs on -> read 16, signal = on - off,
                              irSignal += 0.7 * (raw - irSignal)
  ballOnPlate()             : max(irSignal) - min(irSignal) >= 350
  calculateWeightedCenter() : v = (s - min) / (max - min), w = v^3 over a 6x6 grid whose border
                              repeats the edge sensors, result in sensor-pitch units, centre = 0

Numbers marked GUESS need to be checked on the real board.
"""
import math

import numpy as np

import ballplate_model as bm

MIL = 0.0254e-3   # m

# ---------------- Layout from the Gerber pick-and-place data (mil, PCB top view) ----------------
LED_XY_MIL = np.array([
    [-838.9, 1682.1], [0.0, 1680.0], [839.8, 1677.1],                                     # IR1-3
    [-1681.2, 840.1], [-840.0, 840.0], [-0.2, 841.1], [840.0, 840.0], [1681.4, 842.1],    # IR4-8
    [-1681.6, 0.1], [-840.2, 0.1], [-1.2, 2.1], [840.4, 2.1], [1681.0, -1.0],             # IR9-13
    [-1681.3, -840.9], [-840.7, -840.9], [-0.1, -840.9], [840.4, -840.9], [1681.0, -840.9],  # IR14-18
    [-840.6, -1679.9], [0.0, -1681.0], [841.0, -1679.9],                                  # IR19-21
])
PT_XY_MIL = np.array([[x, y] for y in (1260, 420, -420, -1260) for x in (-1260, -420, 420, 1260)])
LED_XY = LED_XY_MIL * MIL     # (21, 2) m
PT_XY = PT_XY_MIL * MIL       # (16, 2) m, index i = Q(i+1) = mux channel i (ASSUMED)
PT_PITCH = 840 * MIL          # 21.34 mm, the firmware's position unit
PCB_TAB_ANGLES = (90, 210, 330)

# ---------------- Optical parameters ----------------
PART_RADIUS     = 0.0015      # 3 mm T-1 packages (BOM footprints LED-TH_BD3.0 / TEFT4300)
PART_HEIGHT     = 0.0053      # T-1 body height
EMIT_HEIGHT     = 0.0030      # GUESS: LED die / sensor chip height above the PCB
LED_HALF_ANGLE  = 25.0        # IR204/H60 datasheet: 2θ½ = 50° typ (37..63)
LED_SIDE_LIGHT  = 0.08        # GUESS: share of wide-angle light leaking from the epoxy body
PT_HALF_ANGLE   = 30.0        # WL-TTRB: 60° angle of half sensitivity (full angle)
PMMA_REFLECT    = 0.04        # Fresnel reflection per PMMA surface at normal incidence (n = 1.49)
PMMA_TRANSMIT   = (1 - PMMA_REFLECT) ** 2   # through the plate once (two surfaces)
# ball: (diffuse albedo, mirror-like fraction, gloss exponent)  -- all GUESS
BALL_OPTICS = {
    "pingpong":     (0.55, 0.35, 25),   # "reflects IR directly back down like a mirror"
    "plastic_20mm": (0.60, 0.20, 25),
    "steel_20mm":   (0.05, 0.60, 60),   # polished: mostly mirror
}
N_PATCHES       = 400        # samples on the lower half of the ball (the only half the LEDs see)

# ---------------- Electronics (Arduino 10-bit ADC) ----------------
ADC_MAX         = 1023
ADC_DARK        = 15          # GUESS counts with LEDs off in the dark
AMBIENT_COUNTS  = 40          # GUESS room IR, present in both reads; cancelled by on - off
ADC_REF_SIGNAL  = 600         # GUESS: ping-pong ball right over a sensor gives +600 counts
                              # (firmware needs max-min >= 350 to see a ball, so it must be > 350)
NOISE_COUNTS    = 3.0         # GUESS 1-sigma noise of one analogRead

# ---------------- Firmware constants (main.ino) ----------------
SAMPLE_RATE     = 30.0        # every30Hz()
IR_SMOOTH_ALPHA = 0.7
BALL_THRESHOLD  = 350.0


def _cos_power(half_angle_deg):
    """Exponent n so that cos(theta)^n = 0.5 at the half-power angle."""
    return math.log(0.5) / math.log(math.cos(math.radians(half_angle_deg)))


def _lower_hemisphere(n):
    """n roughly evenly spread unit normals with z < 0 (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = -i / n                                   # uniform in z on (-1, 0) = uniform area
    rxy = np.sqrt(1 - z * z)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.c_[rxy * np.cos(th), rxy * np.sin(th), z]


def firmware_center(sig):
    """calculateWeightedCenter() from main.ino. sig: 16 values, channel order.
    Returns (x, y) in sensor-pitch units (column / row index - 1.5, row 0 = +Y side), or None
    when ballOnPlate() is false."""
    sig = np.asarray(sig, float)
    lo, hi = sig.min(), sig.max()
    if hi - lo < BALL_THRESHOLD:
        return None
    v = np.clip((sig.reshape(4, 4) - lo) / (hi - lo), 0, None)
    w = np.pad(v, 1, mode="edge") ** 3           # 6x6, border = copy of the edge (mirrorIdx)
    idx = np.arange(-1, 5)
    s = w.sum()
    if s < 1e-6:
        return None
    return (w.sum(0) @ idx) / s - 1.5, (w.sum(1) @ idx) / s - 1.5


def grid_to_plate(gx, gy):
    """Firmware units -> metres in the plate frame (column 0 is -X, row 0 is +Y)."""
    return gx * PT_PITCH, -gy * PT_PITCH


class IRSensorArray:
    """read(model, data)  -> one raw ambient-subtracted sample (16 counts, channel order)
    estimate(sig)          -> firmware ball position in metres (plate frame), or None
    update(model, data)    -> one full firmware cycle: read, smooth, estimate (call at 30 Hz)
    """

    def __init__(self, ball="pingpong", seed=0, noise=True):
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.ball = ball
        z_emit = bm.PCB_Z_REL + bm.PCB_THICK / 2 + EMIT_HEIGHT          # plate frame
        self.led = np.c_[LED_XY, np.full(len(LED_XY), z_emit)]
        self.pt = np.c_[PT_XY, np.full(len(PT_XY), z_emit)]
        self.z_plate_bottom = bm.PLATE_Z_REL - bm.PLATE_THICK / 2
        self.z_plate_top = bm.PLATE_Z_REL + bm.PLATE_THICK / 2
        self.n_led = _cos_power(LED_HALF_ANGLE)
        self.n_pt = _cos_power(PT_HALF_ANGLE)
        self.normals = _lower_hemisphere(N_PATCHES)
        self.background = self._pmma_background()

        # scale so that the reference ball (ping-pong) right over a sensor gives ADC_REF_SIGNAL
        r_ref = bm.BALLS["pingpong"][0]
        ref = self._ball_light(np.array([PT_XY[5, 0], PT_XY[5, 1], self.z_plate_top + r_ref]),
                               r_ref, BALL_OPTICS["pingpong"])
        self.gain = ADC_REF_SIGNAL / (ref + self.background).max()
        self.baseline = self.gain * self.background      # signal with no ball (noise-free)
        self.signal = self.baseline.copy()                # firmware irSignal (smoothed)
        self.raw = self.baseline.copy()

    # ---------- optics ----------
    def _ball_light(self, c, r, optics):
        """Light (arbitrary units) reaching each sensor after reflecting off a ball centred at c."""
        albedo, spec, gloss = optics
        n = self.normals                                # (K, 3)
        P = c + r * n
        dA = 2 * math.pi * r * r / N_PATCHES
        v = P[None] - self.led[:, None]                 # (J, K, 3) LED -> patch
        d = np.linalg.norm(v, axis=2)
        u = v / d[..., None]
        cos_in = np.clip(-(u * n[None]).sum(2), 0, None)
        E = self._led_beam(u[..., 2]) * cos_in / d ** 2 * PMMA_TRANSMIT   # (J, K)

        w = P[None] - self.pt[:, None]                  # (I, K, 3) sensor -> patch
        dw = np.linalg.norm(w, axis=2)
        uw = w / dw[..., None]
        cos_out = np.clip(-(uw * n[None]).sum(2), 0, None)
        geo = cos_out * np.clip(uw[..., 2], 0, None) ** self.n_pt / dw ** 2 * dA * PMMA_TRANSMIT  # (I, K)

        out = albedo / math.pi * (geo @ E.sum(0))       # diffuse part
        if spec > 0:
            refl = u - 2 * (u * n[None]).sum(2)[..., None] * n[None]          # (J, K, 3) mirror dir
            # cos between mirror direction and direction to each sensor, as (K, J, I)
            c = np.matmul(refl.transpose(1, 0, 2), -uw.transpose(1, 2, 0))
            np.clip(c, 0, None, out=c)
            lobe = np.exp(gloss * np.log(c + 1e-12))
            brdf = spec * (gloss + 2) / (2 * math.pi)
            out += brdf * (lobe * E.T[:, :, None] * geo.T[:, None, :]).sum((0, 1))
        return out

    def _led_beam(self, cos_t):
        """LED intensity vs angle: main cos^n lobe + LED_SIDE_LIGHT of Lambertian side light."""
        c = np.clip(cos_t, 0, None)
        return (1 - LED_SIDE_LIGHT) * c ** self.n_led + LED_SIDE_LIGHT * c

    def _pmma_background(self):
        """Specular reflection of every LED in the two PMMA surfaces (image-source method)."""
        total = np.zeros(len(self.pt))
        for z_s, refl in ((self.z_plate_bottom, PMMA_REFLECT),
                          (self.z_plate_top, PMMA_REFLECT * PMMA_TRANSMIT)):
            img = self.led.copy()
            img[:, 2] = 2 * z_s - img[:, 2]
            v = img[None] - self.pt[:, None]            # (I, J, 3)
            d = np.linalg.norm(v, axis=2)
            cz = v[..., 2] / d
            total += (refl * self._led_beam(cz) * cz ** self.n_pt / d ** 2).sum(1)
        return total

    def _adc(self, counts):
        if self.noise:
            counts = counts + self.rng.normal(0, NOISE_COUNTS, counts.shape)
        return np.clip(np.round(counts), 0, ADC_MAX)

    # ---------- public ----------
    def ball_in_plate_frame(self, data):
        plate, ball = data.body("plate"), data.body("ball")
        return plate.xmat.reshape(3, 3).T @ (ball.xpos - plate.xpos)

    def read(self, model, data):
        """One measureIR() raw sample: (LEDs on) - (LEDs off), in ADC counts, channel order."""
        light = self.background.copy()
        if self.ball is not None:
            r = bm.BALLS[self.ball][0]
            light += self._ball_light(self.ball_in_plate_frame(data), r, BALL_OPTICS[self.ball])
        off = self._adc(np.full(16, ADC_DARK + AMBIENT_COUNTS, float))
        on = self._adc(ADC_DARK + AMBIENT_COUNTS + self.gain * light)
        self.raw = on - off
        return self.raw

    def estimate(self, sig):
        """Firmware ball position (metres, plate frame) from 16 signals, or None if no ball."""
        g = firmware_center(sig)
        return None if g is None else grid_to_plate(*g)

    def update(self, model, data):
        """One 30 Hz firmware cycle. Returns (smoothed signals, estimate or None)."""
        raw = self.read(model, data)
        self.signal = self.signal + IR_SMOOTH_ALPHA * (raw - self.signal)
        return self.signal, self.estimate(self.signal)

    def grid(self, sig):
        """Signals as a 4x4 array laid out like the board seen from above (row 0 = +Y side)."""
        return np.asarray(sig).reshape(4, 4)


# Checking against the real board:
#   1. enable sendSerialData() in main.ino: it prints the 16 irSignal values + centre
#   2. no ball               -> compare with IRSensorArray(None).baseline
#   3. ball over one sensor  -> sets ADC_REF_SIGNAL
#   4. ball moved along a row in 5 mm steps -> tune EMIT_HEIGHT and BALL_OPTICS until the
#      simulated profile (python diagnose.py) matches
