"""
BalanceBot ball-and-plate platform for MuJoCo, built from the CAD files.

Units: metres, kilograms, radians.

Mechanism (3 identical legs at 120 degrees, a "3-RRS" platform):
    servo shaft --[crank]--> elbow hinge --[rod]--> magnetic ball --> plate holder
        R (servo)               R (pin)                S (ball joint)
There is NO central support: the plate is free in the model and is held only by the
three magnetic ball joints. 3 servos then control 3 plate degrees of freedom
(height + two tilts); the plate's small sideways motion and yaw come from the legs.

Numbers marked CAD were measured from the STL files. Numbers marked GUESS are placeholders:
replace them once you have the real parts.
"""
import math
import numpy as np

# ---------------- Leg geometry (CAD) ----------------
N_LEGS        = 3
SERVO_ANGLE0  = math.pi / 2       # leg 0 on the +Y axis, as in the CAD file
SERVO_RADIUS  = 0.01892           # CAD: servo shaft distance from the centre axis
SERVO_Z       = 0.02352           # CAD: servo shaft height
CRANK_LENGTH  = 0.0500            # CAD: shaft -> elbow pin
CRANK_ANGLE0  = math.radians(30)  # CAD: crank angle above horizontal at neutral
ROD_LENGTH    = 0.039             # GUESS: elbow pin -> centre of the magnetic ball. CHECK IN CAD
ARM_LIMIT     = math.radians(45)  # servo travel each side of neutral

# derived neutral positions
ELBOW_RADIUS  = SERVO_RADIUS + CRANK_LENGTH * math.cos(CRANK_ANGLE0)   # 0.0622
ELBOW_Z       = SERVO_Z + CRANK_LENGTH * math.sin(CRANK_ANGLE0)        # 0.0485
BALL_Z        = ELBOW_Z + ROD_LENGTH     # height of the magnetic ball centres at neutral
ATTACH_RADIUS = ELBOW_RADIUS             # rod is vertical at neutral

# ---------------- Masses (3D-printed parts, GUESS) ----------------
PRINT_DENSITY = 1240 * 0.6        # PLA at ~60 % infill, kg/m^3
CRANK_MASS    = 6006e-9 * PRINT_DENSITY   # CAD volume 6006 mm^3  (~4.5 g)
ROD_MASS      = 3774e-9 * PRINT_DENSITY + 0.001   # CAD volume 3774 mm^3 + 1 g steel ball
ROD_RADIUS    = 0.003
CRANK_RADIUS  = 0.003

# ---------------- Servo model (rough micro-servo, GUESS) ----------------
SERVO_KP      = 10.0              # N*m/rad
SERVO_KV      = 0.2               # N*m*s/rad  (kv/kp = 20 ms lag)
SERVO_TORQUE  = 0.17              # N*m  (~1.8 kg*cm)
SERVO_INERTIA = 2e-4              # kg*m^2 reflected rotor inertia

# ---------------- Plate stack ----------------
# CAD says the holders reach 68 mm from the centre, so the plate must be at least that big.
PLATE_RADIUS  = 0.070             # GUESS (your "same size as base" = 52 mm cannot reach the holders)
PLATE_THICK   = 0.0015            # 1.5 mm PMMA
PMMA_DENSITY  = 1190
PCB_RADIUS    = 0.060             # GUESS
PCB_THICK     = 0.0016
PCB_MASS      = 0.040             # GUESS: FR4 + IR components
HOLDER_MASS   = 0.001             # each, ~1.3 cm^3 of print
HOLDER_TOP_Z  = 0.1043            # CAD: top of the holder, PCB sits on it
SENSOR_GAP    = 0.003             # GUESS: PCB top -> PMMA underside (IR components)
HOLDER_CENTER_RADIUS = 0.0567     # CAD
HOLDER_CENTER_Z      = 0.0974     # CAD

# ---------------- Contact ----------------
# sliding, torsional, rolling. Rolling/torsional only act with condim=6 (set on the ball).
FRICTION      = (0.8, 0.005, 0.0002)
TIMESTEP      = 0.001

# ---------------- Balls ----------------
# name: (radius, mass, inertia factor)  inertia = factor * m * r^2
BALLS = {
    "pingpong":      (0.020, 0.0027, 2 / 3),   # thin hollow shell
    "steel_20mm":    (0.010, 0.0327, 2 / 5),
    "plastic_20mm":  (0.010, 0.0046, 2 / 5),
}

# ---------------- Derived plate geometry (relative to the plate frame at BALL_Z) ----------------
PLATE_MASS    = math.pi * PLATE_RADIUS ** 2 * PLATE_THICK * PMMA_DENSITY
PCB_Z_REL     = HOLDER_TOP_Z + PCB_THICK / 2 - BALL_Z
PLATE_Z_REL   = HOLDER_TOP_Z + PCB_THICK + SENSOR_GAP + PLATE_THICK / 2 - BALL_Z
PLATE_TOP_Z   = BALL_Z + PLATE_Z_REL + PLATE_THICK / 2    # world height of the rolling surface


def servo_angle(i):
    return SERVO_ANGLE0 + i * 2 * math.pi / N_LEGS


_U_R = [np.array([math.cos(servo_angle(i)), math.sin(servo_angle(i)), 0.0]) for i in range(N_LEGS)]
_U_T = [np.array([-math.sin(servo_angle(i)), math.cos(servo_angle(i)), 0.0]) for i in range(N_LEGS)]
_P_LOCAL = [ATTACH_RADIUS * u for u in _U_R]   # ball centres in the plate frame


def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def solve_plate_xy_yaw(z, rx, ry):
    """Sideways position (x, y) and yaw the plate takes for a given height offset and tilt.

    The elbow hinges force every ball centre to stay in its leg's vertical plane."""
    def res(q):
        R = _Rz(q[2]) @ _Rx(rx) @ _Ry(ry)
        c = np.array([q[0], q[1], BALL_Z + z])
        return np.array([_U_T[i] @ (R @ _P_LOCAL[i] + c) for i in range(N_LEGS)])

    q = np.zeros(3)
    for _ in range(40):
        r = res(q)
        if np.max(np.abs(r)) < 1e-13:
            break
        J = np.empty((3, 3))
        for k in range(3):
            dq = np.zeros(3)
            dq[k] = 1e-7
            J[:, k] = (res(q + dq) - r) / 1e-7
        q = q - np.linalg.solve(J, r)
    return q


def ik(z=0.0, rx=0.0, ry=0.0):
    """Servo commands (rad, 0 = neutral) for plate height offset z (m) and tilts rx, ry (rad).

    Plate orientation is Rz(yaw) @ Rx(rx) @ Ry(ry); yaw is whatever the legs force.
    Raises ValueError if the pose is not reachable."""
    x, y, psi = solve_plate_xy_yaw(z, rx, ry)
    R = _Rz(psi) @ _Rx(rx) @ _Ry(ry)
    c = np.array([x, y, BALL_Z + z])
    Lc, Lr = CRANK_LENGTH, ROD_LENGTH
    out = []
    for i in range(N_LEGS):
        P = R @ _P_LOCAL[i] + c
        u = P @ _U_R[i] - SERVO_RADIUS
        v = P[2] - SERVO_Z
        K = (u * u + v * v + Lc * Lc - Lr * Lr) / (2 * Lc)
        M = math.hypot(u, v)
        if abs(K) > M:
            raise ValueError(f"pose unreachable for servo {i}")
        alpha, beta = math.atan2(v, u), math.acos(K / M)
        theta = min((alpha - beta, alpha + beta), key=lambda t: abs(t - CRANK_ANGLE0))
        q = theta - CRANK_ANGLE0
        if abs(q) > ARM_LIMIT:
            raise ValueError(f"servo {i} would need {math.degrees(q):.1f} deg "
                             f"(limit {math.degrees(ARM_LIMIT):.0f})")
        out.append(q)
    return out


def joint_gaps(data):
    """Distance (m) between each rod's ball centre and its plate attachment point.
    About 0 means the magnetic joint is closed."""
    plate = data.body("plate")
    Rp = plate.xmat.reshape(3, 3)
    gaps = []
    for i in range(N_LEGS):
        rod = data.body(f"rod{i}")
        end = rod.xpos + rod.xmat.reshape(3, 3) @ np.array([0.0, 0.0, ROD_LENGTH])
        gaps.append(float(np.linalg.norm(end - (plate.xpos + Rp @ _P_LOCAL[i]))))
    return gaps


def plate_tilt(data):
    """Approximate plate tilt (rx, ry) in rad, from the plate's normal vector."""
    n = data.body("plate").xmat.reshape(3, 3)[:, 2]
    return math.atan2(-n[1], n[2]), math.asin(max(-1.0, min(1.0, n[0])))


def build_xml(ball="pingpong", ball_xy=(0.02, 0.0), timestep=TIMESTEP):
    """ball=None builds the platform without a ball (useful for testing the legs alone)."""
    fr = " ".join(str(f) for f in FRICTION)
    ball_xml = ""
    if ball is not None:
        r, m, k = BALLS[ball]
        ball_inertia = k * m * r * r
        ball_z = PLATE_TOP_Z + r + 0.002
        ball_xml = f"""    <body name="ball" pos="{ball_xy[0]} {ball_xy[1]} {ball_z:.6f}">
      <freejoint name="ball_free"/>
      <inertial pos="0 0 0" mass="{m}" diaginertia="{ball_inertia:.4e} {ball_inertia:.4e} {ball_inertia:.4e}"/>
      <geom name="ball_geom" type="sphere" size="{r}" condim="6" friction="{fr}"
            rgba="1 0.5 0.1 1"/>
    </body>"""
    cx, cz = CRANK_LENGTH * math.cos(CRANK_ANGLE0), CRANK_LENGTH * math.sin(CRANK_ANGLE0)

    legs, eqs, acts, plate_extras = [], [], [], []
    for i in range(N_LEGS):
        phi = servo_angle(i)
        sx, sy = SERVO_RADIUS * math.cos(phi), SERVO_RADIUS * math.sin(phi)
        ax, ay = ATTACH_RADIUS * math.cos(phi), ATTACH_RADIUS * math.sin(phi)
        hx, hy = HOLDER_CENTER_RADIUS * math.cos(phi), HOLDER_CENTER_RADIUS * math.sin(phi)
        legs.append(f"""
    <body name="crank{i}" pos="{sx:.6f} {sy:.6f} {SERVO_Z}" euler="0 0 {phi:.6f}">
      <joint name="servo{i}" type="hinge" axis="0 -1 0" range="{-ARM_LIMIT:.5f} {ARM_LIMIT:.5f}"
             damping="0.0005" armature="{SERVO_INERTIA}"/>
      <geom name="crank{i}_geom" type="capsule" fromto="0 0 0 {cx:.6f} 0 {cz:.6f}"
            size="{CRANK_RADIUS}" mass="{CRANK_MASS:.6f}" rgba="0.95 0.6 0.1 1"
            contype="0" conaffinity="0"/>
      <body name="rod{i}" pos="{cx:.6f} 0 {cz:.6f}">
        <joint name="elbow{i}" type="hinge" axis="0 -1 0" damping="0.0002" armature="1e-6"/>
        <geom name="rod{i}_geom" type="capsule" fromto="0 0 0 0 0 {ROD_LENGTH}"
              size="{ROD_RADIUS}" mass="{ROD_MASS:.6f}" rgba="0.6 0.6 0.65 1"
              contype="0" conaffinity="0"/>
        <geom name="rod{i}_ball" type="sphere" pos="0 0 {ROD_LENGTH}" size="0.003"
              rgba="1 0 1 1" contype="0" conaffinity="0" mass="0"/>
      </body>
    </body>""")
        eqs.append(f'    <connect name="ball{i}_joint" body1="rod{i}" body2="plate" '
                   f'anchor="0 0 {ROD_LENGTH}" solref="0.004 1"/>')
        acts.append(f'    <position name="servo{i}" joint="servo{i}" kp="{SERVO_KP}" kv="{SERVO_KV}" '
                    f'ctrlrange="{-ARM_LIMIT:.5f} {ARM_LIMIT:.5f}" '
                    f'forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')
        plate_extras.append(
            f'      <geom name="cup{i}" type="sphere" pos="{ax:.6f} {ay:.6f} 0" size="0.0045" '
            f'rgba="0 0.9 0.9 0.6" contype="0" conaffinity="0" mass="0"/>')
        plate_extras.append(
            f'      <geom name="holder{i}" type="box" size="0.0115 0.006 0.0068" '
            f'pos="{hx:.6f} {hy:.6f} {HOLDER_CENTER_Z - BALL_Z:.6f}" euler="0 0 {phi:.6f}" '
            f'mass="{HOLDER_MASS}" rgba="0.3 0.3 0.8 1" contype="0" conaffinity="0"/>')

    return f"""<mujoco model="balancebot_3rrs">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <statistic center="0 0 0.07" extent="0.3"/>

  <worldbody>
    <light pos="0 0 1.0" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.1" rgba="0.85 0.85 0.85 1"/>
    <geom name="base" type="cylinder" size="0.052 0.018" pos="0 0 0.018" rgba="0.3 0.3 0.3 0.25"
          contype="0" conaffinity="0" mass="0"/>
{''.join(legs)}

    <!-- Plate assembly (holders + PCB + PMMA). Frame origin = plane of the ball centres. -->
    <body name="plate" pos="0 0 {BALL_Z:.6f}">
      <freejoint name="plate_free"/>
      <geom name="plate_geom" type="cylinder" size="{PLATE_RADIUS} {PLATE_THICK / 2}"
            pos="0 0 {PLATE_Z_REL:.6f}" mass="{PLATE_MASS:.6f}" friction="{fr}"
            rgba="0.6 0.8 0.95 0.5"/>
      <geom name="pcb_geom" type="cylinder" size="{PCB_RADIUS} {PCB_THICK / 2}"
            pos="0 0 {PCB_Z_REL:.6f}" mass="{PCB_MASS}" rgba="0.1 0.5 0.2 1"
            contype="0" conaffinity="0"/>
{chr(10).join(plate_extras)}
    </body>

{ball_xml}
  </worldbody>

  <equality>
{chr(10).join(eqs)}
  </equality>

  <actuator>
{chr(10).join(acts)}
  </actuator>
</mujoco>
"""


if __name__ == "__main__":
    with open("balancebot_3rrs.xml", "w") as f:
        f.write(build_xml())
    print(f"Ball centres at {BALL_Z * 1000:.1f} mm, plate top at {PLATE_TOP_Z * 1000:.1f} mm, "
          f"plate mass {PLATE_MASS * 1000:.1f} g")
    print("Wrote balancebot_3rrs.xml")
