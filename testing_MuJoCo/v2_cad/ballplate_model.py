"""
BalanceBot ball-and-plate platform for MuJoCo, built from the CAD files.

Units: metres, kilograms, radians (helper functions that say _deg take degrees).

Mechanism (3 identical legs at 120 degrees, a "3-RRS" platform):
    servo shaft --[arm_bottom]--> elbow pin --[arm_top]--> 12 mm magnetic ball --> holder (plate)
         R (servo)                  R (pin)                   S (ball joint)
There is NO central support: the plate is held only by the three magnetic ball joints,
so the 3 servos set the plate height and its two tilts.

Numbers marked CAD were measured from the STL files in "../../stl files"
(the STLs are exported in their assembled position, leg 0 on the +Y axis).
Numbers marked GUESS are placeholders: replace them when you have the real parts.
"""
import math
import os

import numpy as np

# ---------------- Leg geometry (CAD, measured from 3X_arm_bottom / 3X_arm_top / 3X_holder) ------
N_LEGS        = 3
SERVO_ANGLE0  = math.pi / 2       # leg 0 on the +Y axis, as in the CAD file
SERVO_RADIUS  = 0.018997          # CAD: servo shaft hole centre, distance from the centre axis
SERVO_Z       = 0.023626          # CAD: servo shaft height
ELBOW_RADIUS  = 0.062299          # CAD: elbow pin hole centre (4.5 mm hole)
ELBOW_Z       = 0.048626          # CAD
MAG_BALL_RADIUS = 0.006           # CAD: 11.8 mm gap between arm_top tip (z=83.6) and holder (z=95.4)
BALL_Z        = 0.0895            # CAD: centre of the magnetic balls at neutral (mid gap)
HOLDER_BOTTOM_Z = 0.0954          # CAD: bottom of the holder cylinder (touches the magnetic ball)
HOLDER_TOP_Z  = 0.1043            # CAD: top of the holder, the PCB sits on it
HOLDER_RADIUS = 0.006             # CAD: holder is a 12 mm cylinder centred over the rod

CRANK_LENGTH  = math.hypot(ELBOW_RADIUS - SERVO_RADIUS, ELBOW_Z - SERVO_Z)       # 50.0 mm
CRANK_ANGLE0  = math.atan2(ELBOW_Z - SERVO_Z, ELBOW_RADIUS - SERVO_RADIUS)       # 30.0 deg
ROD_LENGTH    = BALL_Z - ELBOW_Z                                                  # 40.9 mm
ATTACH_RADIUS = ELBOW_RADIUS                                                      # rod vertical at neutral
ARM_LIMIT     = math.radians(45)  # servo travel each side of neutral (mechanical limit in the sim)

# ---------------- Masses (3D-printed parts) ----------------
PRINT_DENSITY = 1240 * 0.6        # GUESS: PLA at ~60 % infill, kg/m^3
STEEL_DENSITY = 7850
MAG_BALL_MASS = STEEL_DENSITY * 4 / 3 * math.pi * MAG_BALL_RADIUS ** 3     # ~7 g
MAGNET_MASS   = 0.0005            # GUESS: small neodymium magnet in the arm tip
CRANK_MASS    = 6006e-9 * PRINT_DENSITY                                  # CAD volume 6006 mm^3
ROD_MASS      = 3774e-9 * PRINT_DENSITY + MAG_BALL_MASS + MAGNET_MASS    # CAD volume 3774 mm^3
HOLDER_MASS   = 1270e-9 * PRINT_DENSITY                                  # CAD volume 1270 mm^3
ROD_RADIUS    = 0.004
CRANK_RADIUS  = 0.004

# ---------------- Servo model (SG90-class micro servo, GUESS) ----------------
SERVO_KP      = 10.0              # N*m/rad   stiffness of the servo's internal position loop
SERVO_KV      = 0.2               # N*m*s/rad
SERVO_TORQUE  = 0.17              # N*m  (~1.8 kg*cm stall)
SERVO_SPEED   = math.radians(600) # rad/s  (0.1 s / 60 deg), applied by ServoCommander
SERVO_INERTIA = 2e-4              # kg*m^2 reflected rotor inertia

# ---------------- Magnetic joint ----------------
MAGNET_PULL_N = 2.0               # GUESS: force that pulls the ball off its magnet. MEASURE IT.

# ---------------- Plate stack ----------------
PLATE_RADIUS  = 0.070             # GUESS (holders reach 68.3 mm from the centre)
PLATE_THICK   = 0.0015            # 1.5 mm PMMA
PMMA_DENSITY  = 1190
PCB_RADIUS    = 0.060             # GUESS
PCB_THICK     = 0.0016
PCB_MASS      = 0.040             # GUESS: FR4 + IR components
SENSOR_GAP    = 0.003             # GUESS: PCB top -> PMMA underside (IR components)

# ---------------- Contact ----------------
# sliding, torsional, rolling. Rolling/torsional only act with condim=6 (set on the ball).
FRICTION      = (0.8, 0.005, 0.0002)
TIMESTEP      = 0.001

# ---------------- Balls that roll on the plate ----------------
# name: (radius, mass, inertia factor)  inertia = factor * m * r^2
BALLS = {
    "pingpong":      (0.020, 0.0027, 2 / 3),   # thin hollow shell
    "steel_20mm":    (0.010, 0.0327, 2 / 5),
    "plastic_20mm":  (0.010, 0.0046, 2 / 5),
}

# ---------------- CAD meshes (visual only) ----------------
MESH_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "stl files"))
MESH_FILES = {"base": "Base.stl", "arm_bottom": "3X_arm_bottom.stl",
              "arm_top": "3X_arm_top.stl", "holder": "3X_holder.stl"}

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


def ik_deg(rx_deg=0.0, ry_deg=0.0, z_mm=0.0):
    """Same as ik(), in the units you think in: tilts in degrees, height in mm -> servo degrees."""
    q = ik(z_mm / 1000, math.radians(rx_deg), math.radians(ry_deg))
    return [math.degrees(a) for a in q]


class ServoCommander:
    """Turns the servo angles you ask for into what a real hobby servo would do.

    * clamps to +-ARM_LIMIT
    * limits the speed to SERVO_SPEED (a real servo cannot jump)
    * optional resolution (deadband) in degrees, e.g. 0.3 for a cheap SG90

    Use set_servos_deg([...]) or set_tilt_deg(rx, ry, z_mm), then call update(data, dt)
    once per control step; it writes data.ctrl.
    """

    def __init__(self, speed=SERVO_SPEED, resolution_deg=0.0):
        self.speed = speed
        self.resolution = math.radians(resolution_deg)
        self.target = np.zeros(N_LEGS)    # rad, what you asked for
        self.output = np.zeros(N_LEGS)    # rad, what is sent to the servo after rate limiting
        self.last_error = None

    def set_servos_deg(self, angles_deg):
        self.target = np.clip(np.radians(np.asarray(angles_deg, float)), -ARM_LIMIT, ARM_LIMIT)
        self.last_error = None

    def set_tilt_deg(self, rx_deg=0.0, ry_deg=0.0, z_mm=0.0):
        """Returns True if the pose is reachable. If not, the previous target is kept."""
        try:
            self.target = np.array(ik(z_mm / 1000, math.radians(rx_deg), math.radians(ry_deg)))
            self.last_error = None
            return True
        except ValueError as e:
            self.last_error = str(e)
            return False

    def update(self, data, dt):
        step = self.speed * dt
        self.output += np.clip(self.target - self.output, -step, step)
        out = self.output
        if self.resolution > 0:
            out = np.round(out / self.resolution) * self.resolution
        data.ctrl[:N_LEGS] = out


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


def joint_loads(model, data):
    """Force (N) along each rod at the magnetic joint. + = compression (rod pushes the plate up,
    the joint is safe), - = tension (the plate is being pulled off the ball: a real magnet lets go
    when the tension is bigger than MAGNET_PULL_N)."""
    import mujoco
    out = [0.0] * N_LEGS
    for i in range(N_LEGS):
        eq_id = model.equality(f"ball{i}_joint").id
        rows = np.where((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                        & (data.efc_id == eq_id))[0]
        if len(rows) != 3:
            continue
        f_on_rod = data.efc_force[rows]                       # world frame, force on body1 (rod)
        axis = data.body(f"rod{i}").xmat.reshape(3, 3)[:, 2]  # elbow -> ball
        out[i] = float(-(f_on_rod @ axis))
    return out


def plate_tilt(data):
    """Plate tilt (rx, ry) in rad, from the plate's normal vector (matches ik's rx, ry)."""
    n = data.body("plate").xmat.reshape(3, 3)[:, 2]
    return math.atan2(-n[1], n[2]), math.asin(max(-1.0, min(1.0, n[0])))


def _meshes_available():
    return all(os.path.isfile(os.path.join(MESH_DIR, f)) for f in MESH_FILES.values())


def build_xml(ball="pingpong", ball_xy=(0.02, 0.0), timestep=TIMESTEP, meshes=True):
    """ball=None builds the platform without a ball (useful for testing the legs alone).
    meshes=True draws the real CAD parts (if the STL files are found), the physics always
    uses simple shapes."""
    meshes = meshes and _meshes_available()
    fr = " ".join(str(f) for f in FRICTION)
    prim_group = 3 if meshes else 0      # hide the simple shapes when the CAD meshes are drawn
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

    # The STLs are in mm, in the assembled position of leg 0 (on +Y). In a leg's body frame x points
    # outwards (world +Y for leg 0), so a mesh vertex v maps to Rz(-90deg) v - (body origin).
    mrot = f'euler="0 0 {-math.pi / 2:.6f}"'
    assets = ""
    if meshes:
        assets = "<asset>\n" + "\n".join(
            f'    <mesh name="{n}" file="{f}" scale="0.001 0.001 0.001"/>' for n, f in MESH_FILES.items()
        ) + "\n  </asset>"

    legs, eqs, acts, plate_extras = [], [], [], []
    for i in range(N_LEGS):
        phi = servo_angle(i)
        sx, sy = SERVO_RADIUS * math.cos(phi), SERVO_RADIUS * math.sin(phi)
        ax, ay = ATTACH_RADIUS * math.cos(phi), ATTACH_RADIUS * math.sin(phi)
        crank_mesh = rod_mesh = ""
        if meshes:
            crank_mesh = (f'\n      <geom type="mesh" mesh="arm_bottom" {mrot} class="visual" '
                          f'rgba="0.95 0.6 0.1 1" pos="{-SERVO_RADIUS:.6f} 0 {-SERVO_Z:.6f}"/>')
            rod_mesh = (f'\n        <geom type="mesh" mesh="arm_top" {mrot} class="visual" '
                        f'rgba="0.75 0.75 0.8 1" pos="{-ELBOW_RADIUS:.6f} 0 {-ELBOW_Z:.6f}"/>')
            plate_extras.append(
                f'      <geom name="holder{i}" type="mesh" mesh="holder" class="visual" '
                f'euler="0 0 {phi - math.pi / 2:.6f}" pos="0 0 {-BALL_Z:.6f}" rgba="0.3 0.3 0.8 1"/>')
        else:
            hz = (HOLDER_BOTTOM_Z + HOLDER_TOP_Z) / 2 - BALL_Z
            plate_extras.append(
                f'      <geom name="holder{i}" type="cylinder" size="{HOLDER_RADIUS} '
                f'{(HOLDER_TOP_Z - HOLDER_BOTTOM_Z) / 2:.6f}" pos="{ax:.6f} {ay:.6f} {hz:.6f}" '
                f'class="visual" rgba="0.3 0.3 0.8 1"/>')
        legs.append(f"""
    <body name="crank{i}" pos="{sx:.6f} {sy:.6f} {SERVO_Z}" euler="0 0 {phi:.6f}">
      <joint name="servo{i}" type="hinge" axis="0 -1 0" range="{-ARM_LIMIT:.5f} {ARM_LIMIT:.5f}"
             damping="0.0005" armature="{SERVO_INERTIA}"/>
      <geom name="crank{i}_geom" type="capsule" fromto="0 0 0 {cx:.6f} 0 {cz:.6f}"
            size="{CRANK_RADIUS}" mass="{CRANK_MASS:.6f}" rgba="0.95 0.6 0.1 1"
            contype="0" conaffinity="0" group="{prim_group}"/>{crank_mesh}
      <body name="rod{i}" pos="{cx:.6f} 0 {cz:.6f}">
        <joint name="elbow{i}" type="hinge" axis="0 -1 0" damping="0.0002" armature="1e-6"/>
        <geom name="rod{i}_geom" type="capsule" fromto="0 0 0 0 0 {ROD_LENGTH - MAG_BALL_RADIUS:.6f}"
              size="{ROD_RADIUS}" mass="{ROD_MASS:.6f}" rgba="0.6 0.6 0.65 1"
              contype="0" conaffinity="0" group="{prim_group}"/>{rod_mesh}
        <geom name="mag_ball{i}" type="sphere" pos="0 0 {ROD_LENGTH:.6f}" size="{MAG_BALL_RADIUS}"
              class="visual" rgba="0.8 0.8 0.85 1"/>
      </body>
    </body>""")
        # The joint itself: ball centre (rod frame) is pinned to the same point in the plate frame.
        eqs.append(f'    <connect name="ball{i}_joint" body1="rod{i}" body2="plate" '
                   f'anchor="0 0 {ROD_LENGTH:.6f}" solref="0.002 1" solimp="0.95 0.99 0.0005"/>')
        acts.append(f'    <position name="servo{i}" joint="servo{i}" kp="{SERVO_KP}" kv="{SERVO_KV}" '
                    f'ctrlrange="{-ARM_LIMIT:.5f} {ARM_LIMIT:.5f}" '
                    f'forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')

    base_geom = (f'<geom name="base" type="mesh" mesh="base" class="visual" rgba="0.3 0.3 0.3 1"/>'
                 if meshes else
                 '<geom name="base" type="cylinder" size="0.052 0.018" pos="0 0 0.018" '
                 'class="visual" rgba="0.3 0.3 0.3 0.4"/>')
    meshdir = MESH_DIR.replace("\\", "/")

    return f"""<mujoco model="balancebot_3rrs">
  <compiler angle="radian" autolimits="true" meshdir="{meshdir}"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <statistic center="0 0 0.07" extent="0.3"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <default>
    <default class="visual">
      <geom contype="0" conaffinity="0" mass="0" group="1"/>
    </default>
  </default>
  {assets}

  <worldbody>
    <light pos="0 0 1.0" dir="0 0 -1"/>
    <light pos="0.4 -0.4 0.6" dir="-1 1 -1" diffuse="0.4 0.4 0.4"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.1" rgba="0.85 0.85 0.85 1"/>
    {base_geom}
{''.join(legs)}

    <!-- Plate assembly (holders + PCB + PMMA). Frame origin = plane of the magnetic ball centres. -->
    <body name="plate" pos="0 0 {BALL_Z:.6f}">
      <freejoint name="plate_free"/>
      <geom name="plate_geom" type="cylinder" size="{PLATE_RADIUS} {PLATE_THICK / 2}"
            pos="0 0 {PLATE_Z_REL:.6f}" mass="{PLATE_MASS:.6f}" friction="{fr}"
            rgba="0.6 0.8 0.95 0.5"/>
      <geom name="pcb_geom" type="cylinder" size="{PCB_RADIUS} {PCB_THICK / 2}"
            pos="0 0 {PCB_Z_REL:.6f}" mass="{PCB_MASS}" rgba="0.1 0.5 0.2 1"
            contype="0" conaffinity="0"/>
      <geom name="holders_mass" type="sphere" size="0.001" pos="0 0 {HOLDER_TOP_Z - 0.004 - BALL_Z:.6f}"
            mass="{3 * HOLDER_MASS:.6f}" contype="0" conaffinity="0" group="3"/>
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
    print(f"Crank {CRANK_LENGTH * 1000:.2f} mm at {math.degrees(CRANK_ANGLE0):.2f} deg, "
          f"rod {ROD_LENGTH * 1000:.2f} mm")
    print(f"Ball centres at {BALL_Z * 1000:.1f} mm, plate top at {PLATE_TOP_Z * 1000:.1f} mm, "
          f"plate mass {PLATE_MASS * 1000:.1f} g")
    print("Wrote balancebot_3rrs.xml")
