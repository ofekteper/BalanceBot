"""
3-servo (3-RRS) ball-and-plate platform for MuJoCo.

Units: metres, kilograms, radians.
EVERY NUMBER BELOW IS A PLACEHOLDER. Replace them with measurements from your
real platform (ruler / calipers / kitchen scale).

Mechanism modelled (per leg, 3 legs at 120 degrees):
    servo shaft --(arm)--> ball joint --(rod)--> ball joint on the plate
The plate itself sits on a passive central support that allows only
height (z) and two tilts (rx, ry). With 3 servos this gives exactly
3 inputs for 3 plate degrees of freedom.
ASSUMPTION: check that your real platform has such a central support.
"""
import math
import numpy as np

# ---------------- Base and servos ----------------
N_LEGS        = 3
SERVO_RADIUS  = 0.045            # servo shaft distance from the centre axis
SERVO_Z       = 0.020            # servo shaft height above the ground
SERVO_ANGLE0  = 0.0              # angular position of servo 0 (0 = on +x axis)
ARM_LENGTH    = 0.020            # shaft -> rod joint
ARM_MASS      = 0.002
ARM_LIMIT     = math.radians(40) # max arm swing from horizontal (+ = tip up)

# Servo model (rough micro-servo numbers, e.g. SG90 class)
SERVO_KP      = 1.0              # N*m/rad   position gain
SERVO_KV      = 0.02             # N*m*s/rad velocity damping (limits speed)
SERVO_TORQUE  = 0.17             # N*m       max torque (~1.8 kg*cm)
SERVO_INERTIA = 2e-4             # kg*m^2    reflected rotor inertia (MuJoCo "armature")

# ---------------- Rods (second link) ----------------
ROD_LENGTH    = 0.075
ROD_MASS      = 0.003
ROD_RADIUS    = 0.0015

# ---------------- Plate ----------------
PLATE_RADIUS  = 0.100
PLATE_THICK   = 0.0015           # 1.5 mm PMMA
PLATE_MASS    = 0.060            # PMMA disc
PCB_THICK     = 0.0016
PCB_MASS      = 0.025            # IR sensor PCB under the plate
ATTACH_RADIUS = 0.065            # radius where the rods attach to the plate
MAX_TILT      = math.radians(20) # joint limit of the plate tilt
Z_TRAVEL      = 0.03             # joint limit of the plate height

# ---------------- Contact ----------------
# sliding, torsional, rolling. Rolling/torsional only act with condim=6 (set on the ball).
FRICTION      = (0.8, 0.005, 0.0002)
TIMESTEP      = 0.001

# ---------------- Balls ----------------
# name: (radius, mass, inertia factor)  inertia = factor * m * r^2
# solid sphere = 2/5, thin hollow shell (ping-pong) = 2/3
BALLS = {
    "pingpong":      (0.020, 0.0027, 2 / 3),
    "steel_20mm":    (0.010, 0.0327, 2 / 5),
    "plastic_20mm":  (0.010, 0.0046, 2 / 5),
}

# ---------------- Derived geometry ----------------
_DR = ATTACH_RADIUS - SERVO_RADIUS - ARM_LENGTH           # rod radial offset at neutral
_DZ = math.sqrt(ROD_LENGTH ** 2 - _DR ** 2)               # rod height at neutral
PLATE_Z0 = SERVO_Z + _DZ                                  # plate centre height at neutral


def servo_angle(i):
    return SERVO_ANGLE0 + i * 2 * math.pi / N_LEGS


def ik(z=0.0, rx=0.0, ry=0.0):
    """Servo angles (rad) for plate height offset z (m) and tilts rx, ry (rad).

    Plate orientation is R = Rx(rx) @ Ry(ry), same order as the MuJoCo joints.
    Raises ValueError if the pose is not reachable.
    """
    cx, sx, cy, sy = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R = Rx @ Ry
    out = []
    for i in range(N_LEGS):
        phi = servo_angle(i)
        u_r = np.array([math.cos(phi), math.sin(phi), 0.0])    # radial direction
        u_t = np.array([-math.sin(phi), math.cos(phi), 0.0])   # tangential direction
        p_local = ATTACH_RADIUS * u_r
        P = R @ p_local + np.array([0.0, 0.0, PLATE_Z0 + z])    # attach point, world
        u = P @ u_r - SERVO_RADIUS      # radial distance from the servo shaft
        v = P[2] - SERVO_Z              # height above the servo shaft
        tau = P @ u_t                   # sideways offset (out of the arm's plane)
        a, L = ARM_LENGTH, ROD_LENGTH
        K = (u * u + v * v + tau * tau + a * a - L * L) / (2 * a)
        M = math.hypot(u, v)
        if abs(K) > M:
            raise ValueError(f"pose unreachable for servo {i}")
        alpha = math.atan2(v, u)
        beta = math.acos(K / M)
        cands = (alpha - beta, alpha + beta)
        theta = min(cands, key=abs)     # pick the "elbow" solution near horizontal
        if abs(theta) > ARM_LIMIT:
            raise ValueError(f"servo {i} would need {math.degrees(theta):.1f} deg (limit "
                             f"{math.degrees(ARM_LIMIT):.0f})")
        out.append(theta)
    return out


def build_xml(ball="pingpong", ball_xy=(0.03, 0.0)):
    r, m, k = BALLS[ball]
    ball_inertia = k * m * r * r
    ball_z = PLATE_Z0 + PLATE_THICK / 2 + r + 0.002
    fr = " ".join(str(f) for f in FRICTION)

    legs, eqs, acts, visuals = [], [], [], []
    for i in range(N_LEGS):
        phi = servo_angle(i)
        sx, sy = SERVO_RADIUS * math.cos(phi), SERVO_RADIUS * math.sin(phi)
        legs.append(f"""
    <body name="arm{i}" pos="{sx:.6f} {sy:.6f} {SERVO_Z}" euler="0 0 {phi:.6f}">
      <joint name="servo{i}" type="hinge" axis="0 -1 0" range="{-ARM_LIMIT:.5f} {ARM_LIMIT:.5f}"
             damping="0.0005" armature="{SERVO_INERTIA}"/>
      <geom name="arm{i}_geom" type="capsule" fromto="0 0 0 {ARM_LENGTH} 0 0" size="0.002"
            mass="{ARM_MASS}" rgba="0.95 0.6 0.1 1" contype="0" conaffinity="0"/>
      <body name="rod{i}" pos="{ARM_LENGTH} 0 0">
        <joint name="rod{i}_ball" type="ball" damping="1e-5" armature="1e-6"/>
        <geom name="rod{i}_geom" type="capsule" fromto="0 0 0 {_DR:.6f} 0 {_DZ:.6f}"
              size="{ROD_RADIUS}" mass="{ROD_MASS}" rgba="0.6 0.6 0.65 1"
              contype="0" conaffinity="0"/>
      </body>
    </body>""")
        eqs.append(f'    <connect name="rod{i}_to_plate" body1="rod{i}" body2="plate" '
                   f'anchor="{_DR:.6f} 0 {_DZ:.6f}" solref="0.004 1"/>')
        acts.append(f'    <position name="servo{i}" joint="servo{i}" kp="{SERVO_KP}" kv="{SERVO_KV}" '
                    f'ctrlrange="{-ARM_LIMIT:.5f} {ARM_LIMIT:.5f}" '
                    f'forcerange="{-SERVO_TORQUE} {SERVO_TORQUE}"/>')
        # visual-only servo body (no collision, no mass)
        visuals.append(f'    <geom name="servo{i}_body" type="box" size="0.0115 0.006 0.012" '
                       f'pos="{sx:.6f} {sy:.6f} {SERVO_Z}" euler="0 0 {phi:.6f}" '
                       f'rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0" mass="0"/>')

    return f"""<mujoco model="ball_plate_3rrs">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{TIMESTEP}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <statistic center="0 0 0.08" extent="0.3"/>

  <worldbody>
    <light pos="0 0 1.0" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.1" rgba="0.85 0.85 0.85 1"/>
    <geom name="base" type="cylinder" size="0.085 0.005" pos="0 0 0.005" rgba="0.3 0.3 0.3 1"
          contype="0" conaffinity="0" mass="0"/>
{chr(10).join(visuals)}
{''.join(legs)}

    <!-- Plate on a passive central support: height + two tilts -->
    <body name="plate" pos="0 0 {PLATE_Z0:.6f}">
      <joint name="plate_z"  type="slide" axis="0 0 1" range="{-Z_TRAVEL} {Z_TRAVEL}" damping="0.2"/>
      <joint name="plate_rx" type="hinge" axis="1 0 0" range="{-MAX_TILT:.5f} {MAX_TILT:.5f}" damping="0.005"/>
      <joint name="plate_ry" type="hinge" axis="0 1 0" range="{-MAX_TILT:.5f} {MAX_TILT:.5f}" damping="0.005"/>
      <geom name="plate_geom" type="cylinder" size="{PLATE_RADIUS} {PLATE_THICK / 2}"
            mass="{PLATE_MASS}" friction="{fr}" rgba="0.6 0.8 0.95 0.5"/>
      <geom name="pcb_geom" type="cylinder" size="{PLATE_RADIUS * 0.9} {PCB_THICK / 2}"
            pos="0 0 {-(PLATE_THICK + PCB_THICK) / 2}" mass="{PCB_MASS}"
            rgba="0.1 0.5 0.2 1" contype="0" conaffinity="0"/>
    </body>

    <body name="ball" pos="{ball_xy[0]} {ball_xy[1]} {ball_z:.6f}">
      <freejoint name="ball_free"/>
      <inertial pos="0 0 0" mass="{m}" diaginertia="{ball_inertia:.4e} {ball_inertia:.4e} {ball_inertia:.4e}"/>
      <geom name="ball_geom" type="sphere" size="{r}" condim="6" friction="{fr}"
            rgba="1 0.5 0.1 1"/>
    </body>
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
    with open("ballplate_3rrs.xml", "w") as f:
        f.write(build_xml())
    print(f"Plate neutral height: {PLATE_Z0 * 1000:.1f} mm")
    print("Wrote ballplate_3rrs.xml")
