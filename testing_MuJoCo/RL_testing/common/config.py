"""
Shared settings for every testbench (DDPG, TD3, SAC, PID).

Everything that must be IDENTICAL across controllers for a fair comparison lives here:
the environment, the reward, the evaluation scenarios and the metric definitions.
Algorithm-specific hyper-parameters live in each testbench's own train script.
"""
import os

# ---------------- paths ----------------
RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../RL_testing
SIM_DIR = os.path.normpath(os.path.join(RL_ROOT, "..", "v2_cad"))       # the MuJoCo simulator

# ---------------- environment ----------------
BALL = "pingpong"            # object on the plate (key of ballplate_model.BALLS)
CONTROL_HZ = 30              # firmware loop rate (every30Hz)
EPISODE_STEPS = 300          # 10 s per episode
MAX_TILT_DEG = 8.0           # RL action range: roll, pitch in [-8, +8] deg
HISTORY = 3                  # observation keeps the last 3 sensor frames (gives velocity)
START_RADIUS_MM = (10.0, 45.0)   # ball dropped at a random distance from the setpoint
START_SPEED = 0.05           # m/s, random initial ball speed (uniform 0..START_SPEED)
RANDOM_SETPOINT = False      # True: setpoint random inside SETPOINT_RADIUS_MM each episode
SETPOINT_RADIUS_MM = 20.0
LOST_LIMIT_S = 1.0           # episode ends if the IR board does not see the ball for 1 s
SENSOR_NOISE = True
REWARD_SOURCE = "true"       # "true": reward from the simulated true position (privileged)
                             # "estimate": reward from the IR estimate (what the robot can measure)

# ---------------- reward (see claude/rl_formulation_2026-11-18.md) ----------------
REWARD = dict(
    pos_scale_mm=10.0,       # r_pos = exp(-|e| / 10 mm)
    w_vel=0.1,               # - w_vel * |v| / (0.1 m/s)
    w_action=0.05,           # - w_action * |a|^2      (a normalised to [-1, 1])
    w_action_change=0.1,     # - w_action_change * |a - a_prev|^2
    fall_penalty=-10.0,      # ball off the plate, or lost for more than LOST_LIMIT_S
    magnet_penalty=-1.0,     # per step with magnet tension above MAGNET_PULL_N
)

# ---------------- evaluation protocol (same scenarios for every controller) ----------------
EVAL_SEEDS = list(range(1000, 1030))       # 30 test episodes, final evaluation
CURVE_SEEDS = list(range(500, 505))        # 5 episodes for the learning curve (not the test set)
CURVE_EVAL_EVERY = 5000                    # training steps between learning-curve points

# ---------------- metric definitions ----------------
SETTLE_BAND_MM = 5.0        # settled = error stays below 5 mm until the end of the episode
SUCCESS_BAND_MM = 10.0      # success = ball never lost AND steady-state error below 10 mm
SS_WINDOW_S = 2.0           # steady-state error = mean error over the last 2 s
SAMPLE_EFF_FRACTION = 0.9   # "steps to 90 % of final performance"

# ---------------- network (same for all RL algorithms) ----------------
NET_ARCH = [64, 64]          # MLP 2x64, small enough to run on the robot's microcontroller later
GAMMA = 0.99
