"""
PID testbench: the BaBot firmware PID (main.ino, ported in v2_cad/babot_firmware.py) run on the
same environment, scenarios and metrics as the RL agents.

Two variants are measured:
  PID_firmware : exactly the firmware (P=3, I=0.04, D=25). The tilt is limited only by the
                 servo range, like on the robot.                        -> the real baseline
  PID_clip8    : same PID, but the tilt is clipped to +-8 deg like the RL action range,
                 so differences with RL are not just the larger allowed tilt.

The PID needs no training, so its sample efficiency is 0 steps by definition.

    py run_pid.py                       # both variants, 30 test episodes each
    py run_pid.py --P 3 --I 0.04 --D 25 # other gains
    py run_pid.py --ball pingpong
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # RL_testing/

from common import config as C                     # noqa: E402
from common.babot_env import BaBotEnv              # noqa: E402
from common.controllers import FirmwarePIDPolicy   # noqa: E402
from common.evaluate import evaluate, print_summary   # noqa: E402

VARIANTS = {
    "PID_firmware": 30.0,          # effectively unclipped: the servo range limits first
    "PID_clip8": C.MAX_TILT_DEG,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--P", type=float, default=None)
    p.add_argument("--I", type=float, default=None)
    p.add_argument("--D", type=float, default=None)
    p.add_argument("--ball", default=C.BALL)
    p.add_argument("--quick", action="store_true", help="3 test episodes only")
    args = p.parse_args()
    seeds = C.EVAL_SEEDS[:3] if args.quick else C.EVAL_SEEDS
    root = os.path.join(HERE, "results_quick" if args.quick else "results")

    for name, max_tilt in VARIANTS.items():
        env = BaBotEnv(ball=args.ball, max_tilt_deg=max_tilt)
        policy = FirmwarePIDPolicy(args.P, args.I, args.D)
        out = os.path.join(root, name)
        _, summary = evaluate(env, policy, seeds=seeds, out_dir=out, label=name)
        # learning curve for the comparison plot: PID return on the learning-curve seeds
        curve_eps, curve_sum = evaluate(env, policy, seeds=C.CURVE_SEEDS)
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump({"controller": name, "type": "pid", "timesteps": 0,
                       "gains": {"P": policy.fw.P, "I": policy.fw.I, "D": policy.fw.D},
                       "max_tilt_deg": max_tilt, "ball": args.ball,
                       "curve_seed_return": curve_sum["cumulative_reward"]["mean"],
                       "metrics": summary}, f, indent=2)
        print_summary(f"{name} (P={policy.fw.P}, I={policy.fw.I}, D={policy.fw.D}, "
                      f"max tilt {max_tilt} deg)", summary)
    print(f"\nSaved to {root}")


if __name__ == "__main__":
    main()
