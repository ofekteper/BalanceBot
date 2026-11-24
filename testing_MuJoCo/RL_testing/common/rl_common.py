"""
Training runner shared by the DDPG, TD3 and SAC testbenches.

Each testbench supplies only its algorithm and hyper-parameters; this file makes sure all three
are trained and measured the same way:
  * same environment and reward (config.py)
  * learning curve: every CURVE_EVAL_EVERY steps, 5 deterministic episodes on fixed seeds
  * final test on the 30 EVAL_SEEDS scenarios (same as the PID testbench)
  * several training seeds per algorithm, then mean / std across seeds

Output layout:  <testbench>/results/seed_<k>/{model.zip, monitor.csv, learning_curve.csv,
                                              eval_episodes.csv, eval_summary.json, trajectories.csv}
                <testbench>/results/summary.json   (all seeds together)
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np

from . import config as C
from .babot_env import BaBotEnv
from .controllers import SB3Policy
from .evaluate import evaluate, print_summary, run_episode
from .metrics import EPISODE_METRICS


def common_args(default_timesteps=100_000):
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=default_timesteps, help="training steps per seed")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="training seeds")
    p.add_argument("--ball", default=C.BALL, help="object on the plate (ballplate_model.BALLS)")
    p.add_argument("--quick", action="store_true",
                   help="smoke test: 6000 steps, 1 seed, 3 test episodes (checks everything runs)")
    p.add_argument("--evaluate-only", action="store_true",
                   help="skip training, re-evaluate the saved results/seed_k/model.zip files")
    return p


class LearningCurveCallback:
    """Built lazily so this module imports without stable-baselines3 installed."""

    @staticmethod
    def make(eval_env, seeds, every, path):
        from stable_baselines3.common.callbacks import BaseCallback

        class _CB(BaseCallback):
            def __init__(self):
                super().__init__()
                self.next_eval = 0
                self.t0 = time.time()
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(["timesteps", "wall_s", "mean_return", "std_return",
                                            "success_rate", "mean_ss_err_mm"])

            def _point(self):
                pol = SB3Policy(self.model, "")
                rets, succ, sse = [], [], []
                for s in seeds:
                    _, m = run_episode(eval_env, pol, s)
                    rets.append(m["cumulative_reward"])
                    succ.append(m["success"])
                    sse.append(m["steady_state_err_mm"])
                sse = [v for v in sse if np.isfinite(v)]
                row = [self.num_timesteps, round(time.time() - self.t0, 1), float(np.mean(rets)),
                       float(np.std(rets)), float(np.mean(succ)), float(np.mean(sse)) if sse else math.nan]
                with open(path, "a", newline="") as f:
                    csv.writer(f).writerow(row)
                print(f"  [curve] step {row[0]:7d}  return {row[2]:7.1f}  success {row[4]:.2f}")

            def _on_training_start(self):
                self._point()
                self.next_eval = every

            def _on_step(self):
                if self.num_timesteps >= self.next_eval:
                    self._point()
                    self.next_eval += every
                return True

        return _CB()


def run_testbench(algo_name, make_model, hyperparams, here, args):
    """make_model(env, seed, timesteps) -> an untrained SB3 model."""
    from stable_baselines3.common.monitor import Monitor

    timesteps, seeds, eval_seeds = args.timesteps, args.seeds, C.EVAL_SEEDS
    curve_every = C.CURVE_EVAL_EVERY
    if args.quick:
        timesteps, seeds, eval_seeds, curve_every = 6000, seeds[:1], C.EVAL_SEEDS[:3], 2000
    res_root = os.path.join(here, "results_quick" if args.quick else "results")
    os.makedirs(res_root, exist_ok=True)
    env_kwargs = dict(ball=args.ball)
    with open(os.path.join(res_root, "hyperparams.json"), "w") as f:
        json.dump({"algorithm": algo_name, "timesteps": timesteps, "seeds": seeds, "env": env_kwargs,
                   "net_arch": C.NET_ARCH, "reward": C.REWARD,
                   "hyperparams": {k: str(v) for k, v in hyperparams.items()}}, f, indent=2)

    per_seed = []
    for seed in seeds:
        out = os.path.join(res_root, f"seed_{seed}")
        os.makedirs(out, exist_ok=True)
        eval_env = BaBotEnv(**env_kwargs)
        model_path = os.path.join(out, "model.zip")
        train_s = math.nan
        if args.evaluate_only:
            model = make_model(None, seed, timesteps, load=model_path)
        else:
            print(f"\n##### {algo_name}  seed {seed}  ({timesteps} steps) #####")
            env = Monitor(BaBotEnv(**env_kwargs), filename=os.path.join(out, "monitor.csv"))
            model = make_model(env, seed, timesteps)
            cb = LearningCurveCallback.make(eval_env, C.CURVE_SEEDS, curve_every,
                                            os.path.join(out, "learning_curve.csv"))
            t0 = time.time()
            model.learn(total_timesteps=timesteps, callback=cb, progress_bar=False)
            train_s = time.time() - t0
            model.save(model_path)
        _, summary = evaluate(eval_env, SB3Policy(model, algo_name), seeds=eval_seeds, out_dir=out,
                              label=f"{algo_name} seed {seed}")
        print_summary(f"{algo_name} seed {seed}", summary)
        per_seed.append({"seed": seed, "train_seconds": train_s, "metrics": summary})

    # mean / std ACROSS SEEDS of each seed's mean (the right spread for comparing algorithms)
    agg = {}
    for k in EPISODE_METRICS + ["fell"]:
        v = np.array([ps["metrics"][k]["mean"] for ps in per_seed], float)
        v = v[np.isfinite(v)]
        agg[k] = {"mean": float(v.mean()) if len(v) else math.nan,
                  "std": float(v.std()) if len(v) > 1 else 0.0, "n": int(len(v))}
    with open(os.path.join(res_root, "summary.json"), "w") as f:
        json.dump({"controller": algo_name, "type": "rl", "timesteps": timesteps, "seeds": seeds,
                   "metrics": agg, "per_seed": per_seed}, f, indent=2)
    print_summary(f"{algo_name}, mean over {len(seeds)} seed(s)", agg)
    print(f"\nSaved to {res_root}")
