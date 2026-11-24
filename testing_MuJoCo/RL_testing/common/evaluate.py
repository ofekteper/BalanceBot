"""
Shared evaluation protocol: every controller is run on the SAME 30 seeded scenarios
(config.EVAL_SEEDS: same drop position, initial speed and sensor noise), and the same
metrics are computed and saved in the same file layout:

    <out_dir>/eval_episodes.csv     one row per test episode with every metric
    <out_dir>/eval_summary.json     mean / std of every metric
    <out_dir>/trajectories.csv      the full 30 Hz trace of every test episode
"""
import csv
import json
import math
import os

import numpy as np

from . import config as C
from .metrics import episode_metrics, summarize


def run_episode(env, policy, seed, options=None):
    obs, info = env.reset(seed=seed, options=options)
    policy.reset(env)
    rows = []
    done = False
    while not done:
        a = policy.act(obs, env)
        obs, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        rows.append(dict(t=info["t"], x_mm=info["x_mm"], y_mm=info["y_mm"],
                         est_x_mm=info["est_x_mm"], est_y_mm=info["est_y_mm"],
                         sp_x_mm=info["sp_x_mm"], sp_y_mm=info["sp_y_mm"],
                         err_mm=math.hypot(info["x_mm"] - info["sp_x_mm"], info["y_mm"] - info["sp_y_mm"]),
                         roll=info["tilt"][0], pitch=info["tilt"][1],
                         s0=info["servo_deg"][0], s1=info["servo_deg"][1], s2=info["servo_deg"][2],
                         reward=r, detected=int(info["detected"]), magnet_over=int(info["magnet_over"])))
    log = {k: np.array([row[k] for row in rows]) for k in ("t", "err_mm", "roll", "pitch", "reward")}
    log["servo"] = np.array([[row["s0"], row["s1"], row["s2"]] for row in rows])
    log["failed"] = bool(info["failed"])
    return rows, episode_metrics(log)


def evaluate(env, policy, seeds=None, out_dir=None, label=""):
    seeds = C.EVAL_SEEDS if seeds is None else seeds
    episodes, traces = [], []
    for i, s in enumerate(seeds):
        rows, m = run_episode(env, policy, s)
        m = dict(episode=i, seed=s, **m)
        episodes.append(m)
        for row in rows:
            traces.append(dict(episode=i, seed=s, **row))
    summary = summarize(episodes)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        _write_csv(os.path.join(out_dir, "eval_episodes.csv"), episodes)
        _write_csv(os.path.join(out_dir, "trajectories.csv"), traces)
        with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
            json.dump({"controller": label, "n_episodes": len(seeds), "metrics": summary}, f, indent=2)
    return episodes, summary


def print_summary(label, summary):
    print(f"\n=== {label} ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v['mean']:10.3f} +- {v['std']:.3f}   (n={v['n']})")


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
