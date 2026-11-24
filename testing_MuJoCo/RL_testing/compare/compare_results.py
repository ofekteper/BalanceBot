"""
Compare every controller that has results: DDPG, TD3, SAC (all training seeds) and the two PID
variants. Run it after the testbenches:

    py compare_results.py            # reads testbench_*/results
    py compare_results.py --quick    # reads the smoke-test results (results_quick)

Writes to compare/output/:
    comparison_table.csv / .md   every metric, mean +- std (RL: std across training seeds;
                                 PID: std across the 30 test episodes). Best value in bold in .md
    learning_curves.png          return vs training steps (mean +- std over seeds), PID as a line
    metrics_bars.png             one panel per metric
    trajectories.png             distance to the setpoint over time, all 30 test episodes
    actions.png                  tilt command of test episode 0 for every controller
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from common import config as C                                   # noqa: E402
from common.metrics import (EPISODE_METRICS, LOWER_IS_BETTER, METRIC_LABELS,   # noqa: E402
                            sample_efficiency)

ORDER = ["PID_firmware", "PID_clip8", "DDPG", "TD3", "SAC"]
# categorical palette (validated: CVD-safe adjacent pairs); PID baselines in neutral ink, dashed
COLORS = {"DDPG": "#2a78d6", "TD3": "#eb6834", "SAC": "#1baf7a",
          "PID_firmware": "#52514e", "PID_clip8": "#9a9893"}
STYLES = {"PID_firmware": "--", "PID_clip8": ":"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(results_dir):
    found = {}
    for path in glob.glob(os.path.join(ROOT, "testbench_*", results_dir, "**", "summary.json"),
                          recursive=True):
        with open(path) as f:
            s = json.load(f)
        s["_dir"] = os.path.dirname(path)
        found[s["controller"]] = s
    return {k: found[k] for k in ORDER if k in found} | {k: v for k, v in found.items() if k not in ORDER}


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ("", "nan") else math.nan for r in rows])
            for k in rows[0]} if rows else {}


def curves(s):
    out = []
    for seed in s.get("seeds", []):
        p = os.path.join(s["_dir"], f"seed_{seed}", "learning_curve.csv")
        if os.path.isfile(p):
            out.append(read_csv(p))
    return out


def build_table(res):
    pid_ret = res.get("PID_firmware", {}).get("curve_seed_return")
    rows = []
    for name, s in res.items():
        row = {"controller": name}
        for k in EPISODE_METRICS:
            m = s["metrics"][k]
            row[k] = (m["mean"], m["std"])
        row["fell"] = (s["metrics"]["fell"]["mean"], s["metrics"]["fell"]["std"])
        if s["type"] == "pid":
            row.update(steps_to_pid=(0.0, 0.0), steps_to_90pct=(0.0, 0.0), auc_norm=(math.nan, math.nan),
                       train_minutes=(0.0, 0.0))
        else:
            se = [sample_efficiency(c["timesteps"], c["mean_return"], pid_ret) for c in curves(s)]
            for k in ("steps_to_pid", "steps_to_90pct", "auc_norm"):
                v = np.array([e[k] for e in se], float)
                v = v[np.isfinite(v)]
                row[k] = (float(v.mean()), float(v.std())) if len(v) else (math.nan, math.nan)
                if k == "steps_to_pid":
                    row["reached_pid"] = f"{len(v)}/{len(se)} seeds"
            t = np.array([p["train_seconds"] for p in s["per_seed"]], float) / 60
            t = t[np.isfinite(t)]
            row["train_minutes"] = (float(t.mean()), float(t.std())) if len(t) else (math.nan, math.nan)
        rows.append(row)
    return rows


COLS = ["steps_to_pid", "steps_to_90pct", "auc_norm", "cumulative_reward", "success",
        "settling_time_s", "smoothness_deg", "effort_rms_deg", "servo_travel_deg",
        "steady_state_err_mm", "fell", "train_minutes"]
EXTRA_LABELS = {"fell": "Dropped / lost rate", "train_minutes": "Training time (min)"}


def fmt(m, s, k):
    if not np.isfinite(m):
        return "n/a"
    if k in ("steps_to_pid", "steps_to_90pct"):
        return f"{m / 1000:.0f}k ± {s / 1000:.0f}k" if m else "0 (no training)"
    if k in ("success", "fell"):
        return f"{100 * m:.0f}% ± {100 * s:.0f}"
    return f"{m:.2f} ± {s:.2f}"


def write_tables(rows, out):
    with open(os.path.join(out, "comparison_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller"] + [f"{k}_{x}" for k in COLS for x in ("mean", "std")] + ["reached_pid"])
        for r in rows:
            w.writerow([r["controller"]] + [v for k in COLS for v in r[k]] + [r.get("reached_pid", "")])

    best = {}
    for k in COLS:
        if k == "train_minutes":
            continue
        vals = [(r[k][0], r["controller"]) for r in rows
                if np.isfinite(r[k][0]) and not (k in ("steps_to_pid", "steps_to_90pct") and r[k][0] == 0)]
        if vals and len({round(v, 9) for v, _ in vals}) > 1:
            pick = min if (k in LOWER_IS_BETTER or k == "fell") else max
            best[k] = pick(vals)[1]
    lines = ["| Metric | " + " | ".join(r["controller"] for r in rows) + " |",
             "|---|" + "---|" * len(rows)]
    for k in COLS:
        cells = []
        for r in rows:
            c = fmt(*r[k], k)
            cells.append(f"**{c}**" if best.get(k) == r["controller"] else c)
        arrow = "" if k == "train_minutes" else (" ↓" if (k in LOWER_IS_BETTER or k == "fell") else " ↑")
        lines.append(f"| {METRIC_LABELS.get(k, EXTRA_LABELS.get(k, k))}{arrow} | " + " | ".join(cells) + " |")
    note = ("\n↑ higher is better, ↓ lower is better; best RL/PID value in bold (training-step "
            "metrics compare RL only).\nRL: mean ± std across training seeds. PID: mean ± std across "
            f"the {len(C.EVAL_SEEDS)} test episodes.\nSettling band {C.SETTLE_BAND_MM} mm, success band "
            f"{C.SUCCESS_BAND_MM} mm, steady-state window {C.SS_WINDOW_S} s.\n")
    with open(os.path.join(out, "comparison_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n" + note)
    print("\n".join(lines) + "\n" + note)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plots(res, rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK2})

    # ---- learning curves ----
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    _style(ax)
    for name, s in res.items():
        if s["type"] == "pid":
            if "curve_seed_return" in s:
                ax.axhline(s["curve_seed_return"], color=COLORS.get(name, INK2), ls=STYLES.get(name, "--"),
                           lw=1.5, label=name)
            continue
        cs = curves(s)
        if not cs:
            continue
        n = min(len(c["timesteps"]) for c in cs)
        x = cs[0]["timesteps"][:n]
        y = np.array([c["mean_return"][:n] for c in cs])
        m, sd = y.mean(0), y.std(0)
        col = COLORS.get(name, INK)
        ax.plot(x, m, color=col, lw=2, label=f"{name} ({len(cs)} seeds)")
        ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.15, lw=0)
    ax.set_xlabel("Training steps (30 Hz control steps)")
    ax.set_ylabel("Episode return (5 fixed episodes)")
    ax.set_title("Learning curves: return vs training steps", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "learning_curves.png"), dpi=160)
    plt.close(fig)

    # ---- metric bars ----
    keys = ["cumulative_reward", "success", "settling_time_s", "steady_state_err_mm",
            "smoothness_deg", "effort_rms_deg", "servo_travel_deg", "steps_to_pid"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), facecolor=SURFACE)
    for ax, k in zip(axes.flat, keys):
        _style(ax)
        names = [r["controller"] for r in rows]
        m = np.array([r[k][0] for r in rows], float)
        sd = np.array([r[k][1] for r in rows], float)
        if k == "success":
            m, sd = 100 * m, 100 * sd
        xs = np.arange(len(names))
        ok = np.isfinite(m)
        ax.bar(xs[ok], m[ok], yerr=np.nan_to_num(sd[ok]), width=0.6, capsize=3,
               color=[COLORS.get(n, INK2) for n, o in zip(names, ok) if o],
               edgecolor=SURFACE, linewidth=2, error_kw=dict(ecolor=INK2, lw=1))
        for x, v, e, o in zip(xs, m, np.nan_to_num(sd), ok):   # value labels above the error bar
            ax.text(x, (v + e) if o else 0, f"{v:.3g}" if o else "n/a", ha="center", va="bottom",
                    fontsize=7, color=INK2)
        ax.set_xticks(xs, [n.replace("_", "\n") for n in names], fontsize=7)
        lab = METRIC_LABELS[k] + (" (%)" if k == "success" else "")
        arrow = "↓ better" if k in LOWER_IS_BETTER else "↑ better"
        ax.set_title(f"{lab}\n{arrow}", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "metrics_bars.png"), dpi=160)
    plt.close(fig)

    # ---- trajectories: error vs time over all test episodes ----
    traj = {}
    for name, s in res.items():
        d = s["_dir"] if s["type"] == "pid" else os.path.join(s["_dir"], f"seed_{s['seeds'][0]}")
        p = os.path.join(d, "trajectories.csv")
        if os.path.isfile(p):
            traj[name] = read_csv(p)
    if traj:
        fig, axes = plt.subplots(1, len(traj), figsize=(3.2 * len(traj), 3.2), sharey=True,
                                 facecolor=SURFACE, squeeze=False)
        for ax, (name, tr) in zip(axes[0], traj.items()):
            _style(ax)
            col = COLORS.get(name, INK)
            for ep in np.unique(tr["episode"]):
                sel = tr["episode"] == ep
                ax.plot(tr["t"][sel], tr["err_mm"][sel], color=col, lw=0.7, alpha=0.35)
            ax.axhline(C.SETTLE_BAND_MM, color=INK2, lw=0.8, ls=":")
            ax.set_title(name, loc="left", fontsize=9)
            ax.set_xlabel("Time (s)")
            ax.set_ylim(0, 60)
        axes[0][0].set_ylabel("Distance to setpoint (mm)")
        fig.suptitle(f"All test episodes (dotted line = {C.SETTLE_BAND_MM} mm settling band; "
                     "RL: first training seed)", x=0.01, ha="left", fontsize=9, color=INK2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "trajectories.png"), dpi=160)
        plt.close(fig)

        # ---- action traces, test episode 0 ----
        fig, axes = plt.subplots(2, 1, figsize=(8, 4.5), sharex=True, facecolor=SURFACE)
        for name, tr in traj.items():
            sel = tr["episode"] == 0
            for ax, k in zip(axes, ("roll", "pitch")):
                ax.plot(tr["t"][sel], tr[k][sel], color=COLORS.get(name, INK), ls=STYLES.get(name, "-"),
                        lw=1.3, label=name)
        for ax, k in zip(axes, ("Roll command (deg)", "Pitch command (deg)")):
            _style(ax)
            ax.set_ylabel(k)
        axes[0].legend(frameon=False, fontsize=8, ncol=len(traj))
        axes[0].set_title("Tilt commands, test episode 0 (smoothness and effort)", loc="left", fontsize=10)
        axes[1].set_xlabel("Time (s)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "actions.png"), dpi=160)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="compare the --quick smoke-test results")
    args = ap.parse_args()
    res = load("results_quick" if args.quick else "results")
    if not res:
        print("No results found. Run the testbenches first (see README.md).")
        return
    print("Found:", ", ".join(res))
    out = os.path.join(HERE, "output_quick" if args.quick else "output")
    os.makedirs(out, exist_ok=True)
    rows = build_table(res)
    write_tables(rows, out)
    plots(res, rows, out)
    print(f"Tables and plots saved to {out}")


if __name__ == "__main__":
    main()
