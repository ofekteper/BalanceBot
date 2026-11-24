"""
Metric definitions, shared by every testbench. All position metrics use the TRUE ball position
from the simulator (not the IR estimate), so sensor bias does not hide controller differences.

Per episode (log sampled at 30 Hz):
  cumulative_reward   sum of rewards over the episode                              (higher = better)
  success             1 if the ball was never lost/dropped AND the steady-state
                      error is below SUCCESS_BAND_MM                                (rate: higher = better)
  settling_time_s     first time after which the error stays below SETTLE_BAND_MM
                      until the end; NaN if it never settles or the ball fell      (lower = better)
  steady_state_err_mm mean error over the last SS_WINDOW_S seconds (NaN if fell)   (lower = better)
  smoothness_deg      mean |change of tilt command| per step, deg/step              (lower = smoother)
  effort_rms_deg      RMS tilt command, deg                                         (lower = less effort)
  servo_travel_deg    total servo movement over the episode, deg (sum of 3 servos)  (lower = less effort)

Per training run (from the learning curve):
  steps_to_threshold  first training step where the learning-curve return reaches a threshold
                      (the PID's return, or 90 % of the run's own final return)     (lower = better)
  auc_norm            area under the learning curve / (final return * steps)       (higher = faster)
"""
import math

import numpy as np

from . import config as C

LOWER_IS_BETTER = {"settling_time_s", "steady_state_err_mm", "smoothness_deg", "effort_rms_deg",
                   "servo_travel_deg", "steps_to_pid", "steps_to_90pct"}

METRIC_LABELS = {
    "cumulative_reward": "Cumulative reward",
    "success": "Success rate",
    "settling_time_s": "Settling time (s)",
    "steady_state_err_mm": "Steady-state error (mm)",
    "smoothness_deg": "Action change (deg/step)",
    "effort_rms_deg": "Effort, RMS tilt (deg)",
    "servo_travel_deg": "Servo travel (deg)",
    "steps_to_pid": "Steps to reach PID return",
    "steps_to_90pct": "Steps to 90% of final",
    "auc_norm": "Learning-curve AUC (norm.)",
}


def episode_metrics(log):
    """log: dict of equal-length arrays: t, err_mm, roll, pitch, reward, servo (N x 3), failed (bool)."""
    t = np.asarray(log["t"], float)
    e = np.asarray(log["err_mm"], float)
    tilt = np.c_[log["roll"], log["pitch"]]
    servo = np.asarray(log["servo"], float)
    failed = bool(log["failed"])
    dt = 1.0 / C.CONTROL_HZ

    out = {"cumulative_reward": float(np.sum(log["reward"])), "fell": float(failed),
           "length_s": float(len(t) * dt)}

    n_ss = max(1, int(round(C.SS_WINDOW_S / dt)))
    if failed or len(e) < n_ss:
        out["steady_state_err_mm"] = math.nan
        out["settling_time_s"] = math.nan
        out["success"] = 0.0
    else:
        ss = float(np.mean(e[-n_ss:]))
        out["steady_state_err_mm"] = ss
        outside = np.nonzero(e > C.SETTLE_BAND_MM)[0]
        if len(outside) == 0:
            out["settling_time_s"] = 0.0
        elif outside[-1] == len(e) - 1:
            out["settling_time_s"] = math.nan            # still outside the band at the end
        else:
            out["settling_time_s"] = float(t[outside[-1] + 1])
        out["success"] = float(ss <= C.SUCCESS_BAND_MM)

    if len(tilt) > 1:
        out["smoothness_deg"] = float(np.mean(np.linalg.norm(np.diff(tilt, axis=0), axis=1)))
        out["servo_travel_deg"] = float(np.sum(np.abs(np.diff(servo, axis=0))))
    else:
        out["smoothness_deg"] = out["servo_travel_deg"] = math.nan
    out["effort_rms_deg"] = float(np.sqrt(np.mean(np.sum(tilt ** 2, axis=1)))) if len(tilt) else math.nan
    return out


EPISODE_METRICS = ["cumulative_reward", "success", "settling_time_s", "steady_state_err_mm",
                   "smoothness_deg", "effort_rms_deg", "servo_travel_deg"]


def summarize(episodes):
    """List of per-episode metric dicts -> {metric: {mean, std, n}} (NaNs ignored)."""
    out = {}
    for k in EPISODE_METRICS + ["fell"]:
        v = np.array([ep[k] for ep in episodes], float)
        v = v[np.isfinite(v)]
        out[k] = {"mean": float(v.mean()) if len(v) else math.nan,
                  "std": float(v.std()) if len(v) else math.nan,
                  "n": int(len(v))}
    return out


def sample_efficiency(steps, returns, pid_return=None):
    """steps, returns: learning-curve arrays of one training run."""
    steps, returns = np.asarray(steps, float), np.asarray(returns, float)
    out = {"steps_to_pid": math.nan, "steps_to_90pct": math.nan, "auc_norm": math.nan}
    if len(steps) == 0:
        return out
    final = float(np.mean(returns[-3:]))           # average of the last 3 points
    lo = float(returns.min())
    target = lo + C.SAMPLE_EFF_FRACTION * (final - lo)   # 90 % of the way from worst to final
    hit = np.nonzero(returns >= target)[0]
    out["steps_to_90pct"] = float(steps[hit[0]]) if len(hit) else math.nan
    if pid_return is not None and np.isfinite(pid_return):
        hit = np.nonzero(returns >= pid_return)[0]
        out["steps_to_pid"] = float(steps[hit[0]]) if len(hit) else math.nan
    if len(steps) > 1 and final != lo:
        y = (returns - lo) / (final - lo)
        trapz = getattr(np, "trapezoid", None) or np.trapz
        out["auc_norm"] = float(trapz(y, steps) / (steps[-1] - steps[0]))
    return out
