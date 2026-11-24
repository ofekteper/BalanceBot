# RL_testing: DDPG vs TD3 vs SAC vs firmware PID on the BalanceBot simulator

This folder trains and tests three deep-RL controllers on the MuJoCo simulator in `../v2_cad`
and compares them with the BaBot firmware PID, using the same scenarios and the same metrics.

## Folder layout

```
RL_testing/
├── common/                  shared by every testbench, so the comparison is fair
│   ├── config.py            env settings, reward weights, test seeds, metric thresholds
│   ├── babot_env.py         Gymnasium env: IR board -> controller -> firmware IK -> servos -> physics
│   ├── controllers.py       firmware PID policy + wrapper for trained RL models
│   ├── metrics.py           definitions of all metrics
│   ├── evaluate.py          runs the 30 test scenarios and saves the results
│   └── rl_common.py         training loop shared by DDPG / TD3 / SAC (learning curve, seeds)
├── testbench_DDPG/train_ddpg.py
├── testbench_TD3/train_td3.py
├── testbench_SAC/train_sac.py
├── testbench_PID/run_pid.py
├── compare/compare_results.py   tables + plots of all controllers
├── smoke_test.py            step 1: checks the env runs, measures speed
└── requirements.txt
```

## How to run (Windows, from this folder)

```
py -m pip install -r requirements.txt
py smoke_test.py                                  # 1. env works? how fast?
py testbench_PID\run_pid.py                       # 2. PID baseline (a few minutes)
py testbench_DDPG\train_ddpg.py --quick           # 3. smoke test of each RL testbench
py testbench_TD3\train_td3.py --quick
py testbench_SAC\train_sac.py --quick
py compare\compare_results.py --quick
py testbench_DDPG\train_ddpg.py                   # 4. full runs: 100k steps x 3 seeds each
py testbench_TD3\train_td3.py
py testbench_SAC\train_sac.py
py compare\compare_results.py                     # 5. final tables and plots in compare\output
```

Options for every RL testbench: `--timesteps 200000`, `--seeds 0 1 2 3 4`, `--ball pingpong`,
`--evaluate-only` (re-test saved models without training).

## What is identical for every controller

| | |
|---|---|
| Signal path | IR board (firmware smoothing + cubic centroid) -> roll/pitch -> firmware `movePlatform()` IK -> whole-degree servo commands -> ~20 ms latency -> rate-limited servos |
| Rate | 30 Hz, episodes of 300 steps (10 s) |
| Test scenarios | 30 fixed seeds (`EVAL_SEEDS`): ball dropped 10–45 mm from the setpoint with up to 0.05 m/s, same sensor noise |
| RL settings | same network (MLP 2×64), γ = 0.99, learning rate, replay buffer, batch size, τ, warm-up steps |
| Reward | exp(−‖e‖/10 mm) − velocity, action and action-change penalties; −10 if dropped / lost > 1 s; −1 per step above `MAGNET_PULL_N` |

PID is run twice: `PID_firmware` (exactly the robot: P=3, I=0.04, D=25, tilt limited only by the
servos) and `PID_clip8` (same PID with the RL's ±8° tilt limit).

## Metrics

All position metrics use the **true** ball position from the simulator, not the IR estimate.

| Metric | Definition | Better |
|---|---|---|
| Sample efficiency | training steps until the learning-curve return reaches the PID's return, and until 90 % of the run's own final return; plus normalised area under the learning curve. PID = 0 (no training) | fewer steps |
| Cumulative reward | sum of rewards over a test episode | higher |
| Success rate | ball never dropped/lost AND steady-state error < 10 mm | higher |
| Settling time | first time after which the error stays < 5 mm until the end | lower |
| Action smoothness | mean change of the tilt command per step (deg/step) | lower |
| Effort | RMS tilt command (deg) and total servo travel (deg) | lower |
| Steady-state error | mean distance to the setpoint over the last 2 s (mm) | lower |

RL results are given as mean ± std **across training seeds** (each seed tested on all 30 scenarios).
PID results are mean ± std across the 30 test episodes. Thresholds are in `common/config.py`.

## Outputs

* `testbench_<ALGO>/results/seed_<k>/`: `model.zip`, `learning_curve.csv`, `monitor.csv`
  (training episodes), `eval_episodes.csv`, `eval_summary.json`, `trajectories.csv`
* `testbench_<ALGO>/results/summary.json`: all seeds together
* `testbench_PID/results/PID_firmware/`, `.../PID_clip8/`: same files
* `compare/output/`: `comparison_table.md/.csv`, `learning_curves.png`, `metrics_bars.png`,
  `trajectories.png`, `actions.png`

## Notes and limits

* The reward uses the true position by default (`REWARD_SOURCE = "true"`). For training on the
  real robot later it must come from the IR estimate: set `REWARD_SOURCE = "estimate"`.
* Only the ping-pong ball is detectable by the IR board in the sim today. New objects (offset-weight
  or water-filled ping-pong balls) need to be added to `BALLS` in `ballplate_model.py`; then
  run every testbench with `--ball <name>`.
* The sim has no servo backlash, plate flex or calibration error, so absolute numbers will be
  better than on the robot; the comparison between controllers is the useful result.
