"""
Step 1 check: does the environment run, and how fast?

    py smoke_test.py

1. checks the env against the Gymnasium API (and Stable-Baselines3's checker if installed)
2. runs 1 episode with random actions
3. runs 1 episode with the firmware PID and prints its metrics
4. prints the simulation speed, so you can estimate the training time
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.babot_env import BaBotEnv          # noqa: E402
from common.controllers import FirmwarePIDPolicy   # noqa: E402
from common.evaluate import run_episode        # noqa: E402

env = BaBotEnv()
print("observation:", env.observation_space.shape, " action:", env.action_space.shape)

try:
    from stable_baselines3.common.env_checker import check_env
    check_env(env, warn=True)
    print("[ok] stable-baselines3 env checker passed")
except ImportError:
    print("[skip] stable-baselines3 not installed yet")

obs, info = env.reset(seed=0)
t0, n, done, ret = time.time(), 0, False, 0.0
while not done:
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    ret += r
    n += 1
    done = term or trunc
dt = time.time() - t0
print(f"[ok] random actions: {n} steps, return {ret:.1f}, "
      f"{'ball dropped' if info['failed'] else 'ball stayed on'}  |  {n / dt:.0f} steps/s")

rows, m = run_episode(env, FirmwarePIDPolicy(), seed=1000)
print(f"[ok] firmware PID: return {m['cumulative_reward']:.1f}, settling {m['settling_time_s']:.2f} s, "
      f"steady-state error {m['steady_state_err_mm']:.2f} mm, success {bool(m['success'])}")

sps = n / dt
for steps in (100_000, 200_000):
    est = steps / sps * 1.5 / 3600     # +50 % for network updates and learning-curve episodes
    print(f"estimated training time for {steps // 1000}k steps: ~{est:.1f} h per seed (without GPU)")
