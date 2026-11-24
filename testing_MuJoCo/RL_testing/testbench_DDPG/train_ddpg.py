"""
DDPG testbench: Deep Deterministic Policy Gradient (Lillicrap et al., 2015).

One actor, one critic, Gaussian exploration noise on the action.
Known weak points that this testbench is meant to show: Q-value over-estimation and
sensitivity to hyper-parameters / seeds.

    py train_ddpg.py --quick          # smoke test (~1-2 min)
    py train_ddpg.py                  # full run: 100k steps x 3 seeds
    py train_ddpg.py --timesteps 200000 --seeds 0 1 2 3 4
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # RL_testing/

from common import config as C                     # noqa: E402
from common.rl_common import common_args, run_testbench   # noqa: E402

# ---------------- DDPG hyper-parameters (shared values first, so the comparison is fair) ----------
HP = dict(
    learning_rate=3e-4,
    buffer_size=200_000,
    learning_starts=5_000,       # random actions first, to fill the replay buffer
    batch_size=256,
    tau=0.005,                   # soft target update
    gamma=C.GAMMA,
    train_freq=1,
    gradient_steps=1,
    # DDPG-specific
    action_noise_sigma=0.1,      # Gaussian exploration noise (fraction of the action range)
)


def make_model(env, seed, timesteps, load=None):
    import numpy as np
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    if load:
        return DDPG.load(load)
    hp = dict(HP)
    sigma = hp.pop("action_noise_sigma")
    noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma * np.ones(2))
    return DDPG("MlpPolicy", env, action_noise=noise, seed=seed, verbose=0,
                policy_kwargs=dict(net_arch=C.NET_ARCH), **hp)


if __name__ == "__main__":
    run_testbench("DDPG", make_model, HP, HERE, common_args().parse_args())
