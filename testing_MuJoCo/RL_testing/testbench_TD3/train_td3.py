"""
TD3 testbench: Twin Delayed DDPG (Fujimoto et al., 2018).

DDPG plus three fixes: twin critics (take the minimum -> less over-estimation),
delayed actor updates, and target-policy smoothing noise.

    py train_td3.py --quick           # smoke test (~1-2 min)
    py train_td3.py                   # full run: 100k steps x 3 seeds
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # RL_testing/

from common import config as C                     # noqa: E402
from common.rl_common import common_args, run_testbench   # noqa: E402

# ---------------- TD3 hyper-parameters (shared values identical to DDPG / SAC) ----------------
HP = dict(
    learning_rate=3e-4,
    buffer_size=200_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=C.GAMMA,
    train_freq=1,
    gradient_steps=1,
    # TD3-specific
    action_noise_sigma=0.1,      # exploration noise, same as DDPG
    policy_delay=2,              # actor updated every 2 critic updates
    target_policy_noise=0.2,     # smoothing noise on the target action
    target_noise_clip=0.5,
)


def make_model(env, seed, timesteps, load=None):
    import numpy as np
    from stable_baselines3 import TD3
    from stable_baselines3.common.noise import NormalActionNoise
    if load:
        return TD3.load(load)
    hp = dict(HP)
    sigma = hp.pop("action_noise_sigma")
    noise = NormalActionNoise(mean=np.zeros(2), sigma=sigma * np.ones(2))
    return TD3("MlpPolicy", env, action_noise=noise, seed=seed, verbose=0,
               policy_kwargs=dict(net_arch=C.NET_ARCH), **hp)


if __name__ == "__main__":
    run_testbench("TD3", make_model, HP, HERE, common_args().parse_args())
