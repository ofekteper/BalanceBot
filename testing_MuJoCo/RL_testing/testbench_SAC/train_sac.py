"""
SAC testbench: Soft Actor-Critic (Haarnoja et al., 2018).

Stochastic policy that maximises reward + entropy (the entropy weight is tuned automatically),
twin critics like TD3. Explores by itself, no added action noise. The evaluation uses the
deterministic (mean) action.

    py train_sac.py --quick           # smoke test (~1-2 min)
    py train_sac.py                   # full run: 100k steps x 3 seeds
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # RL_testing/

from common import config as C                     # noqa: E402
from common.rl_common import common_args, run_testbench   # noqa: E402

# ---------------- SAC hyper-parameters (shared values identical to DDPG / TD3) ----------------
HP = dict(
    learning_rate=3e-4,
    buffer_size=200_000,
    learning_starts=5_000,
    batch_size=256,
    tau=0.005,
    gamma=C.GAMMA,
    train_freq=1,
    gradient_steps=1,
    # SAC-specific
    ent_coef="auto",             # entropy weight learned so the policy keeps a target entropy
    target_entropy="auto",       # = -dim(action) = -2
)


def make_model(env, seed, timesteps, load=None):
    from stable_baselines3 import SAC
    if load:
        return SAC.load(load)
    return SAC("MlpPolicy", env, seed=seed, verbose=0,
               policy_kwargs=dict(net_arch=C.NET_ARCH), **HP)


if __name__ == "__main__":
    run_testbench("SAC", make_model, HP, HERE, common_args().parse_args())
