import gymnax
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from bordax.policies.utils import policy_factory_mlp
from bordax.policies.mlp import make_policy_value_mlp

from bordax.algorithms.ppo.alg import train, PPOConfig

import time
import os
from tqdm import trange

if __name__ == "__main__":
    env, env_params = gymnax.make("CartPole-v1")
    # env= gym.make_vec("LunarLander-v3")
    # env_params = {}

    results = []

    architectures = [
        make_policy_value_mlp,
    ]

    for architecture in architectures:

        n_seeds = 1
        init_seed = 800
        iterator = trange(n_seeds, desc="Seeds")

        for seed in iterator:
            start_time = time.time()
            config = PPOConfig(
                learning_rate=0.002,
                num_checkpoints=200,  # number of evaluations
                epochs_per_checkpoint=1,  # number of epochs between evaluations
                unroll_length=1024,  # length of the experience buffer
                sgd_steps=10,  # num of sgd passes through the same experience buffer
                epoch_steps=1,  # num of training steps per epoch (the number of rollouts per epoch???)
                num_minibatches=16,  # num of minibatches for each sgd pass
                num_envs=1,  # num of parallel environments
                seed=init_seed + seed,
                epsilon=0.2,
                gamma=0.99,
                vf_coef=0.5,
                entropy_coef=0.01,
                gae_lambda=0.85,
                normalize_advantage=True,
                max_grad_norm=0.5,
                debug=True,
                env_jitable=True,
            )
            training_state, checkpoints = train(
                env, env_params, architecture, policy_factory_mlp, config
            )
            checkpoints.block_until_ready()
            end_time = time.time()
            print(f"Training time: {end_time - start_time}")
            results.append(
                {
                    "architecture": architecture.__name__,
                    "seed": seed,
                    "training_time": end_time - start_time,
                    "checkpoints": checkpoints,
                    "average": jnp.mean(checkpoints, axis=1),
                }
            )

            print(f"Seed {seed} took {end_time - start_time} seconds")
            # print(info)
            reward_fp = f"logs/cart_pole/{architecture.__name__}/"
            if not os.path.exists(reward_fp):
                os.makedirs(reward_fp)
            fn = f"{reward_fp}seed{seed}_rewards.txt"
            with open(fn, "w") as file:
                np.savetxt(fn, jnp.mean(checkpoints, axis=1), fmt="%.1f")
