import gymnasium
import gymnax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from bordax.policies.utils import policy_factory_mlp
from bordax.policies.mlp import make_actor_critic_mlp
from bordax.policies.dtsemnet import make_actor_critic_dtsemnet
from bordax.policies.boolean import make_actor_critic_boolean

from bordax.algorithms.ppo.alg import train, PPOConfig

import os
import time


if __name__ == "__main__":
    flag = True
    if flag:
        env = gymnasium.make_vec("LunarLander-v3", num_envs=1, vectorization_mode="sync", vector_kwargs={"autoreset_mode": "NextStep"})
        env_params = {}
    else:
        env, env_params = gymnax.make("CartPole-v1")

    results = []

    architectures = [make_actor_critic_dtsemnet]

    for architecture in architectures:
        for seed in range(5):
            start_time = time.time()
            config = PPOConfig(
                learning_rate=2.4e-3,
                num_checkpoints=500,          # number of evaluations
                epochs_per_checkpoint=10,    # number of epochs between evaluations
                unroll_length=1024,         # length of the experience buffer
                sgd_steps=4,                # num of sgd passes through the same experience buffer
                epoch_steps=1,            # num of training steps per epoch
                num_minibatches=16,          # num of minibatches for each sgd pass
                num_envs=1,                 # num of parallel environments
                seed=seed,
                epsilon=0.2,
                gamma=0.999,
                vf_coef=0.5,
                entropy_coef=0.01,
                gae_lambda=0.98,
                normalize_advantage=True,
                max_grad_norm=0.5,
                debug=True,
                env_jitable = False if isinstance(env, gymnasium.vector.VectorEnv) else True
            )
            training_state, checkpoints = train(env, env_params, architecture, policy_factory_mlp, config)
            end_time = time.time()

            results.append({
                "architecture": architecture.__name__,
                "seed": seed,
                "training_time": end_time - start_time,
                "checkpoints": checkpoints,
                "average": jnp.mean(checkpoints, axis=1)
            })
            print(f"Seed {seed} took {end_time - start_time} seconds")
            
            reward_fp = f'logs/lunarlander/{architecture.__name__}'
            if not os.path.exists(reward_fp):
                os.makedirs(reward_fp)
            fn = f"{reward_fp}seed{seed}_rewards.txt"
            with open(fn, "w") as file:
                np.savetxt(fn, jnp.mean(checkpoints, axis=1), fmt='%.1f')
