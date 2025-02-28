import gymnax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from bordax.policies.utils import policy_factory_mlp, q_function_factory
from bordax.policies.mlp import make_actor_critic_mlp, make_q_mlp
from bordax.policies.dtsemnet import make_actor_critic_dtsemnet
from bordax.policies.boolean import make_actor_critic_boolean

from bordax.algorithms.dqn.alg import train, DQNConfig

import pandas as pd
from scipy.stats import sem, t
import time
import os

if __name__ == "__main__":
    env, env_params = gymnax.make("CartPole-v1")

    results = []

    architectures = [
        # make_actor_critic_dtsemnet_value, 
        make_q_mlp, 
        # make_actor_critic_dtsemnet, 
        # make_actor_critic_boolean, 
        ]

    for architecture in architectures:
        for seed in range(700, 730):
            start_time = time.time()
            config = DQNConfig(seed=seed)
            training_state, checkpoints = train(env, env_params, architecture, q_function_factory, config)
            checkpoints.block_until_ready()
            end_time = time.time()
            print(f"Training time: {end_time - start_time}")
            results.append({
                "architecture": architecture.__name__,
                "seed": seed,
                "training_time": end_time - start_time,
                "checkpoints": checkpoints
            })

    # for result in results:
    #     print(result["training_time"])

    # plot the averges of chechpoints with architectures differentiated with color
    for result in results:
        result["average"] = np.mean(np.asarray(result["checkpoints"]), axis=1)

    # save averages to a txt file
    for result in results:

        rew = result["average"]

        reward_fp = f'logs/main_comparison/{result['architecture']}/'
        # check if the folder exists
        if not os.path.exists(reward_fp):
            os.makedirs(reward_fp)

        fn = f"{reward_fp}seed{result["seed"]}_rewards.txt"
        with open(fn, "w") as file:
            np.savetxt(fn, rew, fmt='%.1f')