import gymnax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from bordax.policies.utils import policy_factory_mlp
from bordax.policies.mlp import make_actor_critic_mlp
from bordax.policies.dtsemnet import make_actor_critic_dtsemnet
from bordax.policies.boolean import make_actor_critic_boolean

from bordax.algorithms.ppo.alg import train, PPOConfig

from bordax.environments.pomdp.parser import parse
from bordax.environments.pomdp.pomdp import jPOMDP, BeliefWrapper


import pandas as pd
from scipy.stats import sem, t
import time
import os

if __name__ == "__main__":
    env_name = ""
    with open("environments/1d.POMDP") as f:
        pomdp = parse(f.read())
    env = jPOMDP(pomdp)
    env = BeliefWrapper(env)
    env_params = {"cutoff": 100}

    results = []

    architectures = [
        # make_actor_critic_dtsemnet_value, 
        make_actor_critic_mlp, 
        make_actor_critic_dtsemnet, 
        make_actor_critic_boolean, 
        ]

    for architecture in architectures:
        for seed in range(700, 701):
            start_time = time.time()
            config = PPOConfig(
                learning_rate=2.4e-3,
                num_checkpoints=100,          # number of evaluations
                epochs_per_checkpoint=1,    # number of epochs between evaluations
                unroll_length=1024,         # length of the experience buffer
                sgd_steps=1,                # num of sgd passes through the same experience buffer
                epoch_steps=1,            # num of training steps per epoch (the number of rollouts per epoch???)
                num_minibatches=16,          # num of minibatches for each sgd pass
                num_envs=1,                 # num of parallel environments
                seed=seed,
                epsilon=0.2,
                gamma=0.99,
                vf_coef=0.5,
                entropy_coef=0.02,
                gae_lambda=0.86,
                normalize_advantage=True,
                max_grad_norm=0.5,
                debug=True,
                env_jitable=True
            )
            training_state, checkpoints = train(env, env_params, architecture, policy_factory_mlp, config)
            checkpoints.block_until_ready()
            end_time = time.time()

            results.append({
                "architecture": architecture.__name__,
                "seed": seed,
                "training_time": end_time - start_time,
                "checkpoints": checkpoints
            })

    for result in results:
        print(result["training_time"])

    # plot the averges of chechpoints with architectures differentiated with color
    for result in results:
        result["average"] = np.mean(np.asarray(result["checkpoints"]), axis=1)

    # save averages to a txt file
    for result in results:

        rew = result["average"]

        reward_fp = f'logs/pomdp/{result['architecture']}/'
        # check if the folder exists
        if not os.path.exists(reward_fp):
            os.makedirs(reward_fp)

        fn = f"{reward_fp}seed{result["seed"]}_rewards.txt"
        with open(fn, "w") as file:
            np.savetxt(fn, rew, fmt='%.1f')
