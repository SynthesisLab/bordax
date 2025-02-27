import gymnasium
import gymnax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from bordax.policies.utils import (
    policy_factory_mlp,
    make_actor_critic_mlp,
    make_actor_critic_dtsemnet,
    make_actor_critic_boolean,
)
from bordax.algorithms.ppo.alg import train, PPOConfig


import time


if __name__ == "__main__":
    flag = True
    if flag:
        env = gymnasium.make_vec("CartPole-v1", num_envs=1, vectorization_mode="sync", vector_kwargs={"autoreset_mode": "NextStep"})
        env_params = {}
    else:
        env, env_params = gymnax.make("CartPole-v1")

    results = []

    architectures = [make_actor_critic_mlp]

    for architecture in architectures:
        for seed in range(1):
            start_time = time.time()
            config = PPOConfig(
                learning_rate=2.4e-3,
                num_checkpoints=200,          # number of evaluations
                epochs_per_checkpoint=1,    # number of epochs between evaluations
                unroll_length=832,         # length of the experience buffer
                sgd_steps=1,                # num of sgd passes through the same experience buffer
                epoch_steps=1,            # num of training steps per epoch
                num_minibatches=26,          # num of minibatches for each sgd pass
                num_envs=1,                 # num of parallel environments
                seed=seed,
                epsilon=0.2,
                gamma=0.99,
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
                "checkpoints": checkpoints
            })
            print(f"Seed {seed} took {end_time - start_time} seconds")

    for result in results:
        result["average"] = jnp.mean(result["checkpoints"], axis=1)

    # save averages to a txt file
    for result in results:

        rew = np.array(result["average"])

        reward_fp = f'logs/gym/'
        fn = f"{reward_fp}seed{result["seed"]}_rewards.txt"
        with open(fn, "w") as file:
            np.savetxt(fn, rew, fmt='%.1f')
