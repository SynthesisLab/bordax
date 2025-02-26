import gymnax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from bordax.policies.utils import (
    policy_factory_mlp,
    make_actor_critic_mlp,
    make_actor_critic_dtsemnet,
    make_actor_critic_boolean,
    make_actor_critic_dtsemnet_value,
    make_actor_critic_dt_actions,
)
from bordax.algorithms.ppo.alg import train, PPOConfig

import pandas as pd
from scipy.stats import sem, t
import time


if __name__ == "__main__":
    env, env_params = gymnax.make("CartPole-v1")

    results = []

    architectures = [
        # make_actor_critic_dtsemnet_value, 
        # make_actor_critic_mlp, 
        make_actor_critic_dtsemnet, 
        # make_actor_critic_boolean, 
        ]

    for architecture in architectures:
        for seed in range(700, 720):
            start_time = time.time()
            config = PPOConfig(
                learning_rate=2.4e-3,
                num_checkpoints=200,          # number of evaluations
                epochs_per_checkpoint=1,    # number of epochs between evaluations
                unroll_length=832,         # length of the experience buffer
                sgd_steps=1,                # num of sgd passes through the same experience buffer
                epoch_steps=1,            # num of training steps per epoch (the number of rollouts per epoch???)
                num_minibatches=26,          # num of minibatches for each sgd pass
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
        result["average"] = jnp.mean(result["checkpoints"], axis=1)

    # save averages to a txt file
    for result in results:

        rew = np.array(result["average"]) #only first 1000 episodes

        reward_fp = f'logslr/'
        fn = f"{reward_fp}seed{result["seed"]}_rewards.txt"
        with open(fn, "w") as file:
            np.savetxt(fn, rew, fmt='%.1f')

    # Convert to DataFrame
    df = pd.DataFrame({
        "architecture": [d["architecture"] for d in results],
        "checkpoints": [d["average"] for d in results]
    })

    # Expand lists into separate columns
    checkpoint_df = pd.DataFrame(df["checkpoints"].to_list())
    checkpoint_df["architecture"] = df["architecture"]

    # Compute mean and SEM explicitly for each time step
    grouped = checkpoint_df.groupby("architecture").agg(['mean', 'sem'])

    # Fix for column names: Ensure we access the correct multi-indexed columns
    mean_values = grouped.xs('mean', axis=1, level=1)
    sem_values = grouped.xs('sem', axis=1, level=1)

    # Compute confidence interval
    confidence_level = 0.95
    n = len(results) // len(mean_values)  # Number of runs per architecture
    t_value = t.ppf((1 + confidence_level) / 2, df=n - 1)

    ci_values = sem_values * t_value  # 95% Confidence Interval

    # Plot
    plt.figure(figsize=(8, 5))

    for arch in mean_values.index:
        time_steps = range(mean_values.shape[1])
        mean = mean_values.loc[arch]
        ci = ci_values.loc[arch]

        plt.plot(time_steps, mean, label=f"Architecture {arch}", marker='o')  # Mean curve
        # plt.fill_between(time_steps, mean - ci, mean + ci, alpha=0.2)  # Shaded CI

    # Formatting
    plt.xlabel("Rollouts")
    plt.ylabel("Metric Value")
    plt.title("Experiment Evolution Over Time")
    plt.legend()
    plt.grid(True)

    plt.show()