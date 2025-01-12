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
    env, env_params = gymnax.make("CartPole-v1")

    results = []

    architectures = [make_actor_critic_mlp, make_actor_critic_dtsemnet, make_actor_critic_boolean]

    for architecture in architectures:
        for seed in range(10):
            start_time = time.time()
            config = PPOConfig(
                learning_rate=2.5e-3,
                num_checkpoints=100,          # number of evaluations
                epochs_per_checkpoint=1,    # number of epochs between evaluations
                unroll_length=1024,         # length of the experience buffer
                sgd_steps=4,                # num of sgd passes through the same experience buffer
                epoch_steps=4,            # num of training steps per epoch
                num_minibatches=4,          # num of minibatches for each sgd pass
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

    color_map = plt.get_cmap('tab10')
    colors = color_map(np.linspace(0,1,len(architectures)))

    label_to_color = {label: colors[i] for i, label in enumerate([a.__name__ for a in architectures])}

    # plot the averges of chechpoints with architectures differentiated with color
    for result in results:
        plt.plot(jnp.mean(result["checkpoints"], axis=1), label=result["architecture"], color=label_to_color[result["architecture"]])
    plt.legend([a.__name__ for a in architectures])
    plt.show()