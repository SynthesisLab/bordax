from bordax.utils import Trainer
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent

import jax
import time
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # Initialize the environment, architecture, and algorithm

    do_plots = True

    training_config = {
        "num_checkpoints": 100,
        "epochs_per_checkpoint": 2,
        "evaluation_episodes": 16,
        "debug": True,
        "timing": True,
        "save_model": False,
        # "log_interval": 10,
    }

    agent_config = {
        "policy_layers": [128, 64, 32],
        "value_layers": [128, 64, 32],
    }
    env_config = {}
    algo_config = {
        "lr": 1e-4,
        "rollout_length": 1024,
        "gamma": 0.99,
        "_lambda": 0.95,
        "num_minibatches": 16,
        "num_sgd_steps": 10,
    }

    env_name = "gymnasium/LunarLander-v3"  # Replace with your environment
    agent_name = "mlp"  # Replace with your agent
    algo_name = "ppo"  # Replace with your algorithm

    env = make_env(env_name)
    eval_env = make_env(env_name, num_envs=1)
    agent = make_agent(agent_name, agent_config)
    algorithm = make_algo(algo_name, algo_config)  # Replace with your algorithm

    # Initialize the trainer
    trainer = Trainer(env, eval_env, agent, algorithm, training_config)
    key = jax.random.PRNGKey(0)  # Random key for JAX
    trainer.init(key)

    start_time = time.time()
    metrics, data = trainer.run(key)
    end_time = time.time()
    print(f"Training time: {end_time - start_time}")

    average_evaluation_rewards = []
    for rollout in data:
        first_done_indices = np.argmax(rollout["done"], axis=1)
        cum_sum = np.cumsum(rollout["reward"], axis=1)
        first_rewards = cum_sum[
            np.arange(cum_sum.shape[0]),
            first_done_indices,
        ]
        average_evaluation_rewards.append(first_rewards.mean())

    if do_plots:
        plt.plot(average_evaluation_rewards)
        plt.xlabel("Checkpoint")
        plt.ylabel("Average Evaluation Reward")
        plt.title("Average Evaluation Reward Over Checkpoints")
        plt.show()
