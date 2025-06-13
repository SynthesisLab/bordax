from bordax.utils import Trainer
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent

import jax
import time
import matplotlib.pyplot as plt
import numpy as np

from plots import visualize_training_metrics

import pickle

if __name__ == "__main__":
    # Initialize the environment, architecture, and algorithm

    do_plots = True

    training_config = {
        "num_checkpoints": 400,
        "epochs_per_checkpoint": 1,
        "evaluation_episodes": 32,
        "debug": True,
        "save_model": True,
        # "log_interval": 10,
    }

    agent_config_mlp = {
        "policy_layers": [128, 64, 32],
        "value_layers": [128, 64, 32],
    }
    agent_config_dt = {"tree_depth": 4, "value_layers": [128, 64, 32]}
    agent_config_bool = {"n": 4, "value_layers": [128, 64, 32]}

    env_config = {}
    algo_config = {
        "lr": 1e-4,
        "rollout_length": 2048,
        "gamma": 0.99,
        "_lambda": 0.95,
        "num_minibatches": 32,
        "num_sgd_steps": 10,
    }

    env_name = "gymnasium/LunarLander-v3"  # Replace with your environment
    agent_name = "mlp/dt"  # Replace with your agent
    algo_name = "ppo"  # Replace with your algorithm

    env = make_env(env_name, 1)
    eval_env = make_env(env_name, num_envs=1)
    agent = make_agent(agent_name, env, agent_config_dt)
    algorithm = make_algo(algo_name, algo_config)  # Replace with your algorithm

    # Initialize the trainer
    trainer = Trainer(env, eval_env, agent, algorithm, training_config)
    key = jax.random.PRNGKey(0)  # Random key for JAX
    trainer.init(key)

    start_time = time.time()
    metrics, data, model_parameters = trainer.run(key)
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

    # find the checkpoint with highers average evaluation reward
    average_evaluation_rewards = np.array(average_evaluation_rewards)
    best_checkpoint_index = np.argmax(average_evaluation_rewards)
    best_parameters = model_parameters[best_checkpoint_index]

    # save the parameters
    if training_config["save_model"]:
        export = {"agent": agent, "params": best_parameters}
        with open("full_model.pkl", "wb") as f:
            pickle.dump(export, f)
        print("Model saved as 'best_model.pkl'")

    if do_plots:

        import seaborn as sns

        visualize_training_metrics(metrics, num_checkpoints=100, num_epochs=1)

        sns.set_theme(style="darkgrid")
        plt.figure(figsize=(8, 4))
        sns.lineplot(x=np.arange(len(average_evaluation_rewards)), y=average_evaluation_rewards)
        plt.xlabel("Checkpoint")
        plt.ylabel("Average Evaluation Reward")
        plt.title("Average Evaluation Reward Over Checkpoints")
        plt.show()
