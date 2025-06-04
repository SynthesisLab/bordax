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

    training_config = {
        "num_checkpoints": 200,
        "epochs_per_checkpoint": 1,
        "evaluation_episodes": 8,
        "debug": True,
        "save_model": False,
        # "log_interval": 10,
    }

    agent_config = {"policy_layers": [10],
                    "value_layers": [32, 32,]}
    env_config = {}
    algo_config = {"rollout_length": 1024, 
                   "gamma": 0.99,
                   "num_minibatches": 16,
                   "num_sdg_steps": 1,
                   }

    env_name = "gymnax/CartPole-v1"  # Replace with your environment
    agent_name = "mlp"  # Replace with your agent
    algo_name= "ppo" # Replace with your algorithm

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
    rollouts = data[0]
    end_time = time.time()
    print(f"Training time: {end_time - start_time}")

    # print(f"{rollouts["obs"].shape=}")
    # print(f"{rollouts["state"]=}")
    # print(f"{rollouts["action"]=}")
    # print(f"{rollouts["reward"][:10]=}")
    # print(f"{rollouts["done"][:10]=}")
    # print(f"{rollouts["info"]}")

    first_done_indices = np.argmax(rollouts["done"], axis=1)
    cum_sum = np.cumsum(rollouts["reward"], axis=1)
    first_rewards = cum_sum[
        np.arange(cum_sum.shape[0]),
        first_done_indices,
    ]
    print(first_rewards)
    print(np.average(first_rewards))

    # # # plot the rewards
    # rewards = np.array(rewards)
    # plt.plot(np.arange(len(rewards)), rewards.mean(axis=1))
    # plt.show()

    # values = np.array(values)

    # T, N = np.array(values).shape

    # x = np.linspace(0, 1, N)         # input axis
    # t = np.arange(T)                 # time axis (e.g., checkpoint indices)
    # X, T_ = np.meshgrid(x, t)

    # # Create the figure and 3D axes
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot the surface
    # surf = ax.plot_surface(X, T_, values, cmap='viridis', edgecolor='none')

    # # Labels and formatting
    # ax.set_xlabel('Input x ∈ [0, 1]')
    # ax.set_ylabel('Training step')
    # ax.set_zlabel('Model output')
    # ax.set_title('3D Surface of Model Output Over Time')

    # # Add a color bar
    # fig.colorbar(surf, shrink=0.5, aspect=10, label='Output Value')

    # plt.tight_layout()
    # plt.show()

    # plt.imshow(np.array(values), aspect='auto', cmap='viridis', origin='lower',)
    # plt.colorbar(label="Model output")

    # plt.show()
