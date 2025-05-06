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
        "evaluation_episodes": 32,
        "debug": False,
        "save_model": False,
        # "log_interval": 10,
    }

    agent_config = {"policy_layers": [10,],
                    "value_layers": [32, 32,]}
    env_config = {}
    algo_config = {"rollout_length": 1024, 
                   "gamma": 0.99,
                   "num_minibatches": 16,
                   "num_sdg_steps": 1,
                   }

    env_name = "CartPole-v1"  # Replace with your environment
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
    metrics, rewards = trainer.run(key)
    rewards.block_until_ready()
    end_time = time.time()
    print(f"Training time: {end_time - start_time}")

    # plot the rewards
    rewards = np.array(rewards)
    plt.plot(np.arange(len(rewards)), rewards.mean(axis=1))

    plt.show()
