from bordax.trainer import Trainer, TrainerConfig
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent

import jax
import time
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    print("=" * 70)
    print(" DQN - CartPole-v1")
    print("=" * 70)
    
    # Environment configuration
    env_name = "gymnax/CartPole-v1"
    env_config = {
        "init_config": {},
        "reset_config": {}, 
    }
    num_envs = 1  # DQN typically uses single environment
    env = make_env(env_name, env_config, num_envs)
    eval_env = make_env(env_name, env_config, num_envs)
    
    print(f"\n✓ Environment: {env_name}")
    print(f"  - Observation space: {env.obs_space()}")
    print(f"  - Action space: {env.action_space()}")
    print(f"  - Num environments: {num_envs}")

    # Agent configuration
    agent_name = "dqn/mlp"
    agent_config = {
        "q_layers": [120, 84],  # Q-network architecture
    }
    agent = make_agent(agent_name, env, agent_config)
    
    print(f"\n✓ Agent: {agent_name}")
    print(f"  - Q-network layers: {agent_config['q_layers']}")

    # Algorithm configuration
    algo_name = "dqn"
    algo_config = {
        "epsilon_schedule": lambda t: max(0.01, 0.9 * (0.995 ** t)),  # Epsilon decay
        "rollout_length": 1,         # Collect 1 step at a time
        "batch_size": 128,           # Batch size for updates
        "gamma": 0.99,               # Discount factor
        "lr": 2.5e-4,                # Learning rate
        "target_update_freq": 500,   # Update target every 500 steps
    }
    algorithm = make_algo(algo_name, algo_config)
    
    print(f"\n✓ Algorithm: {algo_name}")
    print(f"  - Epsilon schedule: decaying from 0.9 to 0.01")
    print(f"  - Batch size: {algo_config['batch_size']}")
    print(f"  - Learning rate: {algo_config['lr']}")
    print(f"  - Gamma: {algo_config['gamma']}")
    print(f"  - Target update frequency: {algo_config['target_update_freq']}")

    # Training configuration
    # For DQN: each epoch collects 'rollout_length' steps and performs 1 update
    # num_checkpoints controls how many evaluations we do
    training_config = TrainerConfig(
        num_checkpoints=200,         # Number of evaluation points
        epochs_per_checkpoint=250,   # Updates between evaluations (250 updates = 250 steps collected)
        evaluation_episodes=10,      # Episodes per evaluation
        debug=True,
        save_model=False,
        # Off-policy specific
        replay_buffer_capacity=10000,
        warmup_steps=1000,           # Fill buffer before training starts
    )
    
    print(f"\n✓ Training Configuration:")
    print(f"  - Replay buffer capacity: {training_config.replay_buffer_capacity}")
    print(f"  - Warmup steps: {training_config.warmup_steps}")
    print(f"  - Checkpoints: {training_config.num_checkpoints}")
    print(f"  - Epochs per checkpoint: {training_config.epochs_per_checkpoint}")
    print(f"  - Total training steps: {training_config.num_checkpoints * training_config.epochs_per_checkpoint}")
    print(f"  - Evaluation episodes: {training_config.evaluation_episodes}")

    # Initialize the trainer
    trainer = Trainer(env, eval_env, agent, algorithm, training_config)
    key = jax.random.PRNGKey(0)
    init_key, key = jax.random.split(key)
    
    print(f"\n{'='*70}")
    print(" Initializing Trainer")
    print("="*70)
    trainer.init(init_key)

    # Run training
    print(f"\n{'='*70}")
    print(" Starting Training")
    print("="*70)
    start_time = time.time()
    metrics, data, model_parameters = trainer.run(key)
    end_time = time.time()
    
    print(f"\n{'='*70}")
    print(" Training Complete")
    print("="*70)
    print(f"Training time: {end_time - start_time:.2f}s")
    print(f"Total updates: {len(metrics)}")
    
    if metrics:
        print(f"Initial loss: {float(metrics[0]['dqn_loss']):.4f}")
        print(f"Final loss: {float(metrics[-1]['dqn_loss']):.4f}")
        print(f"Final Q-value: {float(metrics[-1]['mean_q_value']):.4f}")

    # Compute average evaluation rewards per checkpoint
    average_evaluation_rewards = []
    for rollout in data:
        first_done_indices = np.argmax(rollout["done"], axis=1)
        cum_sum = np.cumsum(rollout["reward"], axis=1)
        first_rewards = cum_sum[
            np.arange(cum_sum.shape[0]),
            first_done_indices,
        ]
        average_evaluation_rewards.append(first_rewards.mean())

    # Find the checkpoint with highest average evaluation reward
    average_evaluation_rewards = np.array(average_evaluation_rewards)
    best_checkpoint_index = np.argmax(average_evaluation_rewards)
    best_parameters = model_parameters[best_checkpoint_index]
    
    print(f"\nBest checkpoint: {best_checkpoint_index}")
    print(f"Best average reward: {average_evaluation_rewards[best_checkpoint_index]:.2f}")

    print("\n✅ DQN training completed successfully!")
    
    # Optional: Plot training metrics
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    
    # Evaluation rewards over checkpoints
    plt.plot(average_evaluation_rewards)
    plt.axhline(y=average_evaluation_rewards.max(), color='r', linestyle='--', 
                       label=f'Best: {average_evaluation_rewards.max():.1f}')
    plt.xlabel('Checkpoint')
    plt.ylabel('Average Reward')
    plt.title('Evaluation Performance')
    plt.legend()
    plt.grid(True)
        
    plt.tight_layout()
    plt.savefig('dqn_training_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Training plots saved to 'dqn_training_results.png'")
    plt.show()
