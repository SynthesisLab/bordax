from bordax.trainer import Trainer, TrainerConfig
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent

import jax
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from datetime import datetime

if __name__ == "__main__":
    print("=" * 70)
    print(" PPO - CartPole-v1")
    print("=" * 70)
    
    # Create output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/ppo_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")
    
    # Environment configuration
    env_name = "gymnax/CartPole-v1"
    env_config = {
        "init_config": {},
        "reset_config": {}, 
    }
    num_envs = 1
    env = make_env(env_name, env_config, num_envs)
    eval_env = make_env(env_name, env_config, num_envs)
    
    print(f"\n✓ Environment: {env_name}")
    print(f"  - Observation space: {env.obs_space()}")
    print(f"  - Action space: {env.action_space()}")
    print(f"  - Num environments: {num_envs}")

    # Agent configuration
    agent_name = "mlp/mlp"
    agent_config = {
        "policy_layers": [128, 128, 64],
        "value_layers": [128, 128, 64],
    }
    agent = make_agent(agent_name, env, agent_config)
    
    print(f"\n✓ Agent: {agent_name}")
    print(f"  - Policy layers: {agent_config['policy_layers']}")
    print(f"  - Value layers: {agent_config['value_layers']}")

    # Algorithm configuration
    algo_name = "ppo"
    algo_config = {
        "lr": 1e-5,
        "rollout_length": 2048,
        "gamma": 0.99,
        "clip_schedule": lambda _: 0.2,
        "vf_schedule": lambda _: 0.5,
        "ent_schedule": lambda _: 0.01,
        "_lambda": 0.95,
        "num_minibatches": 16,
        "num_sgd_steps": 10,
    }
    algorithm = make_algo(algo_name, algo_config)
    
    print(f"\n✓ Algorithm: {algo_name}")
    print(f"  - Learning rate: {algo_config['lr']}")
    print(f"  - Rollout length: {algo_config['rollout_length']}")
    print(f"  - Gamma: {algo_config['gamma']}")
    print(f"  - GAE lambda: {algo_config['_lambda']}")
    print(f"  - Clip epsilon: 0.2 (constant)")
    print(f"  - Value coef: 0.5 (constant)")
    print(f"  - Entropy coef: 0.01 (constant)")
    print(f"  - Num minibatches: {algo_config['num_minibatches']}")
    print(f"  - SGD epochs: {algo_config['num_sgd_steps']}")

    # Training configuration
    training_config = TrainerConfig(
        num_checkpoints=400,
        epochs_per_checkpoint=1,
        evaluation_episodes=10,
        debug=True,
        save_model=True,
    )
    
    print(f"\n✓ Training Configuration:")
    print(f"  - Checkpoints: {training_config.num_checkpoints}")
    print(f"  - Epochs per checkpoint: {training_config.epochs_per_checkpoint}")
    print(f"  - Total rollouts: {training_config.num_checkpoints * training_config.epochs_per_checkpoint}")
    print(f"  - Total timesteps: {training_config.num_checkpoints * training_config.epochs_per_checkpoint * algo_config['rollout_length']}")
    print(f"  - Evaluation episodes: {training_config.evaluation_episodes}")
    print(f"  - Save model: {training_config.save_model}")

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
    print(f"Total checkpoints: {len(metrics)}")
    
    if metrics:
        # PPO metrics have shape (num_sgd_steps, num_minibatches) due to nested scans
        # Take mean across both dimensions for reporting
        print(f"Initial total loss: {float(metrics[0]['total_loss'].mean()):.4f}")
        print(f"Final total loss: {float(metrics[-1]['total_loss'].mean()):.4f}")
        print(f"Initial value loss: {float(metrics[0]['value_loss'].mean()):.4f}")
        print(f"Final value loss: {float(metrics[-1]['value_loss'].mean()):.4f}")

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

    # Save the parameters
    if training_config.save_model:
        export = {"agent": agent, "params": best_parameters}
        model_path = os.path.join(output_dir, "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(export, f)
        print(f"\n✓ Model saved to '{model_path}'")

    # Save metrics to file
    metrics_path = os.path.join(output_dir, "metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    print(f"✓ Metrics saved to '{metrics_path}'")
    
    # Save evaluation rewards
    rewards_path = os.path.join(output_dir, "evaluation_rewards.npy")
    np.save(rewards_path, average_evaluation_rewards)
    print(f"✓ Evaluation rewards saved to '{rewards_path}'")

    print("\n✅ PPO training completed successfully!")
    
    # Plot training metrics
    import seaborn as sns
    sns.set_theme(style="darkgrid")
    
    # Plot 1: Evaluation rewards
    plt.figure(figsize=(8, 6))
    plt.plot(average_evaluation_rewards, marker='o', markersize=3)
    plt.axhline(y=average_evaluation_rewards.max(), color='r', linestyle='--', 
                label=f'Best: {average_evaluation_rewards.max():.1f}')
    plt.xlabel('Checkpoint')
    plt.ylabel('Average Reward')
    plt.title('Evaluation Performance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    eval_plot_path = os.path.join(output_dir, "evaluation_rewards.png")
    plt.savefig(eval_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Policy loss (surrogate loss)
    plt.figure(figsize=(8, 6))
    policy_losses = [m['loss'].mean() for m in metrics]
    plt.plot(policy_losses, alpha=0.7)
    plt.xlabel('Checkpoint')
    plt.ylabel('Policy Loss')
    plt.title('Policy Loss Over Training')
    plt.grid(True)
    plt.tight_layout()
    policy_plot_path = os.path.join(output_dir, "policy_loss.png")
    plt.savefig(policy_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Value loss
    plt.figure(figsize=(8, 6))
    value_losses = [m['value_loss'].mean() for m in metrics]
    plt.plot(value_losses, alpha=0.7)
    plt.xlabel('Checkpoint')
    plt.ylabel('Value Loss')
    plt.title('Value Loss Over Training')
    plt.grid(True)
    plt.tight_layout()
    value_plot_path = os.path.join(output_dir, "value_loss.png")
    plt.savefig(value_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Entropy loss (negative entropy)
    plt.figure(figsize=(8, 6))
    entropy_losses = [m['entropy_loss'].mean() for m in metrics]
    plt.plot(entropy_losses, alpha=0.7)
    plt.xlabel('Checkpoint')
    plt.ylabel('Entropy Loss')
    plt.title('Entropy Loss Over Training')
    plt.grid(True)
    plt.tight_layout()
    entropy_plot_path = os.path.join(output_dir, "entropy_loss.png")
    plt.savefig(entropy_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plots saved to '{output_dir}/':")
    print(f"  - {os.path.basename(eval_plot_path)}")
    print(f"  - {os.path.basename(policy_plot_path)}")
    print(f"  - {os.path.basename(value_plot_path)}")
    print(f"  - {os.path.basename(entropy_plot_path)}")

