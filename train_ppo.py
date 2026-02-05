"""
Train PPO agent on CartPole-v1 using BordAX.

This script demonstrates:
- Setting up a Gymnax environment for fully JIT-compiled training
- Configuring a PPO agent with MLP policy and value networks
- Running training with checkpointing and evaluation
- Generating training curves visualization
"""
from bordax.training.logging import LoggerConfig
from bordax.training.checkpointing import CheckpointerConfig
from bordax.training.trainer import Trainer, TrainerConfig
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent

import jax
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import argparse
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent on CartPole-v1.")
    parser.add_argument(
        "--restore-last",
        action="store_true",
        help="Restore from the last checkpoint if available.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots.",
    )
    return parser.parse_args()


def plot_training_curves(eval_data, output_dir, algo_config, training_config):
    """Generate beautiful training curves from evaluation data."""
    sns.set_theme(style="darkgrid")

    # Extract rewards and compute statistics
    rewards = []
    lengths = []
    steps = []

    steps_per_checkpoint = training_config.epochs_per_checkpoint * algo_config["rollout_length"] * algo_config["num_envs"]

    for idx, data in enumerate(eval_data, start=1):
        if not data:
            continue
        returns = np.asarray(data.get("return", []), dtype=np.float32)
        lens = np.asarray(data.get("length", []), dtype=np.float32)
        if returns.size > 0:
            rewards.append(returns)
            lengths.append(lens)
            steps.append(idx * steps_per_checkpoint)

    if not rewards:
        print("No evaluation data to plot.")
        return

    # Compute mean and std for rewards
    reward_means = [np.mean(r) for r in rewards]
    reward_stds = [np.std(r) for r in rewards]
    length_means = [np.mean(l) for l in lengths]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Episode Rewards
    ax1 = axes[0]
    ax1.plot(steps, reward_means, color='#2ca02c', linewidth=2, label='Mean Reward')
    ax1.fill_between(
        steps,
        np.array(reward_means) - np.array(reward_stds),
        np.array(reward_means) + np.array(reward_stds),
        color='#2ca02c',
        alpha=0.25,
        label='±1 Std'
    )
    ax1.axhline(y=500, color='#d62728', linestyle='--', linewidth=1.5, label='Solved (500)')
    ax1.set_xlabel('Environment Steps', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('PPO Training on CartPole-v1', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 550)

    # Plot 2: Episode Length
    ax2 = axes[1]
    ax2.plot(steps, length_means, color='#1f77b4', linewidth=2)
    ax2.axhline(y=500, color='#d62728', linestyle='--', linewidth=1.5, label='Max Length')
    ax2.set_xlabel('Environment Steps', fontsize=12)
    ax2.set_ylabel('Episode Length', fontsize=12)
    ax2.set_title('Episode Length over Training', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 550)

    plt.tight_layout()

    # Save plots
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Training curves saved to: {plot_path}")

    # Also save a standalone rewards plot for README
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, reward_means, color='#2ca02c', linewidth=2.5, label='Mean Reward')
    ax.fill_between(
        steps,
        np.array(reward_means) - np.array(reward_stds),
        np.array(reward_means) + np.array(reward_stds),
        color='#2ca02c',
        alpha=0.3,
    )
    ax.axhline(y=500, color='#d62728', linestyle='--', linewidth=2, label='Solved (500)')
    ax.set_xlabel('Environment Steps', fontsize=14)
    ax.set_ylabel('Episode Reward', fontsize=14)
    ax.set_title('PPO on CartPole-v1 (Gymnax)', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 550)

    # Add final performance annotation
    if reward_means:
        final_reward = reward_means[-1]
        ax.annotate(
            f'Final: {final_reward:.0f}',
            xy=(steps[-1], final_reward),
            xytext=(-60, 20),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            color='#2ca02c',
            arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5)
        )

    plt.tight_layout()
    standalone_path = os.path.join(output_dir, 'evaluation_rewards_ppo.png')
    plt.savefig(standalone_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Standalone plot saved to: {standalone_path}")

    return reward_means, steps


if __name__ == "__main__":
    args = parse_args()

    print("=" * 70)
    print(" PPO - CartPole-v1 (BordAX)")
    print("=" * 70)

    # Create output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.dirname(os.path.abspath(__file__))

    if args.restore_last:
        # Find the most recent run directory
        runs_dir = os.path.join(root, "runs")
        all_runs = [d for d in os.listdir(runs_dir) if d.startswith("ppo_") and os.path.isdir(os.path.join(runs_dir, d))]
        if not all_runs:
            raise ValueError("No PPO runs found to restore from.")
        latest_run = max(all_runs)
        output_dir = os.path.join(runs_dir, latest_run)
        print(f"\n✓ Restoring from: {output_dir}")

        # Find the latest checkpoint number
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        all_ckpts = [int(d) for d in os.listdir(ckpt_dir) if d.isdigit()]
        if not all_ckpts:
            raise ValueError("No checkpoints found to restore from.")
        checkpoint_to_restore = max(all_ckpts)
        print(f"✓ Restoring from checkpoint: {checkpoint_to_restore}")
    else:
        checkpoint_to_restore = None
        output_dir = os.path.join(root, f"runs/ppo_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n✓ Output directory: {output_dir}")

    # Environment configuration
    env_name = "gymnax/CartPole-v1"
    env_config = {
        "init_config": {},
        "reset_config": {},
    }
    num_envs = 4  # Vectorized environments for faster training
    env = make_env(env_name, env_config, num_envs)
    eval_env = make_env(env_name, env_config, 1)

    print(f"\n✓ Environment: {env_name}")
    print(f"  - Observation space: {env.obs_space()}")
    print(f"  - Action space: {env.action_space()}")
    print(f"  - Num environments: {num_envs}")
    print(f"  - JIT-compiled: {env.is_jittable}")

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
        "lr": 3e-4,
        "rollout_length": 2048,
        "gamma": 0.99,
        "clip_schedule": lambda _: 0.2,
        "vf_schedule": lambda _: 0.5,
        "ent_schedule": lambda _: 0.01,
        "_lambda": 0.95,
        "num_minibatches": 32,
        "num_sgd_steps": 10,
        "num_envs": num_envs,
    }
    algorithm = make_algo(algo_name, algo_config)

    print(f"\n✓ Algorithm: {algo_name}")
    print(f"  - Learning rate: {algo_config['lr']}")
    print(f"  - Rollout length: {algo_config['rollout_length']}")
    print(f"  - Gamma: {algo_config['gamma']}")
    print(f"  - GAE lambda: {algo_config['_lambda']}")
    print(f"  - Clip epsilon: 0.2")
    print(f"  - Value coef: 0.5")
    print(f"  - Entropy coef: 0.01")
    print(f"  - Num minibatches: {algo_config['num_minibatches']}")
    print(f"  - SGD epochs: {algo_config['num_sgd_steps']}")

    # Logger and checkpointer configuration
    logger_config = LoggerConfig(
        log_dir=output_dir,
        use_wandb=False,
    )

    checkpointer_config = CheckpointerConfig(
        save_path=os.path.join(output_dir, "checkpoints"),
        interval=10,
    )

    # Training configuration
    training_config = TrainerConfig(
        num_checkpoints=50,
        epochs_per_checkpoint=1,
        evaluation_episodes=10,
        debug=True,
        logger_config=logger_config,
        checkpointer_config=checkpointer_config,
        restore_checkpoint=checkpoint_to_restore,
    )

    total_timesteps = (
        training_config.num_checkpoints
        * training_config.epochs_per_checkpoint
        * algo_config['rollout_length']
        * num_envs
    )

    print(f"\n✓ Training Configuration:")
    print(f"  - Checkpoints: {training_config.num_checkpoints}")
    print(f"  - Epochs per checkpoint: {training_config.epochs_per_checkpoint}")
    print(f"  - Total timesteps: {total_timesteps:,}")
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
    eval_data = trainer.run(key)
    jax.block_until_ready(eval_data)
    end_time = time.time()

    training_time = end_time - start_time

    print(f"\n{'='*70}")
    print(" Training Complete")
    print("="*70)
    print(f"✓ Training time: {training_time:.2f}s")
    print(f"✓ Throughput: {total_timesteps / training_time:,.0f} steps/s")

    # Generate plots
    if not args.no_plot:
        print(f"\n{'='*70}")
        print(" Generating Plots")
        print("="*70)
        reward_means, steps = plot_training_curves(eval_data, output_dir, algo_config, training_config)

        if reward_means:
            print(f"\n✓ Final mean reward: {reward_means[-1]:.1f}")
            print(f"✓ Best mean reward: {max(reward_means):.1f}")

    print(f"\n{'='*70}")
    print(" Summary")
    print("="*70)
    print(f"✓ Metrics logged to: {output_dir}/metrics.csv")
    print(f"✓ Evaluation logged to: {output_dir}/evaluation.csv")
    print(f"✓ Checkpoints saved to: {output_dir}/checkpoints/")
    if not args.no_plot:
        print(f"✓ Plots saved to: {output_dir}/")
    print("\n✅ PPO training completed successfully!")
