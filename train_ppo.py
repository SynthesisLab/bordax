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
import pickle
import os
from datetime import datetime
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description="Train PPO agent on a Gymnax environment.")

    parser.add_argument(
        "--restore-last",
        action="store_true",
        help="Restore from the last checkpoint if available.",
    )

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    print("=" * 70)
    print(" PPO - CartPole-v1")
    print("=" * 70)
    
    # Create output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # get absolute path
    root = os.path.dirname(os.path.abspath(__file__))
    if args.restore_last:
        # find the most recent run directory
        runs_dir = os.path.join(root, "runs")
        all_runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if not all_runs:
            raise ValueError("No runs found to restore from.")
        latest_run = max(all_runs)
        output_dir = os.path.join(runs_dir, latest_run)
        print(f"\n✓ Restoring from the latest run directory: {output_dir}")

        # find the latest checkpoint number
        ckpt_dir = os.path.join(output_dir, "checkpoints")
        all_ckpts = [int(d) for d in os.listdir(ckpt_dir)]
        if not all_ckpts:
            raise ValueError("No checkpoints found to restore from.")
        checkpoint_to_restore = max(all_ckpts)
        print(f"✓ Restoring from checkpoint number: {checkpoint_to_restore}")
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
    num_envs = 1
    env = make_env(env_name, env_config, num_envs)
    eval_env = make_env(env_name, env_config, 1)  # Single environment for evaluation
    
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
        "num_envs": num_envs,
    }
    ROLLOUT_TOTAL = algo_config["rollout_length"] * algo_config["num_envs"]
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
        chekpointer_config=checkpointer_config,
        restore_checkpoint=checkpoint_to_restore if args.restore_last else None,  
    )
    
    print(f"\n✓ Training Configuration:")
    print(f"  - Checkpoints: {training_config.num_checkpoints}")
    print(f"  - Epochs per checkpoint: {training_config.epochs_per_checkpoint}")
    print(f"  - Total rollouts: {training_config.num_checkpoints * training_config.epochs_per_checkpoint}")
    print(f"  - Total timesteps: {training_config.num_checkpoints * training_config.epochs_per_checkpoint * algo_config['rollout_length']}")
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
    jax.block_until_ready(trainer.run(key))
    end_time = time.time()
    
    print(f"\n{'='*70}")
    print(" Training Complete")
    print("="*70)
    print(f"Training time: {end_time - start_time:.2f}s")