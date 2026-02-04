from bordax.training.trainer import Trainer, TrainerConfig
from bordax.training.logging import LoggerConfig
from bordax.training.checkpointing import CheckpointerConfig
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent

import jax
import time
from datetime import datetime
import os

if __name__ == "__main__":
    print("=" * 70)
    print(" DQN - CartPole-v1")
    print("=" * 70)

    # Create output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root, f"runs/dqn_{timestamp}")
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

    # Logging configuration
    logger_config = LoggerConfig(
        log_dir=output_dir,
        use_wandb=False,
    )

    # Checkpointing configuration
    checkpointer_config = CheckpointerConfig(
        save_path=os.path.join(output_dir, "checkpoints"),
        interval=10,
    )

    # Training configuration
    # For DQN: each epoch collects 'rollout_length' steps and performs 1 update
    # num_checkpoints controls how many evaluations we do
    training_config = TrainerConfig(
        num_checkpoints=200,         # Number of evaluation points
        epochs_per_checkpoint=250,   # Updates between evaluations (250 updates = 250 steps collected)
        evaluation_episodes=10,      # Episodes per evaluation
        debug=True,
        logger_config=logger_config,
        checkpointer_config=checkpointer_config,
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
    jax.block_until_ready(trainer.run(key))
    end_time = time.time()

    print(f"\n{'='*70}")
    print(" Training Complete")
    print("="*70)
    print(f"Training time: {end_time - start_time:.2f}s")

    print("\n✅ DQN training completed successfully!")
    print(f"✓ Metrics logged to: {output_dir}/metrics.csv")
    print(f"✓ Evaluation logged to: {output_dir}/evaluation.csv")
    print(f"✓ Checkpoints saved to: {output_dir}/checkpoints/")
