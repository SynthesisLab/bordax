"""
Example training script for DQN on CartPole-v1.

This script demonstrates how to use the DQN algorithm in the BordAX framework.
It follows the same structure as train.py but uses DQN instead of PPO.
"""

from bordax.trainer import Trainer, TrainerConfig
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent
from bordax.buffer import ReplayBuffer

import jax
import time
import numpy as np

if __name__ == "__main__":
    print("="*70)
    print(" DQN Training Example - CartPole-v1")
    print("="*70)
    
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
        "q_layers": [128, 128, 64],  # Q-network architecture
    }
    agent = make_agent(agent_name, env, agent_config)
    
    print(f"\n✓ Agent: {agent_name}")
    print(f"  - Q-network layers: {agent_config['q_layers']}")

    # Algorithm configuration
    algo_name = "dqn"
    algo_config = {
        "epsilon": 0.1,              # Epsilon for epsilon-greedy
        "rollout_length": 1,         # Collect 1 step at a time
        "batch_size": 64,            # Batch size for updates
        "gamma": 0.99,               # Discount factor
        "lr": 1e-3,                  # Learning rate
        "target_update_freq": 500,   # Update target every 500 steps
    }
    algorithm = make_algo(algo_name, algo_config)
    
    print(f"\n✓ Algorithm: {algo_name}")
    print(f"  - Epsilon: {algo_config['epsilon']}")
    print(f"  - Batch size: {algo_config['batch_size']}")
    print(f"  - Learning rate: {algo_config['lr']}")
    print(f"  - Gamma: {algo_config['gamma']}")
    print(f"  - Target update frequency: {algo_config['target_update_freq']}")

    # Create replay buffer
    obs_shape = env.obs_space().shape
    action_shape = env.action_space().shape
    buffer_capacity = 50000
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity, 
        obs_shape=obs_shape, 
        action_shape=action_shape
    )
    
    print(f"\n✓ Replay Buffer:")
    print(f"  - Capacity: {buffer_capacity}")
    print(f"  - Obs shape: {obs_shape}")
    print(f"  - Action shape: {action_shape}")

    # Training configuration
    training_config = TrainerConfig(
        num_checkpoints=200,
        epochs_per_checkpoint=1,
        evaluation_episodes=32,
        debug=True,
        save_model=False,
    )
    
    print(f"\n✓ Training Configuration:")
    print(f"  - Checkpoints: {training_config.num_checkpoints}")
    print(f"  - Epochs per checkpoint: {training_config.epochs_per_checkpoint}")
    print(f"  - Evaluation episodes: {training_config.evaluation_episodes}")

    # Note: The Trainer class needs to be adapted to work with replay buffers
    # For now, we'll do a simple training loop manually
    
    print(f"\n{'='*70}")
    print(" Starting Training")
    print("="*70)
    
    # Initialize
    key = jax.random.PRNGKey(0)
    init_key, key = jax.random.split(key)
    obs, state = env.reset(init_key)
    ts = algorithm.init_training_state(agent, init_key, obs, env)
    
    # Fill replay buffer with random experience
    print("\nPhase 1: Filling replay buffer...")
    warmup_steps = 1000
    for i in range(warmup_steps):
        key, collect_key = jax.random.split(key)
        (obs, state), replay_buffer = algorithm.collect(
            collect_key, env, obs, state, replay_buffer, agent, ts
        )
        if (i + 1) % 200 == 0:
            print(f"  Warmup: {i+1}/{warmup_steps}, Buffer size: {len(replay_buffer)}")
    
    print(f"✓ Buffer filled with {len(replay_buffer)} transitions\n")
    
    # Training loop
    print("Phase 2: Training...")
    num_training_steps = 5000
    update_freq = 4  # Update every N environment steps
    log_freq = 200
    
    start_time = time.time()
    losses = []
    q_values = []
    
    for step in range(num_training_steps):
        # Collect transition
        key, collect_key = jax.random.split(key)
        (obs, state), replay_buffer = algorithm.collect(
            collect_key, env, obs, state, replay_buffer, agent, ts
        )
        
        # Update policy
        if step % update_freq == 0 and len(replay_buffer) >= algo_config['batch_size']:
            key, batch_key, update_key = jax.random.split(key, 3)
            batch = algorithm.batch_builder(batch_key, replay_buffer)
            ts, metrics = algorithm.update(agent, batch, ts, update_key)
            
            losses.append(float(metrics['dqn_loss']))
            q_values.append(float(metrics['mean_q_value']))
        
        # Logging
        if (step + 1) % log_freq == 0:
            elapsed = time.time() - start_time
            if losses:
                avg_loss = np.mean(losses[-50:])
                avg_q = np.mean(q_values[-50:])
                print(f"  Step {step+1:5d}/{num_training_steps} | "
                      f"Loss: {avg_loss:.4f} | Q-value: {avg_q:.4f} | "
                      f"Buffer: {len(replay_buffer)} | Time: {elapsed:.1f}s")
    
    end_time = time.time()
    
    print(f"\n{'='*70}")
    print(" Training Complete")
    print("="*70)
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    print(f"Training steps: {num_training_steps}")
    print(f"Final buffer size: {len(replay_buffer)}")
    print(f"Total updates: {len(losses)}")
    if losses:
        print(f"Initial loss: {losses[0]:.4f}")
        print(f"Final loss (avg last 50): {np.mean(losses[-50:]):.4f}")
        print(f"Final Q-value (avg last 50): {np.mean(q_values[-50:]):.4f}")
    
    print("\n✅ DQN training completed successfully!")
    
    # Optional: Evaluate the trained agent
    print(f"\n{'='*70}")
    print(" Evaluating Trained Agent")
    print("="*70)
    
    eval_episodes = 10
    episode_rewards = []
    
    for ep in range(eval_episodes):
        key, eval_key = jax.random.split(key)
        obs, state = eval_env.reset(eval_key)
        episode_reward = 0
        done = False
        steps = 0
        max_steps = 500
        
        while not done and steps < max_steps:
            key, action_key = jax.random.split(key)
            # Use deterministic policy (greedy)
            action, _ = agent.action(ts.params, obs, action_key, is_deterministic=True)
            obs, state, reward, done, _ = eval_env.step(key, state, action)
            episode_reward += float(reward[0])
            steps += 1
            done = bool(done[0])
        
        episode_rewards.append(episode_reward)
        print(f"  Episode {ep+1:2d}: Reward = {episode_reward:.1f}, Steps = {steps}")
    
    print(f"\nAverage reward over {eval_episodes} episodes: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print("="*70)
