"""Slow integration tests that verify algorithms actually learn.

These tests run full training loops with production hyperparameters
to ensure PPO and DQN can solve CartPole-v1.

Expected runtime: ~15-20 seconds total
"""
import jax
import jax.numpy as jnp
import pytest
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent
from bordax.data.buffer import ReplayBuffer


@pytest.mark.slow
def test_ppo_learns_cartpole():
    """Verify PPO actually learns to solve CartPole over 20 checkpoints.
    """
    # Setup environment
    env_config = {"init_config": {}, "reset_config": {}}
    env = make_env("gymnax/CartPole-v1", env_config, num_envs=4)
    eval_env = make_env("gymnax/CartPole-v1", env_config, num_envs=1)

    # Create agent with production architecture
    agent_config = {
        "policy_layers": [128, 128, 64],
        "value_layers": [128, 128, 64]
    }
    agent = make_agent("mlp/mlp", env, agent_config)

    # Create algorithm with production hyperparameters
    algo_config = {
        "lr": 1e-5,
        "rollout_length": 2048,
        "num_minibatches": 16,
        "num_sgd_steps": 10,
        "num_envs": 4,
        "gamma": 0.99,
        "_lambda": 0.95,
    }
    algo = make_algo("ppo", algo_config)

    # Initialize
    rng_key = jax.random.PRNGKey(0)
    key_init, key_reset, key_train = jax.random.split(rng_key, 3)

    sample_obs = jnp.zeros(env.obs_space().shape)
    params = agent.init(key_init, sample_obs)
    training_state = algo.updater.init(params)

    obs, env_state = env.reset(key_reset)

    # Training loop (20 checkpoints)
    rewards = []
    losses = []

    for checkpoint in range(20):
        # Split keys
        key_train, key_collect, key_batch, key_update, key_eval = jax.random.split(key_train, 5)

        # Collect rollout
        (obs, env_state), trajectory = algo.collector(
            key_collect, env, obs, env_state, None, agent, training_state
        )

        # Build batch
        batch = algo.batch_builder(key_batch, trajectory)

        # Update
        training_state, metrics = algo.updater(agent, batch, training_state, key_update)

        # Track metrics
        if "total_loss" in metrics:
            losses.append(float(metrics["total_loss"]))

        # Evaluate every checkpoint
        eval_obs, eval_state = eval_env.reset(key_eval)
        eval_rewards = []

        for _ in range(5):  # 5 evaluation episodes
            done = False
            episode_reward = 0.0
            step = 0
            max_steps = 500

            while not done and step < max_steps:
                key_eval, key_action = jax.random.split(key_eval)
                action, _ = agent.action(training_state.params, eval_obs, key_action, is_deterministic=True)
                eval_obs, eval_state, reward, done, _ = eval_env.step(
                    key_eval, eval_state, action
                )
                episode_reward += float(reward[0])
                step += 1

            eval_rewards.append(episode_reward)

        avg_reward = jnp.mean(jnp.array(eval_rewards))
        rewards.append(float(avg_reward))

    # Assertions
    final_reward = rewards[-1]
    initial_reward = rewards[0]

    assert final_reward > 400, \
        f"PPO failed to learn: final reward {final_reward:.1f} < 400"

    assert final_reward > initial_reward + 50, \
        f"PPO learning insufficient: improvement {final_reward - initial_reward:.1f} < 50"

    assert all(jnp.isfinite(r) for r in rewards), \
        "NaN/Inf detected in evaluation rewards"

    assert all(jnp.isfinite(loss) for loss in losses), \
        "NaN/Inf detected in training losses"

    print(f"\n✓ PPO Learning Test Passed:")
    print(f"  Initial reward: {initial_reward:.1f}")
    print(f"  Final reward: {final_reward:.1f}")
    print(f"  Improvement: {final_reward - initial_reward:.1f}")


@pytest.mark.slow
def test_dqn_learns_cartpole():
    """Verify DQN actually learns to solve CartPole over 50 checkpoints.
    """
    # Setup environment (single env for off-policy)
    env_config = {"init_config": {}, "reset_config": {}}
    env = make_env("gymnax/CartPole-v1", env_config, num_envs=1)
    eval_env = make_env("gymnax/CartPole-v1", env_config, num_envs=1)

    # Create agent with production architecture
    agent_config = {"q_layers": [120, 84]}
    agent = make_agent("dqn/mlp", env, agent_config)

    # Create algorithm with production hyperparameters
    epsilon_schedule = lambda t: max(0.01, 0.9 * (0.995 ** t))
    algo_config = {
        "lr": 2.5e-4,
        "rollout_length": 1,
        "batch_size": 128,
        "target_update_freq": 500,
        "gamma": 0.99,
        "epsilon_schedule": epsilon_schedule,
    }
    algo = make_algo("dqn", algo_config)

    # Initialize
    rng_key = jax.random.PRNGKey(42)
    key_init, key_reset, key_warmup = jax.random.split(rng_key, 3)

    sample_obs = jnp.zeros(env.obs_space().shape)
    params = agent.init(key_init, sample_obs)
    training_state = algo.updater.init(params)

    obs, env_state = env.reset(key_reset)

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=10000,
        obs_shape=env.obs_space().shape,
        action_shape=()
    )

    # Warmup: Fill replay buffer with 1000 transitions
    for step in range(1000):
        key_warmup, key_collect = jax.random.split(key_warmup)
        (obs, env_state), replay_buffer = algo.collector(
            key_collect, env, obs, env_state, replay_buffer, agent, training_state
        )

    assert len(replay_buffer) >= 1000, \
        f"Warmup failed: buffer size {len(replay_buffer)} < 1000"

    # Training loop (50 checkpoints × 250 steps each)
    rewards = []
    q_values_list = []
    losses = []
    key_train = key_warmup

    for checkpoint in range(50):
        # Collect and train for 250 steps
        checkpoint_losses = []
        checkpoint_q_values = []

        for step in range(250):
            key_train, key_collect, key_sample, key_update = jax.random.split(key_train, 4)

            # Collect transition
            (obs, env_state), replay_buffer = algo.collector(
                key_collect, env, obs, env_state, replay_buffer, agent, training_state
            )

            # Sample batch and update
            batch = algo.batch_builder(key_sample, replay_buffer)
            training_state, metrics = algo.updater(agent, batch, training_state, key_update)

            # Track metrics
            if "dqn_loss" in metrics:
                checkpoint_losses.append(float(metrics["dqn_loss"]))
            if "mean_q_value" in metrics:
                checkpoint_q_values.append(float(metrics["mean_q_value"]))

        # Store average metrics for checkpoint
        if checkpoint_losses:
            losses.append(jnp.mean(jnp.array(checkpoint_losses)))
        if checkpoint_q_values:
            q_values_list.append(jnp.mean(jnp.array(checkpoint_q_values)))

        # Evaluate every checkpoint
        key_train, key_eval = jax.random.split(key_train)
        eval_obs, eval_state = eval_env.reset(key_eval)
        eval_rewards = []

        for _ in range(5):  # 5 evaluation episodes
            done = False
            episode_reward = 0.0
            step = 0
            max_steps = 500

            while not done and step < max_steps:
                key_eval, key_action = jax.random.split(key_eval)
                # Use greedy policy for evaluation (epsilon=0)
                action, _ = agent.action(training_state.params, eval_obs, key_action, is_deterministic=True)
                eval_obs, eval_state, reward, done, _ = eval_env.step(
                    key_eval, eval_state, action
                )
                episode_reward += float(reward[0])
                step += 1

            eval_rewards.append(episode_reward)

        avg_reward = jnp.mean(jnp.array(eval_rewards))
        rewards.append(float(avg_reward))

    # Assertions
    final_reward = rewards[-1]
    initial_reward = rewards[0]

    assert final_reward > 150, \
        f"DQN failed to learn: final reward {final_reward:.1f} < 150"

    assert final_reward > initial_reward + 50, \
        f"DQN learning insufficient: improvement {final_reward - initial_reward:.1f} < 50"

    assert all(jnp.isfinite(r) for r in rewards), \
        "NaN/Inf detected in evaluation rewards"

    assert all(jnp.isfinite(q) for q in q_values_list), \
        "NaN/Inf detected in Q-values"

    assert all(jnp.isfinite(loss) for loss in losses), \
        "NaN/Inf detected in training losses"

    print(f"\n✓ DQN Learning Test Passed:")
    print(f"  Initial reward: {initial_reward:.1f}")
    print(f"  Final reward: {final_reward:.1f}")
    print(f"  Improvement: {final_reward - initial_reward:.1f}")
    print(f"  Final Q-value: {q_values_list[-1]:.2f}")
