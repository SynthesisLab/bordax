"""Integration tests for full training pipelines.

Tests that full training iterations work end-to-end.
"""
import jax
import jax.numpy as jnp
import pytest
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent
from bordax.types import TrainingState


@pytest.mark.integration
def test_ppo_training_iteration():
    """Full PPO: init → collect → batch → update → verify state changed."""
    # Create environment
    env_config = {"init_config": {}, "reset_config": {}}
    env = make_env("gymnax/CartPole-v1", env_config, num_envs=4)

    # Create agent
    agent_config = {"policy_layers": [32, 32], "value_layers": [32, 32]}
    agent = make_agent("mlp/mlp", env, agent_config)

    # Create algorithm
    algo_config = {
        "lr": 1e-3,
        "rollout_length": 16,
        "num_minibatches": 4,
        "num_sgd_steps": 1,
        "num_envs": 4,  # Must match env creation
    }
    algo = make_algo("ppo", algo_config)

    # Initialize
    rng_key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(rng_key)

    sample_obs = jnp.zeros(env.obs_space().shape)
    params = agent.init(key1, sample_obs)
    training_state = algo.updater.init(params)

    obs, env_state = env.reset(key2)

    # Save initial state
    initial_params_flat = jax.tree_util.tree_leaves(training_state.params)
    initial_step = training_state.step

    # Collect data
    key1, key2 = jax.random.split(key1)
    (final_obs, final_env_state), trajectory = algo.collector(
        key1, env, obs, env_state, None, agent, training_state
    )

    # Build batch
    batch = algo.batch_builder(key2, trajectory)

    # Update
    key1, key2 = jax.random.split(key1)
    new_training_state, metrics = algo.updater(agent, batch, training_state, key2)

    # Verify training state changed
    new_params_flat = jax.tree_util.tree_leaves(new_training_state.params)

    params_changed = False
    for old_p, new_p in zip(initial_params_flat, new_params_flat):
        if not jnp.allclose(old_p, new_p, atol=1e-6):
            params_changed = True
            break

    assert params_changed, "Parameters should change after PPO update"
    assert new_training_state.step == initial_step + 1, "Step should increment"
    assert "total_loss" in metrics, "Should have loss metrics"


@pytest.mark.integration
def test_dqn_training_iteration():
    """Full DQN: init → collect → sample → update → verify state changed."""
    from bordax.data.buffer import ReplayBuffer

    # Create environment (single env for off-policy)
    env_config = {"init_config": {}, "reset_config": {}}
    env = make_env("gymnax/CartPole-v1", env_config, num_envs=1)

    # Create agent
    agent_config = {"q_layers": [32, 32]}
    agent = make_agent("dqn/mlp", env, agent_config)

    # Create algorithm
    algo_config = {
        "lr": 1e-3,
        "rollout_length": 8,
        "batch_size": 4,
        "target_update_freq": 10,
    }
    algo = make_algo("dqn", algo_config)

    # Initialize
    rng_key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(rng_key)

    sample_obs = jnp.zeros(env.obs_space().shape)
    params = agent.init(key1, sample_obs)
    training_state = algo.updater.init(params)

    obs, env_state = env.reset(key2)

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=1000,
        obs_shape=env.obs_space().shape,
        action_shape=()
    )

    # Save initial state
    initial_q_params_flat = jax.tree_util.tree_leaves(training_state.params.q_network)
    initial_step = training_state.step

    # Collect data
    key1, key2 = jax.random.split(key1)
    (final_obs, final_env_state), replay_buffer = algo.collector(
        key1, env, obs, env_state, replay_buffer, agent, training_state
    )

    # Verify buffer has data
    assert len(replay_buffer) > 0, "Replay buffer should have transitions"

    # Sample batch
    key1, key2 = jax.random.split(key1)
    batch = algo.batch_builder(key2, replay_buffer)

    # Update
    key1, key2 = jax.random.split(key1)
    new_training_state, metrics = algo.updater(agent, batch, training_state, key2)

    # Verify training state changed
    new_q_params_flat = jax.tree_util.tree_leaves(new_training_state.params.q_network)

    params_changed = False
    for old_p, new_p in zip(initial_q_params_flat, new_q_params_flat):
        if not jnp.allclose(old_p, new_p, atol=1e-6):
            params_changed = True
            break

    assert params_changed, "Q-network parameters should change after DQN update"
    assert new_training_state.step == initial_step + 1, "Step should increment"
    assert "dqn_loss" in metrics, "Should have DQN loss metrics"
