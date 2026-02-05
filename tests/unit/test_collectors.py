"""Unit tests for data collectors.

Tests collector output structure and key behaviors.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bordax.data.collectors import OnPolicyCollector, EpsGreedyCollector


@pytest.mark.unit
class TestOnPolicyCollector:
    """Tests for on-policy collector."""

    def test_trajectory_structure(self, cartpole_env, mlp_agent, rng_key, training_state):
        """Check output has required keys and correct shapes."""
        collector = OnPolicyCollector(rollout_length=16, gamma=0.99, _lambda=0.95)

        obs, env_state = cartpole_env.reset(rng_key)

        (final_obs, final_state), trajectory = collector(
            rng_key, cartpole_env, obs, env_state, None, mlp_agent, training_state
        )

        # Check required keys
        required_keys = ["obs", "action", "reward", "done", "info", "advantages", "targets"]
        for key in required_keys:
            assert key in trajectory, f"Missing key: {key}"

        # Check shapes: (rollout_length, num_envs, ...)
        assert trajectory["obs"].shape[0] == 16, "Wrong rollout length"
        assert trajectory["obs"].shape[1] == cartpole_env.num_envs, "Wrong num envs"
        assert trajectory["advantages"].shape == (16, cartpole_env.num_envs)
        assert trajectory["targets"].shape == (16, cartpole_env.num_envs)

    def test_gae_computation(self, cartpole_env, mlp_agent, rng_key, training_state):
        """Validate GAE produces reasonable values: targets = advantages + values."""
        collector = OnPolicyCollector(rollout_length=16, gamma=0.99, _lambda=0.95)

        obs, env_state = cartpole_env.reset(rng_key)

        (final_obs, final_state), trajectory = collector(
            rng_key, cartpole_env, obs, env_state, None, mlp_agent, training_state
        )

        # Compute values from trajectory observations
        values = mlp_agent.value(training_state.params, trajectory["obs"])

        # GAE relationship: targets = advantages + values
        expected_targets = trajectory["advantages"] + values

        assert jnp.allclose(trajectory["targets"], expected_targets, rtol=1e-5), \
            "GAE computation incorrect: targets should equal advantages + values"


@pytest.mark.unit
class TestEpsGreedyCollector:
    """Tests for epsilon-greedy collector."""

    def test_transition_structure(self, dqn_agent, rng_key):
        """Check output has: obs, action, reward, next_obs, done."""
        import optax
        from bordax.types import TrainingState
        from bordax.data.buffer import ReplayBuffer
        from bordax.environments.utils import make_env

        # Use single environment for off-policy collection
        env_config = {"init_config": {}, "reset_config": {}}
        single_env = make_env("gymnax/CartPole-v1", env_config, num_envs=1)

        collector = EpsGreedyCollector(epsilon_schedule=lambda s: 0.1, rollout_length=4)

        obs, env_state = single_env.reset(rng_key)

        # Create training state for DQN agent
        sample_obs = jnp.zeros(single_env.obs_space().shape)
        params = dqn_agent.init(rng_key, sample_obs)
        optimizer = optax.adam(1e-3)
        training_state = TrainingState(
            optimizer_state=optimizer.init(params),
            params=params,
            step=jnp.array(0)
        )

        # Create replay buffer
        replay_buffer = ReplayBuffer(
            capacity=1000,
            obs_shape=single_env.obs_space().shape,
            action_shape=()  # Discrete actions are scalars
        )

        (final_obs, final_state), replay_buffer = collector(
            rng_key, single_env, obs, env_state, replay_buffer, dqn_agent, training_state
        )

        # Check that buffer received transitions
        assert len(replay_buffer) == 4, f"Expected 4 transitions in buffer, got {len(replay_buffer)}"

        # Sample from buffer to check structure
        batch = replay_buffer.sample(batch_size=2)

        # Check required keys for off-policy learning
        required_keys = ["obs", "action", "reward", "next_obs", "done"]
        for key in required_keys:
            assert key in batch, f"Missing key: {key}"

        # Check shapes
        assert batch["obs"].shape[0] == 2, "Wrong batch size"
        assert batch["next_obs"].shape == batch["obs"].shape

    def test_epsilon_exploration(self, dqn_agent, rng_key):
        """Verify epsilon > 0 produces different actions than epsilon = 0."""
        import optax
        from bordax.types import TrainingState
        from bordax.data.buffer import ReplayBuffer
        from bordax.environments.utils import make_env

        # Use single environment for off-policy collection
        env_config = {"init_config": {}, "reset_config": {}}
        single_env = make_env("gymnax/CartPole-v1", env_config, num_envs=1)

        # Create training state for DQN agent
        sample_obs = jnp.zeros(single_env.obs_space().shape)
        params = dqn_agent.init(rng_key, sample_obs)
        optimizer = optax.adam(1e-3)
        training_state = TrainingState(
            optimizer_state=optimizer.init(params),
            params=params,
            step=jnp.array(0)
        )

        # Create replay buffers
        buffer1 = ReplayBuffer(capacity=1000, obs_shape=single_env.obs_space().shape, action_shape=())
        buffer2 = ReplayBuffer(capacity=1000, obs_shape=single_env.obs_space().shape, action_shape=())

        # Collect with high epsilon (should explore)
        explore_collector = EpsGreedyCollector(epsilon_schedule=lambda s: 0.9, rollout_length=8)

        # Collect with zero epsilon (pure exploitation)
        exploit_collector = EpsGreedyCollector(epsilon_schedule=lambda s: 0.0, rollout_length=8)

        obs, env_state = single_env.reset(rng_key)

        # Reset to same state
        obs2, env_state2 = obs, env_state

        (_, _), buffer1 = explore_collector(
            rng_key, single_env, obs, env_state, buffer1, dqn_agent, training_state
        )
        (_, _), buffer2 = exploit_collector(
            rng_key, single_env, obs2, env_state2, buffer2, dqn_agent, training_state
        )

        # Sample actions from buffers to compare
        explore_batch = buffer1.sample(batch_size=8)
        exploit_batch = buffer2.sample(batch_size=8)

        # With high epsilon, actions should differ from greedy (at least sometimes)
        # This is probabilistic, but with epsilon=0.9 and 8 steps, very likely to differ
        actions_differ = not jnp.array_equal(explore_batch["action"], exploit_batch["action"])
        assert actions_differ, "Epsilon exploration should produce different actions"


@pytest.mark.unit
class TestOnPolicyCollectorNonJittable:
    """Tests for on-policy collector with non-jittable environments."""

    def test_trajectory_structure_gymnasium(self, gymnasium_cartpole_env, rng_key):
        """Check non-jittable collection produces correct trajectory structure."""
        import optax
        from bordax.types import TrainingState
        from bordax.agents.utils import make_agent

        # Create agent for gymnasium environment
        agent = make_agent("mlp/mlp", gymnasium_cartpole_env, {
            "policy_layers": [64, 64],
            "value_layers": [64],
        })

        # Create training state
        sample_obs = jnp.zeros(gymnasium_cartpole_env.obs_space().shape)
        params = agent.init(rng_key, sample_obs)
        optimizer = optax.adam(3e-4)
        training_state = TrainingState(
            optimizer_state=optimizer.init(params),
            params=params,
            step=jnp.array(0)
        )

        collector = OnPolicyCollector(rollout_length=16, gamma=0.99, _lambda=0.95)

        obs, env_state = gymnasium_cartpole_env.reset(rng_key)

        (final_obs, final_state), trajectory = collector(
            rng_key, gymnasium_cartpole_env, obs, env_state, None, agent, training_state
        )

        # Check environment is non-jittable
        assert not gymnasium_cartpole_env.is_jittable, "Expected non-jittable environment"

        # Check required keys
        required_keys = ["obs", "action", "reward", "done", "info", "advantages", "targets"]
        for key in required_keys:
            assert key in trajectory, f"Missing key: {key}"

        # Check shapes: (rollout_length, num_envs, ...)
        assert trajectory["obs"].shape[0] == 16, "Wrong rollout length"
        assert trajectory["obs"].shape[1] == gymnasium_cartpole_env.num_envs, "Wrong num envs"
        assert trajectory["advantages"].shape == (16, gymnasium_cartpole_env.num_envs)
        assert trajectory["targets"].shape == (16, gymnasium_cartpole_env.num_envs)

        # Check final observation shape
        assert final_obs.shape == (gymnasium_cartpole_env.num_envs,) + gymnasium_cartpole_env.obs_space().shape

    def test_values_are_finite_gymnasium(self, gymnasium_cartpole_env, rng_key):
        """Ensure no NaN/Inf in trajectory values from non-jittable collection."""
        import optax
        from bordax.types import TrainingState
        from bordax.agents.utils import make_agent

        agent = make_agent("mlp/mlp", gymnasium_cartpole_env, {
            "policy_layers": [64, 64],
            "value_layers": [64],
        })

        sample_obs = jnp.zeros(gymnasium_cartpole_env.obs_space().shape)
        params = agent.init(rng_key, sample_obs)
        optimizer = optax.adam(3e-4)
        training_state = TrainingState(
            optimizer_state=optimizer.init(params),
            params=params,
            step=jnp.array(0)
        )

        collector = OnPolicyCollector(rollout_length=16, gamma=0.99, _lambda=0.95)
        obs, env_state = gymnasium_cartpole_env.reset(rng_key)

        (_, _), trajectory = collector(
            rng_key, gymnasium_cartpole_env, obs, env_state, None, agent, training_state
        )

        # Check all numeric values are finite
        assert jnp.all(jnp.isfinite(trajectory["obs"])), "Observations contain NaN/Inf"
        assert jnp.all(jnp.isfinite(trajectory["reward"])), "Rewards contain NaN/Inf"
        assert jnp.all(jnp.isfinite(trajectory["advantages"])), "Advantages contain NaN/Inf"
        assert jnp.all(jnp.isfinite(trajectory["targets"])), "Targets contain NaN/Inf"


@pytest.mark.unit
class TestEpsGreedyCollectorNonJittable:
    """Tests for epsilon-greedy collector with non-jittable environments."""

    def test_transition_structure_gymnasium(self, gymnasium_cartpole_env, rng_key):
        """Check non-jittable collection produces correct transition structure."""
        import optax
        from bordax.types import TrainingState
        from bordax.data.buffer import ReplayBuffer
        from bordax.agents.utils import make_agent

        # Create DQN agent for gymnasium environment
        agent = make_agent("dqn/mlp", gymnasium_cartpole_env, {"q_layers": [64, 64]})

        collector = EpsGreedyCollector(epsilon_schedule=lambda s: 0.1, rollout_length=10)

        obs, env_state = gymnasium_cartpole_env.reset(rng_key)

        # Create training state
        sample_obs = jnp.zeros(gymnasium_cartpole_env.obs_space().shape)
        params = agent.init(rng_key, sample_obs)
        optimizer = optax.adam(1e-3)
        training_state = TrainingState(
            optimizer_state=optimizer.init(params),
            params=params,
            step=jnp.array(0)
        )

        # Create replay buffer
        replay_buffer = ReplayBuffer(
            capacity=1000,
            obs_shape=gymnasium_cartpole_env.obs_space().shape,
            action_shape=()
        )

        # Check environment is non-jittable
        assert not gymnasium_cartpole_env.is_jittable, "Expected non-jittable environment"

        (final_obs, final_state), replay_buffer = collector(
            rng_key, gymnasium_cartpole_env, obs, env_state, replay_buffer, agent, training_state
        )

        # Check that buffer received transitions
        assert len(replay_buffer) == 10, f"Expected 10 transitions in buffer, got {len(replay_buffer)}"

        # Sample from buffer to check structure
        batch = replay_buffer.sample(batch_size=5)

        # Check required keys for off-policy learning
        required_keys = ["obs", "action", "reward", "next_obs", "done"]
        for key in required_keys:
            assert key in batch, f"Missing key: {key}"

        # Check shapes
        assert batch["obs"].shape[0] == 5, "Wrong batch size"
        assert batch["next_obs"].shape == batch["obs"].shape

        # Check final observation shape
        assert final_obs.shape == (gymnasium_cartpole_env.num_envs,) + gymnasium_cartpole_env.obs_space().shape

    def test_epsilon_exploration_gymnasium(self, gymnasium_cartpole_env, rng_key):
        """Verify epsilon exploration works correctly with non-jittable environments."""
        import optax
        from bordax.types import TrainingState
        from bordax.data.buffer import ReplayBuffer
        from bordax.agents.utils import make_agent

        agent = make_agent("dqn/mlp", gymnasium_cartpole_env, {"q_layers": [64, 64]})

        sample_obs = jnp.zeros(gymnasium_cartpole_env.obs_space().shape)
        params = agent.init(rng_key, sample_obs)
        optimizer = optax.adam(1e-3)
        training_state = TrainingState(
            optimizer_state=optimizer.init(params),
            params=params,
            step=jnp.array(0)
        )

        # Create replay buffers
        buffer1 = ReplayBuffer(capacity=1000, obs_shape=gymnasium_cartpole_env.obs_space().shape, action_shape=())
        buffer2 = ReplayBuffer(capacity=1000, obs_shape=gymnasium_cartpole_env.obs_space().shape, action_shape=())

        # Collect with high epsilon (should explore)
        explore_collector = EpsGreedyCollector(epsilon_schedule=lambda s: 0.9, rollout_length=20)

        # Collect with zero epsilon (pure exploitation)
        exploit_collector = EpsGreedyCollector(epsilon_schedule=lambda s: 0.0, rollout_length=20)

        obs, env_state = gymnasium_cartpole_env.reset(rng_key)
        obs2, env_state2 = gymnasium_cartpole_env.reset(rng_key)

        key1, key2 = jax.random.split(rng_key)

        (_, _), buffer1 = explore_collector(
            key1, gymnasium_cartpole_env, obs, env_state, buffer1, agent, training_state
        )
        (_, _), buffer2 = exploit_collector(
            key2, gymnasium_cartpole_env, obs2, env_state2, buffer2, agent, training_state
        )

        # Sample actions from buffers
        explore_batch = buffer1.sample(batch_size=20)
        exploit_batch = buffer2.sample(batch_size=20)

        # With high epsilon (0.9), we expect mostly random actions
        # Count how many actions are different
        actions_differ_count = np.sum(np.asarray(explore_batch["action"]) != np.asarray(exploit_batch["action"]))

        # With epsilon=0.9 and 20 samples, very likely to have differences
        assert actions_differ_count > 0, "Epsilon exploration should produce at least some different actions"

    def test_values_are_finite_gymnasium(self, gymnasium_cartpole_env, rng_key):
        """Ensure no NaN/Inf in transitions from non-jittable collection."""
        import optax
        from bordax.types import TrainingState
        from bordax.data.buffer import ReplayBuffer
        from bordax.agents.utils import make_agent

        agent = make_agent("dqn/mlp", gymnasium_cartpole_env, {"q_layers": [64, 64]})

        sample_obs = jnp.zeros(gymnasium_cartpole_env.obs_space().shape)
        params = agent.init(rng_key, sample_obs)
        optimizer = optax.adam(1e-3)
        training_state = TrainingState(
            optimizer_state=optimizer.init(params),
            params=params,
            step=jnp.array(0)
        )

        collector = EpsGreedyCollector(epsilon_schedule=lambda s: 0.5, rollout_length=10)

        replay_buffer = ReplayBuffer(
            capacity=1000,
            obs_shape=gymnasium_cartpole_env.obs_space().shape,
            action_shape=()
        )

        obs, env_state = gymnasium_cartpole_env.reset(rng_key)

        (_, _), replay_buffer = collector(
            rng_key, gymnasium_cartpole_env, obs, env_state, replay_buffer, agent, training_state
        )

        batch = replay_buffer.sample(batch_size=10)

        # Check all numeric values are finite
        assert jnp.all(jnp.isfinite(batch["obs"])), "Observations contain NaN/Inf"
        assert jnp.all(jnp.isfinite(batch["next_obs"])), "Next observations contain NaN/Inf"
        assert jnp.all(jnp.isfinite(batch["reward"])), "Rewards contain NaN/Inf"
