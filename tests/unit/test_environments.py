"""Unit tests for environment adapters.

Tests environment wrapper functionality, vectorization, and interface compliance.
"""
import jax
import jax.numpy as jnp
import pytest
from bordax.environments.utils import make_env, EnvAdapter


@pytest.mark.unit
class TestEnvAdapter:
    """Tests for environment adapter interface."""

    def test_make_env_gymnax(self):
        """Test making a Gymnax environment."""
        env_config = {"init_config": {}, "reset_config": {}}
        env = make_env("gymnax/CartPole-v1", env_config, num_envs=1)

        assert isinstance(env, EnvAdapter), "Should return EnvAdapter instance"
        assert env.num_envs == 1, f"Expected 1 environment, got {env.num_envs}"
        assert env.is_jittable, "Gymnax environments should be jittable"

    def test_vectorized_env_creation(self):
        """Test creating vectorized environment."""
        env_config = {"init_config": {}, "reset_config": {}}
        num_envs = 8
        env = make_env("gymnax/CartPole-v1", env_config, num_envs=num_envs)

        assert env.num_envs == num_envs, \
            f"Expected {num_envs} environments, got {env.num_envs}"

    def test_reset_output_shapes(self, cartpole_env, rng_key):
        """Test reset returns correct shapes."""
        obs, state = cartpole_env.reset(rng_key)

        # CartPole observation is 4D
        expected_obs_shape = (cartpole_env.num_envs, 4)
        assert obs.shape == expected_obs_shape, \
            f"Expected obs shape {expected_obs_shape}, got {obs.shape}"
        assert state is not None, "State should not be None"

    def test_step_output_shapes(self, cartpole_env, rng_key):
        """Test step returns correct shapes and types."""
        obs, state = cartpole_env.reset(rng_key)

        # Take random actions
        actions = jax.random.randint(rng_key, (cartpole_env.num_envs,), 0, 2)
        next_obs, next_state, rewards, dones, info = cartpole_env.step(
            rng_key, state, actions
        )

        # Verify shapes
        assert next_obs.shape == obs.shape, \
            f"Next obs shape {next_obs.shape} should match obs shape {obs.shape}"
        assert rewards.shape == (cartpole_env.num_envs,), \
            f"Expected rewards shape ({cartpole_env.num_envs},), got {rewards.shape}"
        assert dones.shape == (cartpole_env.num_envs,), \
            f"Expected dones shape ({cartpole_env.num_envs},), got {dones.shape}"

        # Verify types
        assert jnp.issubdtype(dones.dtype, jnp.bool_) or jnp.issubdtype(dones.dtype, jnp.integer), \
            f"Dones should be boolean or integer, got {dones.dtype}"

    def test_action_space(self, cartpole_env):
        """Test action space is correctly exposed."""
        action_space = cartpole_env.action_space()

        assert hasattr(action_space, 'n'), "CartPole should have discrete action space"
        assert action_space.n == 2, \
            f"CartPole should have 2 actions, got {action_space.n}"

    def test_obs_space(self, cartpole_env):
        """Test observation space is correctly exposed."""
        obs_space = cartpole_env.obs_space()

        assert hasattr(obs_space, 'shape'), "Observation space should have shape"
        assert obs_space.shape == (4,), \
            f"CartPole obs should be 4D, got {obs_space.shape}"

    def test_multiple_steps(self, cartpole_env, rng_key):
        """Test multiple environment steps work correctly."""
        obs, state = cartpole_env.reset(rng_key)

        # Take 10 steps
        for i in range(10):
            key = jax.random.fold_in(rng_key, i)
            actions = jax.random.randint(key, (cartpole_env.num_envs,), 0, 2)
            obs, state, rewards, dones, info = cartpole_env.step(key, state, actions)

            # Verify all outputs are valid
            assert jnp.all(jnp.isfinite(obs)), f"Step {i}: obs contains non-finite values"
            assert jnp.all(jnp.isfinite(rewards)), f"Step {i}: rewards contain non-finite values"

    def test_deterministic_reset(self, cartpole_env):
        """Test that same seed produces same initial state."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(42)

        obs1, state1 = cartpole_env.reset(key1)
        obs2, state2 = cartpole_env.reset(key2)

        assert jnp.allclose(obs1, obs2), \
            "Same seed should produce identical initial observations"


@pytest.mark.unit
class TestEnvFactory:
    """Tests for environment factory function."""

    def test_invalid_env_prefix(self):
        """Test that invalid environment prefix raises error."""
        env_config = {"init_config": {}, "reset_config": {}}
        with pytest.raises(ValueError, match="Unknown environment prefix"):
            make_env("invalid/SomeEnv-v1", env_config, num_envs=1)

    def test_missing_init_config(self):
        """Test that missing init_config raises error."""
        with pytest.raises(KeyError):
            make_env("gymnax/CartPole-v1", {}, num_envs=1)

    def test_num_envs_parameter(self):
        """Test that num_envs parameter is respected."""
        env_config = {"init_config": {}, "reset_config": {}}
        for num_envs in [1, 4, 8, 16]:
            env = make_env("gymnax/CartPole-v1", env_config, num_envs=num_envs)
            assert env.num_envs == num_envs, \
                f"Expected {num_envs} envs, got {env.num_envs}"
