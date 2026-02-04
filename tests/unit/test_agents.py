"""Unit tests for agent implementations.

Tests agent initialization, forward passes, and parameter shapes.
"""
import jax
import jax.numpy as jnp
import pytest
from bordax.agents.utils import make_agent
from bordax.agents.base import Agent


@pytest.mark.unit
class TestMLPAgent:
    """Tests for MLP policy-value agent."""

    def test_initialization(self, cartpole_env, mlp_agent_config, rng_key):
        """Test MLP agent initializes with correct parameter structure."""
        agent = make_agent("mlp/mlp", cartpole_env, mlp_agent_config)
        sample_obs = jnp.ones(cartpole_env.obs_space().shape)

        params = agent.init(rng_key, sample_obs)

        # Verify parameter structure
        assert hasattr(params, 'policy'), "Parameters should have 'policy' attribute"
        assert hasattr(params, 'value'), "Parameters should have 'value' attribute"

    def test_policy_forward_pass(self, mlp_agent, cartpole_env, sample_cartpole_obs, rng_key):
        """Test policy network produces correct output shapes."""
        params = mlp_agent.init(rng_key, sample_cartpole_obs)

        dist, _ = mlp_agent.policy(params, sample_cartpole_obs, rng_key)

        # Verify output shape matches action space
        action_dim = cartpole_env.action_space().n
        assert dist.logits.shape == (action_dim,), \
            f"Expected logits shape ({action_dim},), got {dist.logits.shape}"

    def test_value_forward_pass(self, mlp_agent, sample_cartpole_obs, rng_key):
        """Test value network produces scalar output."""
        params = mlp_agent.init(rng_key, sample_cartpole_obs)

        value = mlp_agent.value(params, sample_cartpole_obs)

        # Verify scalar output
        assert value.shape == (), f"Expected scalar value, got shape {value.shape}"
        assert jnp.isfinite(value), "Value should be finite"

    def test_action_sampling(self, mlp_agent, sample_cartpole_obs, rng_key):
        """Test action sampling produces valid actions."""
        params = mlp_agent.init(rng_key, sample_cartpole_obs)
        key1, key2 = jax.random.split(rng_key)

        action, info = mlp_agent.action(params, sample_cartpole_obs, key1)

        # Verify action is an integer
        assert jnp.issubdtype(action.dtype, jnp.integer), \
            f"Action should be integer, got {action.dtype}"
        # Verify action is in valid range [0, 1] for CartPole
        assert 0 <= action <= 1, f"Action {action} not in valid range [0, 1]"

    def test_batched_forward_pass(self, mlp_agent, sample_cartpole_batch_obs, rng_key):
        """Test batched forward pass produces correct shapes."""
        batch_size = sample_cartpole_batch_obs.shape[0]
        sample_obs = sample_cartpole_batch_obs[0]
        params = mlp_agent.init(rng_key, sample_obs)

        # Vectorize the policy call
        batched_policy = jax.vmap(lambda obs: mlp_agent.policy(params, obs, rng_key))
        dists, _ = batched_policy(sample_cartpole_batch_obs)

        # Verify batch dimension is preserved
        assert dists.logits.shape[0] == batch_size, \
            f"Expected batch size {batch_size}, got {dists.logits.shape[0]}"

    def test_deterministic_initialization(self, cartpole_env, mlp_agent_config):
        """Test that same seed produces same initialization."""
        agent = make_agent("mlp/mlp", cartpole_env, mlp_agent_config)
        sample_obs = jnp.ones(cartpole_env.obs_space().shape)

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(42)

        params1 = agent.init(key1, sample_obs)
        params2 = agent.init(key2, sample_obs)

        # Compare parameter trees
        tree1 = jax.tree_util.tree_leaves(params1)
        tree2 = jax.tree_util.tree_leaves(params2)

        assert len(tree1) == len(tree2), "Parameter trees should have same structure"
        for p1, p2 in zip(tree1, tree2):
            assert jnp.allclose(p1, p2), "Same seed should produce identical parameters"


@pytest.mark.unit
class TestDQNAgent:
    """Tests for DQN agent."""

    def test_initialization(self, cartpole_env, rng_key):
        """Test DQN agent initializes correctly."""
        agent = make_agent("dqn/mlp", cartpole_env, {"q_layers": [64, 64]})
        sample_obs = jnp.ones(cartpole_env.obs_space().shape)

        params = agent.init(rng_key, sample_obs)

        # DQN has q_network and target_network
        assert hasattr(params, 'q_network'), "DQN should have q_network parameters"
        assert hasattr(params, 'target_network'), "DQN should have target_network parameters"

    def test_q_value_forward_pass(self, dqn_agent, cartpole_env, sample_cartpole_obs, rng_key):
        """Test Q-network produces correct output shape."""
        params = dqn_agent.init(rng_key, sample_cartpole_obs)

        # DQN agent doesn't expose q_values directly, use policy
        dist, info = dqn_agent.policy(params, sample_cartpole_obs, rng_key)

        # Verify output shape matches action space
        action_dim = cartpole_env.action_space().n
        assert dist.logits.shape == (action_dim,), \
            f"Expected logits shape ({action_dim},), got {dist.logits.shape}"

    def test_greedy_action_selection(self, dqn_agent, sample_cartpole_obs, rng_key):
        """Test greedy action selection."""
        params = dqn_agent.init(rng_key, sample_cartpole_obs)

        # Get deterministic action
        action, info = dqn_agent.action(params, sample_cartpole_obs, rng_key, is_deterministic=True)

        # Verify action is an integer
        assert jnp.issubdtype(action.dtype, jnp.integer), \
            f"Action should be integer, got {action.dtype}"

    def test_q_values_are_finite(self, dqn_agent, sample_cartpole_obs, rng_key):
        """Test Q-values are finite (not NaN or inf)."""
        params = dqn_agent.init(rng_key, sample_cartpole_obs)

        dist, info = dqn_agent.policy(params, sample_cartpole_obs, rng_key)
        q_values = dist.logits

        assert jnp.all(jnp.isfinite(q_values)), \
            f"Q-values should be finite, got {q_values}"


@pytest.mark.unit
class TestAgentFactory:
    """Tests for agent factory function."""

    def test_invalid_agent_type_raises_error(self, cartpole_env):
        """Test make_agent raises descriptive error for invalid agent type."""
        with pytest.raises(ValueError, match="not supported"):
            make_agent("invalid_type", cartpole_env, {})

    def test_missing_required_config_raises_error(self, cartpole_env):
        """Test make_agent raises error when required config is missing."""
        # MLP agents require policy_layers and value_layers
        with pytest.raises(KeyError):
            make_agent("mlp/mlp", cartpole_env, {})  # Empty config

    def test_all_registered_agents_are_creatable(self, cartpole_env):
        """Test that all agents in registry can be instantiated."""
        from bordax.agents.utils import AGENT_REGISTRY
        
        # Define minimal configs for each agent type
        configs = {
            "mlp/mlp": {"policy_layers": [32], "value_layers": [32]},
            "dqn/mlp": {"q_layers": [32]},
            # Add other agent types as they exist
        }
        
        for agent_type in AGENT_REGISTRY.keys():
            if agent_type in configs:  # Skip if we don't have a test config
                agent = make_agent(agent_type, cartpole_env, configs[agent_type])
                assert isinstance(agent, Agent), f"{agent_type} should return Agent"
