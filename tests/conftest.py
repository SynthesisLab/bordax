"""Shared pytest fixtures for BordAX tests.

This module provides common fixtures used across unit, integration,
and end-to-end tests.
"""
import jax
import jax.numpy as jnp
import pytest
from bordax.agents.utils import make_agent
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env


@pytest.fixture
def rng_key():
    """Provide a fixed PRNG key for reproducible tests."""
    return jax.random.PRNGKey(0)


@pytest.fixture
def cartpole_env():
    """Provide a CartPole-v1 environment with 4 parallel environments."""
    env_config = {"init_config": {}, "reset_config": {}}
    return make_env("gymnax/CartPole-v1", env_config, num_envs=4)


@pytest.fixture
def cartpole_single_env():
    """Provide a single CartPole-v1 environment for evaluation."""
    env_config = {"init_config": {}, "reset_config": {}}
    return make_env("gymnax/CartPole-v1", env_config, num_envs=1)


@pytest.fixture
def gymnasium_cartpole_env():
    """Provide a non-jittable Gymnasium CartPole-v1 environment."""
    env_config = {"init_config": {}, "reset_config": {}}
    return make_env("gymnasium/CartPole-v1", env_config, num_envs=1)


@pytest.fixture
def sample_cartpole_obs():
    """Provide a sample CartPole observation."""
    # CartPole observation: [x, x_dot, theta, theta_dot]
    return jnp.array([0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def sample_cartpole_batch_obs():
    """Provide a batch of CartPole observations."""
    # Batch of 4 observations
    return jnp.zeros((4, 4))


@pytest.fixture
def mlp_agent_config():
    """Provide standard MLP agent configuration."""
    return {
        "policy_layers": [64, 64],
        "value_layers": [64],
    }


@pytest.fixture
def mlp_agent(cartpole_env, mlp_agent_config):
    """Provide an MLP agent for CartPole."""
    return make_agent("mlp/mlp", cartpole_env, mlp_agent_config)


@pytest.fixture
def dqn_agent(cartpole_env):
    """Provide a DQN agent for CartPole."""
    return make_agent("dqn/mlp", cartpole_env, {"q_layers": [64, 64]})


@pytest.fixture
def ppo_algo_config():
    """Provide standard PPO algorithm configuration."""
    return {
        "rollout_length": 128,
        "gamma": 0.99,
        "_lambda": 0.95,
        "lr": 3e-4,
        "num_minibatches": 4,
        "num_sgd_steps": 4,
    }


@pytest.fixture
def ppo_algo(ppo_algo_config):
    """Provide a PPO algorithm instance."""
    return make_algo("ppo", ppo_algo_config)


@pytest.fixture
def dqn_algo_config():
    """Provide standard DQN algorithm configuration."""
    return {
        "lr": 1e-3,
        "gamma": 0.99,
        "buffer_size": 10000,
        "batch_size": 32,
        "warmup_steps": 100,
    }


@pytest.fixture
def dqn_algo(dqn_algo_config):
    """Provide a DQN algorithm instance."""
    return make_algo("dqn", dqn_algo_config)


@pytest.fixture
def trainer_config():
    """Provide minimal trainer configuration for testing."""
    # Import here to avoid circular dependencies
    from bordax.training.trainer import TrainerConfig
    return TrainerConfig(
        num_checkpoints=2,
        epochs_per_checkpoint=1,
        evaluation_episodes=5,
        enable_evaluation=True,
        debug=False,
    )


@pytest.fixture
def training_state(mlp_agent, cartpole_env, rng_key):
    """Provide a training state for tests."""
    import optax
    from bordax.types import TrainingState

    sample_obs = jnp.zeros(cartpole_env.obs_space().shape)
    params = mlp_agent.init(rng_key, sample_obs)
    optimizer = optax.adam(3e-4)
    optimizer_state = optimizer.init(params)

    return TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        step=jnp.array(0)
    )
