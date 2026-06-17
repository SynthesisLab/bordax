"""Agent base classes and simple policy/value MLP implementations.

This module defines the Agent abstract base class and several concrete
agents and neural modules used by the project:

- Agent: abstract interface for agents (init, policy, action, value).
- BlankAgent: a simple uniform (random) discrete action agent.
- MLP / MLP_dtsemnet / MLP_boolean: small neural modules used as policy
    architectures.
- MLPPolicyValue / MLPPolicyValueContinuous: actor-critic wrappers that
    expose a policy (Categorical or Normal) and a value function.

Docstrings are provided for classes and public methods to aid reading and
automatic documentation generation.
"""

from bordax.environments.utils import EnvAdapter
from bordax.types import PRNGKey, Params, AgentParameters, PolicyValueParameters
from bordax.agents.components import MLP, MLP_dtsemnet, MLP_boolean
from distrax import DistributionLike, Categorical, Normal

import jax
import jax.numpy as jnp
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Mapping, NamedTuple, Tuple, List
import flax.linen as nn
import functools

class Agent(ABC):
    """Abstract base class for all agents.

    Subclasses must implement ``init`` and ``policy``. The ``action``
    method is provided as a JIT-compiled convenience wrapper around
    ``policy``. Override ``value`` if the agent supports a value function
    (required for actor-critic algorithms such as PPO).
    """

    @abstractmethod
    def init(self, key: PRNGKey, sample_obs: Any) -> AgentParameters:
        """Initialise network parameters.

        Args:
            key: JAX random key for weight initialisation.
            sample_obs: A sample observation with the correct shape
                (including the ``num_envs`` batch dimension).

        Returns:
            An ``AgentParameters`` pytree (e.g. ``PolicyValueParameters``
            or ``DQNParameters``).
        """
        ...

    @abstractmethod
    def policy(
        self, params: AgentParameters, obs: Any, key: PRNGKey
    ) -> Tuple[DistributionLike, Mapping[str, Any]]:
        """Compute the policy distribution for a batch of observations.

        Args:
            params: Current network parameters.
            obs: Batch of observations, shape ``(num_envs, *obs_shape)``.
            key: JAX random key (for stochastic policy heads).

        Returns:
            Tuple of ``(distribution, info)`` where ``distribution`` is a
            Distrax distribution and ``info`` is a dict of auxiliary data.
        """
        ...

    @functools.partial(jax.jit, static_argnames=("self", "is_deterministic"))
    def action(
        self, params: AgentParameters, obs: Any, key: PRNGKey, is_deterministic=False
    ) -> Tuple[DistributionLike, Mapping[str, Any]]:
        """Sample or select an action from the policy distribution."""
        policy_key, sample_key = jax.random.split(key)
        dist, info = self.policy(params, obs, policy_key)
        if is_deterministic:
            action = dist.mode()
            logp = dist.log_prob(action)
        else:
            action, logp = dist.sample_and_log_prob(seed=sample_key)
            if isinstance(logp, jnp.ndarray) and logp.ndim > 1:
                logp = jnp.sum(logp, axis=-1)
        return action, dict(
            logp=logp,
            **info,
        )

    def value(self, params: Params, obs: Any) -> jnp.ndarray:
        """Compute the value estimate for a batch of observations.

        Args:
            params: Current network parameters.
            obs: Batch of observations, shape ``(num_envs, *obs_shape)``.

        Returns:
            Value estimates, shape ``(num_envs,)``.

        Raises:
            NotImplementedError: If the agent has no value function.
        """
        raise NotImplementedError


class BlankAgent(Agent):
    """A trivial agent that returns a uniform categorical policy."""

    def __init__(self, env: EnvAdapter):
        self.action_space = env.action_space()
        if not hasattr(self.action_space, "n"):
            raise ValueError("BlankAgent only supports discrete action spaces.")
        self.batch_dim = None

    def init(self, key: PRNGKey, sample_obs: Any) -> Params:
        self.batch_dim = sample_obs.shape[0]
        return {}

    def policy(self, params: Params, obs: Any, key: PRNGKey) -> Tuple[Any, Mapping[str, Any]]:
        """Return a uniform categorical distribution over actions."""
        pi = Categorical(logits=jnp.ones((self.batch_dim,) + (self.action_space.n,)))
        return pi, {}

    def value(self, params: Params, obs: Any) -> jnp.ndarray:
        return jnp.zeros(obs.shape[:-1])


class MLPPolicyValue(Agent):
    """Actor-critic wrapper for discrete actions."""

    def __init__(self, config: dict, env: EnvAdapter, policy_architecture: str):
        self.config = config
        action_space = env.action_space()
        if policy_architecture == "mlp":
            self.policy_module = MLP(
                layer_sizes=self.config["policy_layers"] + [action_space.n]
            )
        elif policy_architecture == "dt":
            self.policy_module = MLP_dtsemnet(
                tree_depth=self.config["tree_depth"], action_dim=action_space.n
            )
        elif policy_architecture == "bool":
            self.policy_module = MLP_boolean(
                n=self.config["n"], action_dim=action_space.n
            )
        else:
            raise ValueError(f"Unknown policy architecture: {policy_architecture}")

        self.value_module = MLP(layer_sizes=self.config["value_layers"] + [1])

    def init(self, key: PRNGKey, sample_obs: Any) -> PolicyValueParameters:
        policy_key, value_key = jax.random.split(key, 2)
        policy_params = self.policy_module.init(policy_key, sample_obs)
        value_params = self.value_module.init(value_key, sample_obs)
        return PolicyValueParameters(policy=policy_params, value=value_params)

    @functools.partial(jax.jit, static_argnames=("self"))
    def policy(self, params: PolicyValueParameters, obs: Any, key: PRNGKey) -> Tuple[Any, Mapping[str, Any]]:
        logits = self.policy_module.apply(params.policy, obs)
        if isinstance(logits, tuple):
            logits = logits[0]
        pi = Categorical(logits=logits)
        return pi, {}

    @functools.partial(jax.jit, static_argnames=("self"))
    def value(self, params: PolicyValueParameters, obs: Any) -> jnp.ndarray:
        value_out = self.value_module.apply(params.value, obs)
        if isinstance(value_out, tuple):
            value_out = value_out[0]
        return jnp.squeeze(value_out, axis=-1)


class MLPPolicyValueContinuous(Agent):
    """Actor-critic wrapper for continuous actions."""
    def __init__(self, config: dict, env: EnvAdapter, policy_architecture: str):
        self.config = config
        self.n_actions = env.action_space().shape[0]
        if policy_architecture == "mlp":
            self.policy_module = MLP(
                layer_sizes=self.config["policy_layers"] + [2 * self.n_actions]
            )
        else:
            raise ValueError(f"Unknown policy architecture: {policy_architecture}")
        self.value_module = MLP(layer_sizes=self.config["value_layers"] + [1])

    def init(self, key: PRNGKey, sample_obs: Any) -> Params:
        policy_key, value_key = jax.random.split(key, 2)
        policy_params = self.policy_module.init(policy_key, sample_obs)
        value_params = self.value_module.init(value_key, sample_obs)
        return {"policy": policy_params, "value": value_params}

    @functools.partial(jax.jit, static_argnames=("self"))
    def policy(self, params: PolicyValueParameters, obs: Any, key: PRNGKey) -> Tuple[Any, Mapping[str, Any]]:
        distribution_parameters = self.policy_module.apply(params.policy, obs)
        if isinstance(distribution_parameters, tuple):
            distribution_parameters = distribution_parameters[0]
        pi = Normal(
            loc=distribution_parameters[..., : self.n_actions],
            scale=jax.nn.softplus(distribution_parameters[..., self.n_actions :]),
        )
        return pi, {}

    @functools.partial(jax.jit, static_argnames=("self"))
    def value(self, params: PolicyValueParameters, obs: Any) -> jnp.ndarray:
        value_out = self.value_module.apply(params.value, obs)
        if isinstance(value_out, tuple):
            value_out = value_out[0]
        return jnp.squeeze(value_out, axis=-1)

class DQNParameters(NamedTuple):
    q_network: Params
    target_network: Params

class DQNAgent(Agent):
    """A DQN agent with a Q-network and target network."""

    def __init__(self, config: dict, env: EnvAdapter):
        self.config = config
        action_space = env.action_space()
        if not hasattr(action_space, "n"):
            raise ValueError("DQNAgent only supports discrete action spaces.")
        self.n_actions = action_space.n
        self.q_network = MLP(layer_sizes=self.config["q_layers"] + [self.n_actions])
        self.target_network = MLP(layer_sizes=self.config["q_layers"] + [self.n_actions])

    def init(self, key: PRNGKey, sample_obs: Any) -> DQNParameters:
        q_params = self.q_network.init(key, sample_obs)
        return DQNParameters(q_network=q_params, target_network=q_params)
    
    @functools.partial(jax.jit, static_argnames=("self"))
    def policy(self, params: DQNParameters, obs: Any, key: PRNGKey) -> Tuple[Any, Mapping[str, Any]]:
        q_values = self.q_network.apply(params.q_network, obs)
        if isinstance(q_values, tuple):
            q_values = q_values[0]
        pi = Categorical(logits=q_values)
        return pi, {}    
    
    @functools.partial(jax.jit, static_argnames=("self"))
    def value(self, params: DQNParameters, obs: Any) -> jnp.ndarray:
        q_values = self.q_network.apply(params.q_network, obs)
        if isinstance(q_values, tuple):
            q_values = q_values[0]
        max_q_values = jnp.max(q_values, axis=-1)
        return max_q_values

