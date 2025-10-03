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
    """Abstract agent interface."""

    @abstractmethod
    def init(self, key: PRNGKey, sample_obs: Any) -> AgentParameters: ...

    @abstractmethod
    def policy(
        self, params: AgentParameters, obs: Any, key: PRNGKey
    ) -> Tuple[DistributionLike, Mapping[str, Any]]: ...

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


class MixtureAgent(Agent):
    """An agent that mixes the policies of two agents with a given probability."""
    def __init__(self, agents: Tuple[Agent, Agent], prob: float = 1.0):
        self.agent1, self.agent2 = agents
        self.prob = prob

    def init(self, key: PRNGKey, sample_obs: Any) -> AgentParameters:
        key1, key2 = jax.random.split(key)
        params1 = self.agent1.init(key1, sample_obs)
        params2 = self.agent2.init(key2, sample_obs)
        return (params1, params2)

    @functools.partial(jax.jit, static_argnames=("self",))
    def policy(self, params: AgentParameters, obs: Any, key: PRNGKey) -> Tuple[DistributionLike, Mapping[str, Any]]:
        params1, params2 = params
        key1, key2, choice_key = jax.random.split(key, 3)
        
        dist1, info1 = self.agent1.policy(params1, obs, key1)
        dist2, info2 = self.agent2.policy(params2, obs, key2)

        use_agent1 = jax.random.uniform(choice_key) < self.prob
        
        # This assumes the distributions are of the same type (e.g., both Categorical)
        # and can be reconstructed from their parameters.
        # This will fail if mixing discrete and continuous agents.
        selected_logits = jax.lax.cond(use_agent1, lambda: dist1.logits, lambda: dist2.logits)
        selected_info = jax.lax.cond(use_agent1, lambda: info1, lambda: info2)

        return type(dist1)(logits=selected_logits), selected_info

    @functools.partial(jax.jit, static_argnames=("self", "is_deterministic"))
    def action(self, params: AgentParameters, obs: Any, key: PRNGKey, is_deterministic: bool = False) -> Tuple[Any, Mapping[str, Any]]:
        if is_deterministic:
            params1, _ = params
            return self.agent1.action(params1, obs, key, is_deterministic=True)
        
        # Replicate the logic from the base class's action method,
        # as we cannot call super().action() inside a JIT-compiled function.
        policy_key, sample_key = jax.random.split(key)
        dist, info = self.policy(params, obs, policy_key)
        
        action, logp = dist.sample_and_log_prob(seed=sample_key)
        if isinstance(logp, jnp.ndarray) and logp.ndim > 1:
            logp = jnp.sum(logp, axis=-1)
            
        return action, dict(logp=logp, **info)

    def value(self, params: AgentParameters, obs: Any) -> jnp.ndarray:
        params1, _ = params
        return self.agent1.value(params1, obs)
