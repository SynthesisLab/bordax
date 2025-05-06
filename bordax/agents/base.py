from bordax.types import PRNGKey, Params
from distrax import DistributionLike, Categorical

import jax
import jax.numpy as jnp
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Mapping, Tuple, List, Sequence
import flax.linen as nn
import functools

class Agent(ABC):
    @abstractmethod
    def init(self, key: PRNGKey, sample_obs: Any, action_space: Any) -> Params: ...

    @abstractmethod  # that gives the distribution
    def policy(
        self, params: Params, obs: Any
    ) -> Tuple[DistributionLike, Mapping[str, Any]]: ...


    @functools.partial(jax.jit, static_argnames=("self", "is_deterministic"))
    def action(
        self, params: Params, obs: Any, key: PRNGKey, is_deterministic=False
    ) -> Tuple[DistributionLike, Mapping[str, Any]]:
        dist, info = self.policy(params, obs)
        if is_deterministic:
            action = dist.mode()
            logp = dist.log_prob(action)
        else:
            action, logp = dist.sample_and_log_prob(seed=key)
        return action, dict(
            logp=logp,
            **info,
        )

    def value(self, params: Params, obs: Any) -> jnp.ndarray:  # optional
        raise NotImplementedError

# A uniform action agent

class BlankAgent(Agent):
    def init(self, key: PRNGKey, sample_obs: Any, action_space: Any) -> Params:
        self.action_space = action_space
        self.batch_dim = sample_obs.shape[0]
        return None

    def policy(self, params: Params, obs: Any) -> Tuple[Any, Mapping[str, Any]]:
        pi = Categorical(logits=jnp.ones((self.batch_dim,) + (self.action_space.n,)))
        return pi, {}

    def value(self, params: Params, obs: Any) -> jnp.ndarray:
        return jnp.zeros(obs.shape[:-1])

# An MLP-based Actor-Critic

class MLP(nn.Module):
    layer_sizes: List[int]

    def setup(self):
        self.dense_layers = [
            nn.Dense(size, kernel_init=nn.initializers.orthogonal())
            for size in self.layer_sizes
        ]

    def __call__(self, x):
        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        return self.dense_layers[-1](x)


class MLPPolicyValue(Agent):

    def __init__(self, policy_layers: Sequence = [32, 32], value_layers=[32, 32]):
        self.policy_layers = policy_layers
        self.value_layers = value_layers

    def init(self, key: PRNGKey, sample_obs: Any, action_space: Any) -> Params:
        self.action_space = action_space
        self.policy_module = MLP(layer_sizes=self.policy_layers + [action_space.n])
        self.value_module = MLP(layer_sizes=self.value_layers + [1])
        policy_key, value_key = jax.random.split(key, 2)
        policy_params = self.policy_module.init(policy_key, sample_obs)
        value_params = self.value_module.init(value_key, sample_obs)

        return {
            "policy": policy_params,
            "value": value_params,
        }

    @functools.partial(jax.jit, static_argnames=("self"))
    def policy(self, params: Params, obs: Any) -> Tuple[Any, Mapping[str, Any]]:
        logits = self.policy_module.apply(params["policy"], obs)
        pi = Categorical(logits=logits)
        return pi, {}

    @functools.partial(jax.jit, static_argnames=("self"))
    def value(self, params: Params, obs: Any) -> jnp.ndarray:
        return jnp.squeeze(self.value_module.apply(params["value"], obs), axis=-1)