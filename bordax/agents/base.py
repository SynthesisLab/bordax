from bordax.environments.utils import EnvAdapter
from bordax.types import PRNGKey, Params
from distrax import DistributionLike, Categorical

import jax
import jax.numpy as jnp
import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Mapping, Tuple, List
import flax.linen as nn
import functools


class Agent(ABC):
    @abstractmethod
    def init(self, key: PRNGKey, sample_obs: Any) -> Params: ...

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


class MLP_dtsemnet(nn.Module):
    tree_depth: int
    action_dim: int

    def setup(self):
        self.weights = nn.Dense(
            (2 ** (self.tree_depth) - 1),
            kernel_init=nn.initializers.orthogonal(),
            bias_init=nn.initializers.uniform(),
        )

    def __call__(self, x):

        if len(x.shape) == 1:
            x = jnp.array([x])

        x = self.weights(x)

        n_nodes = 2 ** (self.tree_depth) - 1
        n_leaves = n_nodes + 1

        row_indices = jnp.arange(2 * n_nodes)
        col_indices = jnp.arange(n_nodes).repeat(2)
        tiles = jnp.tile(jnp.array([1.0, -1.0]), n_nodes)
        matrix = jnp.zeros((2 * n_nodes, n_nodes), dtype=jnp.float32)
        matrix = matrix.at[row_indices, col_indices].set(tiles)

        x = nn.relu(x @ matrix.T)

        tree_representation = jnp.ones((n_leaves, 2 * n_nodes))
        for i in range(n_leaves):
            virtual_index = i + n_nodes
            relevant_indices = jnp.zeros(self.tree_depth - 1)
            replacement = jnp.ones(2 * n_nodes)
            for j in range(self.tree_depth):
                new_virtual_index = (virtual_index - 1) // 2
                relevant_indices = relevant_indices.at[self.tree_depth - j].set(
                    new_virtual_index
                )
                if virtual_index % 2 == 0:
                    replacement_tile = jnp.array([0, 1])
                else:
                    replacement_tile = jnp.array([1, 0])
                virtual_index = new_virtual_index
                replacement = replacement.at[
                    2 * virtual_index : 2 * virtual_index + 2
                ].set(replacement_tile)
            tree_representation = tree_representation.at[i].set(replacement)

        appendice = jnp.zeros(
            ((self.action_dim - (n_leaves % self.action_dim)), 2 * n_nodes)
        )
        tree_representation = jnp.concatenate((tree_representation, appendice), axis=0)

        x = x @ tree_representation.T

        x = x.reshape((x.shape[0], -1, self.action_dim))
        x = x.max(axis=1)

        return x


class MLP_boolean(nn.Module):
    n: int
    action_dim: int

    def setup(self):
        self.weights = nn.Dense(
            self.n,
            kernel_init=nn.initializers.orthogonal(),
            bias_init=nn.initializers.uniform(),
        )

    def __call__(self, x):

        if len(x.shape) == 1:
            x = jnp.array([x])

        x = self.weights(x)

        numbers = np.arange(2**self.n)

        binary_strings = [np.binary_repr(num, width=self.n) for num in numbers]

        function_representation = np.array(
            [[1 if char == "1" else -1 for char in binary] for binary in binary_strings]
        )
        function_representation = jnp.array(function_representation)

        x = x @ function_representation.T

        x = x.reshape((x.shape[0], -1, self.action_dim))
        x = x.max(axis=1)

        return x


class MLPPolicyValue(Agent):

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

    def init(self, key: PRNGKey, sample_obs: Any) -> Params:

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
        if isinstance(logits, tuple):
            logits = logits[0]
        pi = Categorical(logits=logits)
        return pi, {}

    def policy_activations(self, params, obs):
        logits, state = self.policy_module.apply(
            params["policy"], obs, capture_intermediates=True, mutable=["intermediates"]
        )
        intermediates = state["intermediates"]
        if isinstance(logits, tuple):
            logits = logits[0]
        pi = Categorical(logits=logits)
        return pi, intermediates

    @functools.partial(jax.jit, static_argnames=("self"))
    def value(self, params: Params, obs: Any) -> jnp.ndarray:
        value_out = self.value_module.apply(params["value"], obs)
        if isinstance(value_out, tuple):
            value_out = value_out[0]
        return jnp.squeeze(value_out, axis=-1)
