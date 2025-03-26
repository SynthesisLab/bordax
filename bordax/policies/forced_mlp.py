import flax.struct
from typing import Callable, Sequence, List, Dict, Any
import jax.numpy as jnp
import numpy as np

from bordax.environments.utils import Environment, EnvGymnasium, EnvGymnax

from bordax.policies.utils import Policy, PolicyValue
from bordax.environments.utils import Environment, EnvGymnax
from bordax.environments.pomdp.pomdp import BeliefWrapper

from bordax.policies.mlp import MLP as StandardMLP

import gymnasium


import flax
import flax.linen as nn
import distrax

# standard mlp policy-value


class PieceMLP(nn.Module):
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


class MLP(nn.Module):
    layer_sizes: List[List[int]]

    def setup(self):
        self.pieces = [PieceMLP(layer_sizes) for layer_sizes in self.layer_sizes]

    def __call__(self, x):
        output = x
        intermediate_outputs = []
        for i, piece in enumerate(self.pieces[:-1]):
            output = nn.softmax(piece(output))
            intermediate_outputs.append(output)
            # self.sow(f"pieces", f"{i}", output)
            output = jnp.concat([output, x], axis=-1)

        return self.pieces[-1](output), intermediate_outputs


def make_policy_mlp(
    obs_shape,
    action_dim,
    hidden_layer_sizes: Sequence[Sequence[int]] = ((1,),),
):

    policy_module = MLP(layer_sizes=list(hidden_layer_sizes) + [[action_dim]])

    def apply(policy_params, obs):
        logits, features = policy_module.apply(policy_params, obs)
        pi = distrax.Categorical(logits=logits)
        activation_distributions = [
            distrax.Categorical(probs=probs) for probs in features
        ]
        return pi, activation_distributions

    obs_size = obs_shape[0]
    dummy_obs = jnp.zeros((1, obs_size))
    return Policy(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_value_mlp(
    obs_shape,
    hidden_layer_sizes: Sequence[int] = (32, 32),
):
    value_module = StandardMLP(layer_sizes=list(hidden_layer_sizes) + [1])

    def apply(value_params, obs):
        return jnp.squeeze(value_module.apply(value_params, obs), axis=-1)

    obs_size = obs_shape[0]
    dummy_obs = jnp.zeros((1, obs_size))
    return Policy(init=lambda key: value_module.init(key, dummy_obs), apply=apply)


def make_policy_value_mlp(env, **kwargs) -> PolicyValue:
    if isinstance(env, EnvGymnax):
        obs_space = env.observation_space(kwargs["env_params"])
        obs_shape = obs_space.shape
        action_dim = env.action_space(kwargs["env_params"]).n
    elif isinstance(env, gymnasium.vector.VectorEnv):
        obs_shape = env.observation_space.shape[1:]
        action_dim = env.single_action_space.n
    elif isinstance(env, BeliefWrapper):
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.n

    else:
        raise NotImplementedError

    policy = make_policy_mlp(obs_shape, action_dim)
    value = make_value_mlp(obs_shape)

    return PolicyValue(policy=policy, value=value)
