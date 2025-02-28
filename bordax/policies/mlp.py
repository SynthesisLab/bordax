import flax.struct
from typing import Callable, Sequence, List, Dict, Any
import jax.numpy as jnp
import numpy as np

from bordax.environments.utils import Environment, EnvGymnasium, EnvGymnax

from bordax.policies.utils import Policy, ActorCritic
from bordax.environments.utils import Environment, EnvGymnax
from bordax.environments.pomdp.pomdp import BeliefWrapper

import gymnasium


import flax
import flax.linen as nn
import distrax

# standard mlp actor-critic

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
    

def make_policy_mlp(
    obs_shape,
    action_dim,
    hidden_layer_sizes: Sequence[int] = (10, 10),
):
    policy_module = MLP(layer_sizes=list(hidden_layer_sizes) + [action_dim])

    def apply(policy_params, obs):
        pi = distrax.Categorical(logits=policy_module.apply(policy_params, obs))
        return pi

    obs_size = obs_shape[0]
    dummy_obs = jnp.zeros((1, obs_size))
    return Policy(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


def make_value_mlp(
    obs_shape,
    hidden_layer_sizes: Sequence[int] = (32, 32),
):
    value_module = MLP(layer_sizes=list(hidden_layer_sizes) + [1])

    def apply(value_params, obs):
        return jnp.squeeze(value_module.apply(value_params, obs),axis=-1)

    obs_size = obs_shape[0]
    dummy_obs = jnp.zeros((1, obs_size))
    return Policy(init=lambda key: value_module.init(key, dummy_obs), apply=apply)

def make_actor_critic_mlp(env, **kwargs) -> ActorCritic:
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

    return ActorCritic(actor=policy, critic=value)

def make_q_mlp(env, **kwargs):
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

    q_module = MLP(layer_sizes=[128, 128, action_dim])

    def apply(q_params, obs):
        q_values = q_module.apply(q_params, obs)
        return q_values

    obs_size = obs_shape[0]
    dummy_obs = jnp.zeros((1, obs_size))
    return Policy(init=lambda key: q_module.init(key, dummy_obs), apply=apply)