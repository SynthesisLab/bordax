from bordax.policies.utils import Policy, ActorCritic
from bordax.environments.utils import Environment, EnvGymnax

from bordax.policies.mlp import make_value_mlp

import flax.linen as nn
import jax.numpy as jnp
import distrax
import numpy as np

from bordax.environments.pomdp.pomdp import BeliefWrapper

import gymnasium

# n hyperplanes, boolean combinations for up to 2^n regions

class MLP_boolean(nn.Module):
    n: int
    action_dim: int

    def setup(self):
        self.weights = nn.Dense(self.n, kernel_init=nn.initializers.orthogonal(), bias_init=nn.initializers.uniform())

    def __call__(self, x):
        
            if len(x.shape) == 1:
                x = jnp.array([x])

            x = self.weights(x)
            
            numbers = np.arange(2**self.n)
    
            # Convert numbers to binary with 'bits' bits
            binary_strings = [np.binary_repr(num, width=self.n) for num in numbers]
            
            # Convert binary strings to 2D matrix with -1 and 1
            function_representation = np.array([[1 if char == '1' else -1 for char in binary] for binary in binary_strings])
            function_representation = jnp.array(function_representation)

            x = x @ function_representation.T

            x = x.reshape((x.shape[0], -1, self.action_dim))
            x = x.max(axis=1)

            return x

def make_policy_boolean(
    obs_shape,
    action_dim,
    n
):
    policy_module = MLP_boolean(action_dim=action_dim, n=n)

    def apply(policy_params, obs):
        pi = distrax.Categorical(logits=policy_module.apply(policy_params, obs))
        return pi

    obs_size = obs_shape[0]
    dummy_obs = jnp.zeros((1, obs_size))
    return Policy(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

def make_actor_critic_boolean(env, **kwargs) -> ActorCritic:
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

    policy = make_policy_boolean(obs_shape, action_dim, 3)
    value = make_value_mlp(obs_shape)

    return ActorCritic(actor=policy, critic=value)