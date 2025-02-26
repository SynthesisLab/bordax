import flax.struct
from typing import Callable, Sequence, List, Dict, Any
import jax.numpy as jnp
import numpy as np

from bordax.environments.utils import Environment, EnvGymnasium, EnvGymnax

import gymnasium

import flax
import flax.linen as nn
import distrax


@flax.struct.dataclass
class ActorCriticParams:
    actor_params: Any
    critic_params: Any

@flax.struct.dataclass
class Policy:
    init: Callable[..., Any]
    apply: Callable[..., Any]

@flax.struct.dataclass
class ActorCritic:
    actor: Any
    critic: Any

ActorCriticMaker = Callable[[Environment, Dict[str, Any]], ActorCritic]


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
        self.weights = nn.Dense((2**(self.tree_depth) - 1), kernel_init=nn.initializers.orthogonal(), bias_init=nn.initializers.uniform())

    def __call__(self, x):
        
            if len(x.shape) == 1:
                x = jnp.array([x])

            x = self.weights(x)
            
            n_nodes = 2**(self.tree_depth) - 1
            n_leaves = n_nodes + 1

            row_indices = jnp.arange(2 * n_nodes)
            col_indices = jnp.arange(n_nodes).repeat(2)
            tiles = jnp.tile(jnp.array([1.0, -1.0]), n_nodes)
            matrix = jnp.zeros((2 * n_nodes, n_nodes), dtype=jnp.float32)
            matrix = matrix.at[row_indices, col_indices].set(tiles)

            x = nn.relu(x @ matrix.T)

            tree_representation = jnp.ones((n_leaves, 2*n_nodes))
            for i in range(n_leaves):
                virtual_index = i + n_nodes
                relevant_indices = jnp.zeros(self.tree_depth-1)
                replacement = jnp.ones(2*n_nodes)
                for j in range(self.tree_depth):
                    new_virtual_index = ((virtual_index - 1) // 2)
                    relevant_indices = relevant_indices.at[self.tree_depth - j].set(new_virtual_index)
                    if virtual_index % 2 == 0:
                        replacement_tile = jnp.array([0, 1])
                    else:
                        replacement_tile = jnp.array([1, 0])
                    virtual_index = new_virtual_index
                    replacement = replacement.at[2*virtual_index : 2*virtual_index + 2].set(replacement_tile)
                tree_representation = tree_representation.at[i].set(replacement)

            if n_leaves % self.action_dim != 0:
                appendice = jnp.zeros(((self.action_dim - (n_leaves % self.action_dim)), 2 * n_nodes))
                tree_representation = jnp.concatenate((tree_representation, appendice), axis=0)
            
            x = x @ tree_representation.T

            x = x.reshape((x.shape[0], -1, self.action_dim))
            x = x.max(axis=1)

            return x
    
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

class MLP_dtsemnet_value(nn.Module):
    tree_depth: int
    action_dim: int

    def setup(self):
        self.weights = nn.Dense((2**(self.tree_depth) - 1), kernel_init=nn.initializers.orthogonal(), bias_init=nn.initializers.uniform())

    def __call__(self, x):
        
            if len(x.shape) == 1:
                x = jnp.array([x])

            x = self.weights(x)
            
            n_nodes = 2**(self.tree_depth) - 1
            n_leaves = n_nodes + 1

            row_indices = jnp.arange(2 * n_nodes)
            col_indices = jnp.arange(n_nodes).repeat(2)
            tiles = jnp.tile(jnp.array([1.0, -1.0]), n_nodes)
            matrix = jnp.zeros((2 * n_nodes, n_nodes), dtype=jnp.float32)
            matrix = matrix.at[row_indices, col_indices].set(tiles)

            x = nn.relu(x @ matrix.T)

            tree_representation = jnp.ones((n_leaves, 2*n_nodes))
            for i in range(n_leaves):
                virtual_index = i + n_nodes
                relevant_indices = jnp.zeros(self.tree_depth-1)
                replacement = jnp.ones(2*n_nodes)
                for j in range(self.tree_depth):
                    new_virtual_index = ((virtual_index - 1) // 2)
                    relevant_indices = relevant_indices.at[self.tree_depth - j].set(new_virtual_index)
                    if virtual_index % 2 == 0:
                        replacement_tile = jnp.array([0, 1])
                    else:
                        replacement_tile = jnp.array([1, 0])
                    virtual_index = new_virtual_index
                    replacement = replacement.at[2*virtual_index : 2*virtual_index + 2].set(replacement_tile)
                tree_representation = tree_representation.at[i].set(replacement)

            if n_leaves % self.action_dim != 0:
                appendice = jnp.zeros(((self.action_dim - (n_leaves % self.action_dim)), 2 * n_nodes))
                tree_representation = jnp.concatenate((tree_representation, appendice), axis=0)
            x = x @ tree_representation.T

            # x = x.reshape((x.shape[0], -1, self.action_dim))
            # x = x.max(axis=1)

            return x

def policy_factory_mlp(actor_critic):
    
    def make_policy(policy_params, deterministic=False):
        policy_mlp = actor_critic.actor
        def apply(obs, key):
            pi: distrax.DistributionLike = policy_mlp.apply(policy_params, obs)
            if deterministic:
                return pi.mode(), {}
            action = pi.sample(seed=key)
            log_prob = pi.log_prob(action)
            return action, {"log_prob": log_prob}
        return apply
    
    def make_value(critic_params):
        value_mlp = actor_critic.critic
        def apply(obs):
            return value_mlp.apply(critic_params, obs)
        return apply

    return make_policy, make_value

# standard mlp actor-critic

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
        action_dim = env.action_space.shape[0]
    
    else:
        raise NotImplementedError

    policy = make_policy_mlp(obs_shape, action_dim)
    value = make_value_mlp(obs_shape)

    return ActorCritic(actor=policy, critic=value)

# dtsemnet : every node defines a hyperplane

def make_policy_dtsemnet(
    obs_shape,
    action_dim,
    tree_depth
):
    policy_module = MLP_dtsemnet(action_dim=action_dim, tree_depth=tree_depth)

    def apply(policy_params, obs):
        pi = distrax.Categorical(logits=policy_module.apply(policy_params, obs))
        return pi

    obs_size = obs_shape[0]
    dummy_obs = jnp.zeros((1, obs_size))
    return Policy(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

def make_actor_critic_dtsemnet(env, **kwargs) -> ActorCritic:
    if isinstance(env, EnvGymnax):
        obs_space = env.observation_space(kwargs["env_params"])
        obs_shape = obs_space.shape
        action_dim = env.action_space(kwargs["env_params"]).n
    else:
        raise NotImplementedError

    # tree_depth = kwargs["tree_depth"]
    policy = make_policy_dtsemnet(obs_shape, action_dim, 4)
    value = make_value_mlp(obs_shape)

    return ActorCritic(actor=policy, critic=value)

# n hyperplanes, boolean combinations for up to 2^n regions

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
    else:
        raise NotImplementedError

    policy = make_policy_boolean(obs_shape, action_dim, 3)
    value = make_value_mlp(obs_shape)

    return ActorCritic(actor=policy, critic=value)


def make_actor_critic_dtsemnet_value(env, **kwargs) -> ActorCritic:
    if isinstance(env, EnvGymnax):
        obs_space = env.observation_space(kwargs["env_params"])
        obs_shape = obs_space.shape
        action_dim = env.action_space(kwargs["env_params"]).n
    else:
        raise NotImplementedError
    
    module = MLP_dtsemnet_value(action_dim=action_dim, tree_depth=3)
    def apply_policy(params, obs):
        x = module.apply(params, obs)
        x = x.reshape((x.shape[0], -1, action_dim))
        x = x.max(axis=1)
        pi = distrax.Categorical(logits=x)
        return pi
    
    def apply_value(params, obs):
        x = module.apply(params, obs)
        x = x.min(axis=-1)
        return jnp.array(x)

    obs_size = obs_shape[0]
    dummy_obs = jnp.zeros((1, obs_size))
    policy = Policy(init=lambda key: module.init(key, dummy_obs), apply=apply_policy)
    value = Policy(init=lambda key: module.init(key, dummy_obs), apply=apply_value)

    return ActorCritic(actor=policy, critic=value)

# add something over all leaves of dtsemnet

def make_policy_dt_actions(
    obs_shape,
    action_dim,
    tree_depth
):
    policy_module = MLP_dtsemnet(action_dim=action_dim, tree_depth=tree_depth)

    def apply(policy_params, obs):
        pi = distrax.Categorical(logits=policy_module.apply(policy_params, obs))
        return pi

    obs_size = obs_shape[0]
    dummy_obs = jnp.zeros((1, obs_size))
    return Policy(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)

def make_actor_critic_dt_actions(env, **kwargs) -> ActorCritic:
    if isinstance(env, EnvGymnax):
        obs_space = env.observation_space(kwargs["env_params"])
        obs_shape = obs_space.shape
        action_dim = env.action_space(kwargs["env_params"]).n
    else:
        raise NotImplementedError

    # tree_depth = kwargs["tree_depth"]
    policy = make_policy_dt_actions(obs_shape, action_dim, 3)
    value = make_value_mlp(obs_shape)

    return ActorCritic(actor=policy, critic=value)