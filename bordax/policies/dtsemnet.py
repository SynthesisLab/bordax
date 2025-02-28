from bordax.policies.utils import Policy, ActorCritic
from bordax.environments.utils import Environment, EnvGymnax

from bordax.policies.mlp import make_value_mlp

import flax.linen as nn
import jax.numpy as jnp
import distrax
import numpy as np

from bordax.environments.pomdp.pomdp import BeliefWrapper

import gymnasium

# dtsemnet : every node defines a hyperplane

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
    elif isinstance(env, gymnasium.vector.VectorEnv):
        obs_shape = env.single_observation_space.shape
        action_dim = env.single_action_space.n
    elif isinstance(env, BeliefWrapper):
        obs_shape = env.observation_space.shape
        action_dim = env.action_space.n
    else:
        raise NotImplementedError

    # tree_depth = kwargs["tree_depth"]
    policy = make_policy_dtsemnet(obs_shape, action_dim, 5)
    value = make_value_mlp(obs_shape)

    return ActorCritic(actor=policy, critic=value)

