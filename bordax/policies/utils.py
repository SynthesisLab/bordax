import flax.struct
from typing import Callable, Sequence, List, Dict, Any
import jax
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
class Value:
    init: Callable[..., Any]
    apply: Callable[..., jax.Array]

@flax.struct.dataclass
class ActorCritic:
    actor: Any
    critic: Any


ActorCriticMaker = Callable[[Environment, Dict[str, Any]], ActorCritic]


class MLP_dtsemnet_value(nn.Module):
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

        if n_leaves % self.action_dim != 0:
            appendice = jnp.zeros(
                ((self.action_dim - (n_leaves % self.action_dim)), 2 * n_nodes)
            )
            tree_representation = jnp.concatenate(
                (tree_representation, appendice), axis=0
            )
        x = x @ tree_representation.T

        # x = x.reshape((x.shape[0], -1, self.action_dim))
        # x = x.max(axis=1)

        return x


def policy_factory_mlp(actor_critic: ActorCritic):

    def make_policy(policy_params, deterministic=False):
        policy_mlp = actor_critic.actor

        def apply(obs, key):
            pi, features = policy_mlp.apply(policy_params, obs)
            if deterministic:
                return pi.mode(), {}
            action, log_prob = pi.sample_and_log_prob(seed=key)
            return action, {"log_prob": log_prob, "features": features}

        return apply

    def make_value(critic_params):
        value_mlp = actor_critic.critic

        def apply(obs):
            return value_mlp.apply(critic_params, obs)

        return apply

    return make_policy, make_value


@jax.jit
def epsilon_greedy_exploration(key, q_vals, eps):
    key_action, key_epsilon = jax.random.split(key)
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosed_actions = jnp.where(
        jax.random.uniform(key_epsilon, greedy_actions.shape) < eps,
        jax.random.randint(
            key_action,
            shape=greedy_actions.shape,
            minval=0,
            maxval=q_vals.shape[-1],
        ),
        greedy_actions,
    )
    return chosed_actions


def q_function_factory(actor):
    def make_q_function(params, deterministic=False):
        def apply(obs, key, epsilon):
            q_vals = actor.apply(params, obs)
            if deterministic:
                return jnp.argmax(actor.apply(params, obs), axis=-1)

            action = epsilon_greedy_exploration(key, q_vals, epsilon)

            return action

        return apply

    return make_q_function


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


def make_policy_dt_actions(obs_shape, action_dim, tree_depth):
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
