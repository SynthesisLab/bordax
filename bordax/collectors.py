from bordax.agents.base import Agent
from bordax.environments.utils import EnvAdapter, EnvState, EnvObs
from bordax.types import PRNGKey, Params

from typing import Any, Tuple
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
import functools
import numpy as np


def create_rollout_buffer(env_spec, num_envs, num_steps) -> dict:
    buffer = {
        "obs": jnp.zeros((num_steps, num_envs) + env_spec["obs_shape"]),
        "action": jnp.zeros((num_steps, num_envs) + env_spec["action_shape"]),
        "reward": jnp.zeros((num_steps, num_envs)),
        "done": jnp.zeros((num_steps, num_envs), dtype=jnp.bool),
        "info": {"logp": jnp.zeros((num_steps, num_envs))},
    }

    return buffer


class Collector(ABC):

    @abstractmethod
    def __call__(
        self,
        key: PRNGKey,
        env: EnvAdapter,
        obs: EnvObs,
        env_state: EnvState,
        agent: Agent,
        params: Params,
    ) -> Tuple[PRNGKey, EnvState, Any]: ...


class OnPolicyCollector(Collector):
    def __init__(
        self, rollout_length: int = 1024, gamma: float = 0.99, _lambda: float = 0.99
    ):
        self.rollout_len = rollout_length
        self.gamma = gamma
        self._lambda = _lambda

    @functools.partial(jax.jit, static_argnames=("self", "agent", "env"))
    def collect_jittable(self, key, env, obs, env_state, agent: Agent, params):
        init_obs, init_state = obs, env_state

        def one_step(carry, unused):
            key, obs, env_state = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, info = agent.action(params, obs, act_key)

            n_obs, n_env_state, reward, done, env_info = env.step(
                env_key, env_state, action
            )

            transition = dict(
                obs=obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )

            return (key, n_obs, n_env_state), transition

        (key, last_obs, last_env_state), traj = jax.lax.scan(
            one_step,
            (key, init_obs, init_state),
            None,
            length=self.rollout_len,
        )

        return (key, last_obs, last_env_state), traj

    def collect_non_jittable(
        self, key, env: EnvAdapter, obs, env_state, agent: Agent, params
    ):
        env_spec = dict(
            obs_shape=env.obs_space().shape, action_shape=env.action_space().shape
        )
        buffer = create_rollout_buffer(env_spec, env.num_envs, self.rollout_len)

        for i in range(self.rollout_len):
            key, act_key, env_key = jax.random.split(key, 3)
            buffer["obs"] = buffer["obs"].at[i].set(obs)
            action, action_info = agent.action(params, obs, act_key)

            n_obs, n_env_state, reward, done, env_info = env.step(
                act_key, env_state, np.asarray(action)
            )
            buffer["action"] = buffer["action"].at[i].set(action)
            buffer["reward"] = buffer["reward"].at[i].set(reward)
            buffer["done"] = buffer["done"].at[i].set(done)
            buffer["info"]["logp"] = (
                buffer["info"]["logp"].at[i].set(action_info["logp"])
            )
            obs = n_obs
            env_state = n_env_state

        # print(buffer["info"]["logp"].shape)

        return (obs, obs), buffer

    def __call__(self, key, env, obs, env_state, agent: Agent, params):

        if env.is_jittable:
            (key, last_obs, last_env_state), traj = self.collect_jittable(
                key, env, obs, env_state, agent, params
            )
        else:
            (last_obs, last_env_state), traj = self.collect_non_jittable(
                key, env, obs, env_state, agent, params
            )
        # calculating GAE
        last_value = agent.value(params, last_obs)
        values = agent.value(params, traj["obs"])

        advantages, targets = jax.lax.stop_gradient(
            compute_gae(traj, last_value, values, self.gamma, self._lambda)
        )

        traj["advantages"] = advantages
        traj["targets"] = targets

        return (last_obs, last_env_state), traj


@jax.jit
def compute_gae(traj_batch, last_value, values, gamma, gae_lambda):

    def _get_advantages(gae_and_next_value, transition):

        gae, next_value = gae_and_next_value
        transition, value = transition
        done, reward = (
            transition["done"],
            transition["reward"],
        )

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae

        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), last_value),
        (traj_batch, values),
        reverse=True,
    )

    return advantages, advantages + values


class EpsGreedyCollector(Collector):
    pass


class StochasticOffPolicyCollector(Collector):
    pass


class DeterministicOffPolicyCollector(Collector):
    pass
