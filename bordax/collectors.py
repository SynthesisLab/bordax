from bordax.agents.base import Agent
from bordax.environments.utils import EnvAdapter, EnvState
from bordax.types import PRNGKey, Params

from typing import Any, Tuple
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
import functools


class Collector(ABC):

    @abstractmethod
    def __call__(
        self,
        key: PRNGKey,
        env: EnvAdapter,
        obs_state: Tuple[Any, EnvState],
        agent: Agent,
        params: Params,
    ) -> Tuple[PRNGKey, EnvState, Any]: ...


class OnPolicyCollector(Collector):
    def __init__(self, rollout_length: int = 1024, gamma: float = 0.99, _lambda: float = 0.99):
        self.rollout_len = rollout_length
        self.gamma = gamma
        self._lambda = _lambda

    @functools.partial(jax.jit, static_argnames=("self", "agent", "env"))
    def __call__(self, key, env, obs, env_state, agent: Agent, params):
        init_obs, init_state = obs, env_state

        if env.is_jittable:

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
