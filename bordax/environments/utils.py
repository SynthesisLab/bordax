import jax
import jax.numpy as jnp
import chex

from abc import ABC, abstractmethod
import functools

import gymnax
import gymnasium

from typing import TYPE_CHECKING, Any, Tuple, Mapping, Dict
from bordax.types import PRNGKey

EnvState = Any
EnvObs = Any
Space = Any

if TYPE_CHECKING:
    from dataclasses import dataclass
else:
    from chex import dataclass

# we suppose that the environment is vectorised *by default*.
# if the n is not given, then it's one


@dataclass(frozen=True)
class EnvParams:
    max_steps_in_episode: int


class EnvAdapter(ABC):
    is_jittable: bool
    num_envs: int
    env: Any
    env_params: Any

    @abstractmethod
    def reset(self, key: PRNGKey) -> Tuple[EnvObs, EnvState]: ...

    @abstractmethod
    def step(
        self, key: PRNGKey, state: EnvState, action: Any
    ) -> Tuple[Any, EnvState, float, bool, Mapping[str, Any]]: ...

    @abstractmethod
    def action_space(self) -> Space: ...

    @abstractmethod
    def obs_space(self) -> Space: ...


class EnvGymnaxAdapter(EnvAdapter):
    def __init__(self, env_name: str, env_config, num_envs: int = 1):
        self.is_jittable = True
        self.num_envs = num_envs
        self.config = env_config

        prefix, name = env_name.split("/", 1)
        if prefix == "gymnax":
            self.env, self.env_params = gymnax.make(name, **self.config["init_config"])
        else:
            raise ValueError(f"Unknown environment prefix: {prefix}")

        self.reset_v = jax.vmap(self.env.reset, in_axes=(0,))
        self.step_v = jax.vmap(self.env.step, in_axes=(0, 0, 0))

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKey) -> Tuple[EnvState, Any]:
        keys = jax.random.split(key, self.num_envs)
        obs, state = self.reset_v(keys)
        return obs, state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self, key: PRNGKey, state: EnvState, action: Any
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        keys = jax.random.split(key, self.num_envs)
        obs, state, reward, done, info = self.step_v(keys, state, action)
        return obs, state, reward, done, info

    def action_space(self):
        return self.env.action_space()

    def obs_space(self):
        return self.env.observation_space(self.env_params)


class EnvGymnasiumAdapter(EnvAdapter):
    def __init__(self, env_name: str, env_config, num_envs: int = 1):
        self.is_jittable = False
        self.num_envs = num_envs
        self.config = env_config

        prefix, name = env_name.split("/", 1)
        if prefix == "gymnasium":
            self.env = gymnasium.make_vec(name, num_envs=self.num_envs, **self.config["init_config"])

        self.env_params = EnvParams(
            max_steps_in_episode=self.env.spec.max_episode_steps
        )

    def reset(self, key: PRNGKey):
        seed = jax.random.key_data(key)[1].item()
        obs, info = self.env.reset(seed=seed, **self.config["reset_config"])
        return obs, obs

    def step(self, key: PRNGKey, state: Any, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated | truncated
        return obs, obs, reward, done, info

    def action_space(self):
        return self.env.single_action_space

    def obs_space(self):
        return self.env.single_observation_space


def make_env(env_name: str, env_config, num_envs: int = 1) -> EnvAdapter:

    if len(env_name.split("/")) > 1:
        # the prefix indicates what type environment to use
        if env_name.split("/")[0] == "gymnax":
            return EnvGymnaxAdapter(env_name, env_config, num_envs)
        elif env_name.split("/")[0] == "gymnasium":
            return EnvGymnasiumAdapter(env_name, env_config, num_envs)
        else:
            raise ValueError(f"Unknown environment prefix: {env_name.split("/")[0]}")
    else:
        raise ValueError(
            "Environment name must include a prefix (e.g., 'gymnax/CartPole-v1')."
        )
