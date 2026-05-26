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
    """Unified interface over Gymnax and Gymnasium environments.

    All environments are treated as vectorised: ``reset`` and ``step``
    operate on a batch of ``num_envs`` parallel episodes.

    Attributes:
        is_jittable: Whether the environment can be used inside
            ``jax.jit`` / ``jax.lax.scan``. ``True`` for Gymnax,
            ``False`` for Gymnasium.
        num_envs: Number of parallel environment instances.
        env: The underlying environment object.
        env_params: Static environment parameters (e.g. episode length).
    """

    is_jittable: bool
    num_envs: int
    env: Any
    env_params: Any

    @abstractmethod
    def reset(self, key: PRNGKey) -> Tuple[EnvObs, EnvState]:
        """Reset all environments and return initial observations.

        Args:
            key: JAX random key used to seed the reset.

        Returns:
            Tuple of ``(obs, state)`` with leading batch dimension
            ``num_envs``.
        """
        ...

    @abstractmethod
    def step(
        self, key: PRNGKey, state: EnvState, action: Any
    ) -> Tuple[Any, EnvState, float, bool, Mapping[str, Any]]:
        """Step all environments forward by one timestep.

        Args:
            key: JAX random key for stochastic transitions.
            state: Current environment state (batch of ``num_envs``).
            action: Actions to apply, shape ``(num_envs, ...)``.

        Returns:
            Tuple of ``(obs, state, reward, done, info)``.
        """
        ...

    @abstractmethod
    def action_space(self) -> Space:
        """Return the single-environment action space."""
        ...

    @abstractmethod
    def obs_space(self) -> Space:
        """Return the single-environment observation space."""
        ...


class EnvGymnaxAdapter(EnvAdapter):
    """Adapter for Gymnax environments (fully JIT-compilable).

    Wraps a Gymnax environment with ``jax.vmap`` across ``num_envs``
    parallel instances. Both ``reset`` and ``step`` are JIT-compiled.
    The entire training loop can be compiled end-to-end when using this
    adapter with an on-policy algorithm.
    """

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
    """Adapter for standard Gymnasium environments (non-JIT).

    Wraps a vectorised Gymnasium environment (``gymnasium.make_vec``).
    Only the gradient update step can be JIT-compiled; environment
    stepping runs in Python.
    """

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
    """Create a vectorised environment adapter by name.

    Args:
        env_name: Environment identifier with a backend prefix, e.g.
            ``"gymnax/CartPole-v1"`` or ``"gymnasium/CartPole-v1"``.
            The prefix determines whether the adapter is JIT-compilable.
        env_config: Backend-specific config dict. For Gymnax, pass
            ``{"init_config": {}, "reset_config": {}}``. For Gymnasium,
            pass ``{"init_config": {}, "reset_config": {}}``.
        num_envs: Number of parallel environment instances.

    Returns:
        An ``EnvGymnaxAdapter`` (``is_jittable=True``) or
        ``EnvGymnasiumAdapter`` (``is_jittable=False``).

    Raises:
        ValueError: If the prefix is unknown or if no prefix is given.
    """

    if len(env_name.split("/")) > 1:
        # the prefix indicates what type environment to use
        if env_name.split("/")[0] == "gymnax":
            return EnvGymnaxAdapter(env_name, env_config, num_envs)
        elif env_name.split("/")[0] == "gymnasium":
            return EnvGymnasiumAdapter(env_name, env_config, num_envs)
        else:
            raise ValueError(f"Unknown environment prefix: {env_name.split('/')[0]}")
    else:
        raise ValueError(
            "Environment name must include a prefix (e.g., 'gymnax/CartPole-v1')."
        )
