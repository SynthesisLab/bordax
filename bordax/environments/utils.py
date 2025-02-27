import jax
import jax.numpy as jnp
import flax

import numpy as np

from gymnax.environments.environment import Environment as EnvGymnax
from gymnasium import Env as EnvGymnasium
from bordax.environments.pomdp.pomdp import BeliefWrapper
import gymnasium

from typing import Union, Callable, NamedTuple, Any

Environment = Union[EnvGymnax, EnvGymnasium]

EnvState = Any


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def generate_unroll(
    key: flax.typing.PRNGKey,
    env,
    policy: Callable[[Any, flax.typing.PRNGKey], Any],
    init_obs: Any,
    init_state: EnvState,
    unroll_length: int,
    **kwargs,
):
        if isinstance(env, EnvGymnax) or isinstance(env, BeliefWrapper):
            env_params = kwargs["env_params"]
            num_envs = kwargs["num_envs"]
            step_v = jax.vmap(env.step, in_axes=(0, 0, 0, None))

            def f(carry, unused_t):
                obs, state, current_key = carry
                action_key, env_key, n_key = jax.random.split(current_key, 3)
                action, policy_info = policy(obs, action_key)

                env_key_v = jax.random.split(env_key, num_envs)

                n_obs, n_state, reward, done, info = step_v(
                    env_key_v, state, action, env_params
                )
                transition = Transition(done, action, reward, policy_info["log_prob"], obs, info)
                return (n_obs, n_state, n_key), transition

            (final_obs, final_state, _), traj_batch = jax.lax.scan(
                f, (init_obs, init_state, key), None, length=unroll_length
            )

            return (final_obs, final_state), traj_batch
        elif isinstance(env, gymnasium.vector.VectorEnv):
            
            traj_batch = []
            obs = init_obs

            for i in range(unroll_length):
                action_key, key = jax.random.split(key, 2)
                action, policy_info = policy(obs, action_key)
                n_obs, reward, terminated, truncated, _ = env.step(np.asarray(action))
                done = terminated | truncated
                transition = Transition(done, action, reward, policy_info["log_prob"], obs, jnp.zeros_like(n_obs))
                obs = n_obs
                traj_batch.append(transition)

            traj_batch = transform_batch(traj_batch)
            return (n_obs, []), traj_batch

        else:
            raise NotImplementedError
        
@jax.jit
def transform_batch(traj_batch):
    return jax.tree_map(lambda *args: jnp.stack(args, axis=0), *traj_batch)