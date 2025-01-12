import jax
import jax.numpy as jnp

from typing import Any

EnvState = Any

@jax.jit
def compute_gae(traj_batch, last_value, values, gamma, gae_lambda):
    def _get_advantages(gae_and_next_value, transition):
        
        gae, next_value = gae_and_next_value
        transition, value = transition
        done, reward = (
            transition.done,
            transition.reward,
        )

        delta = reward + gamma * next_value * (1 - done) - value
        gae = (
            delta
            + gamma * gae_lambda * (1 - done) * gae
        )

        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), last_value),
        (traj_batch, values),
        reverse=True, 
    )
    return advantages, advantages + values