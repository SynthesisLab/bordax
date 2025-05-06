from typing import Any, NamedTuple

import chex
import optax
import jax.numpy as jnp

PRNGKey = chex.PRNGKey
Params = Any


class TrainingState(NamedTuple):
    optimizer_state: optax.OptState
    params: Params
    step: jnp.ndarray
