from typing import Any, NamedTuple

import chex
import optax
import jax.numpy as jnp

PRNGKey = chex.PRNGKey
Params = Any

GenericParameters = Any
PolicyParameters = Any
ValueParameters = Any 

class PolicyValueParameters(NamedTuple):
    policy: PolicyParameters
    value: ValueParameters

AgentParameters = PolicyValueParameters | GenericParameters

class TrainingState(NamedTuple):
    optimizer_state: optax.OptState
    params: Params
    step: jnp.ndarray
