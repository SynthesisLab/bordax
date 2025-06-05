from abc import ABC, abstractmethod
from typing import Mapping, Tuple, Any, Sequence
from bordax.types import PRNGKey
import jax
import jax.numpy as jnp
import functools


class BatchBuilder(ABC):
    @abstractmethod
    def __call__(
        self, key: PRNGKey, buffer: Any
    ) -> Tuple[PRNGKey, Mapping[str, jnp.ndarray]]: ...

class FullBufferBatch(BatchBuilder):

    def __init__(self, buffer_size, num_env):
        self.buffer_size = buffer_size
        self.num_env = num_env

    def __call__(self, key: PRNGKey, buffer: Any) -> Tuple[PRNGKey, Mapping[str, jnp.ndarray]]:
        # Sample a batch from the buffer
        # the buffer is a (possibly nested) dictionary with entries of the shape (Time, Batch, shape)

        key, perm_key = jax.random.split(key, 2)

        # flatten the batch from several environments
        batch_size = self.buffer_size * self.num_env
        batch = jax.tree.map(
            lambda x: x.reshape((batch_size,) + x.shape[2:]), buffer
        )

        # shuffling
        permutation = jax.random.permutation(perm_key, batch_size)
        batch = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0), batch
        )

        return batch    

class MiniBatch(BatchBuilder):
    def __init__(self, num_minibatches: int):
        self.num_minibatches = num_minibatches

    def __call__(
        self, key: PRNGKey, buffer: Any
    ) -> Tuple[PRNGKey, Mapping[str, jnp.ndarray]]:

        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((self.num_minibatches, -1) + x.shape[1:]), buffer
        )

        return minibatches


class NormalizeAdvantages(BatchBuilder):
    """Normalizes advantages per minibatch in a batch of minibatches."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, key: PRNGKey, buffer: Any) -> Any:

        def normalize_minibatch(minibatch_data):
            advantages = minibatch_data["advantages"]
            adv_mean = jnp.mean(advantages)
            adv_std = jnp.std(advantages)
            normalized_advantages = (advantages - adv_mean) / (adv_std + self.eps)

            return {**minibatch_data, "advantages": normalized_advantages}

        normalized_buffer = jax.vmap(normalize_minibatch)(buffer)

        return normalized_buffer

class ComposedBatchBuilder(BatchBuilder):
    def __init__(self, batch_builders: Sequence[BatchBuilder]):
        self.batch_builders = batch_builders
    @functools.partial(jax.jit, static_argnames=("self"))
    def __call__(
        self, key: PRNGKey, buffer: Any
    ) -> Tuple[PRNGKey, Mapping[str, jnp.ndarray]]:
        keys = jax.random.split(key, len(self.batch_builders))
        for i, batch_builder in enumerate(self.batch_builders):
            buffer = batch_builder(keys[i], buffer)

        return buffer