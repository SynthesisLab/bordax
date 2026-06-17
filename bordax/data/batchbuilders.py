from abc import ABC, abstractmethod
from typing import Mapping, Tuple, Any, Sequence
from bordax.types import PRNGKey
import jax
import jax.numpy as jnp
import functools


class BatchBuilder(ABC):
    """Abstract base class for batch builders.

    A batch builder transforms a raw buffer (trajectory dict or replay
    buffer) into the format expected by the updater. Batch builders can
    be chained via ``ComposedBatchBuilder``.
    """

    @abstractmethod
    def __call__(
        self, key: PRNGKey, buffer: Any
    ) -> Tuple[PRNGKey, Mapping[str, jnp.ndarray]]:
        """Transform a buffer into a training batch.

        Args:
            key: JAX random key (for shuffling or sampling).
            buffer: Raw data — a trajectory dict (on-policy) or a
                ``ReplayBuffer`` instance (off-policy).

        Returns:
            A batch dict of JAX arrays ready for the updater.
        """
        ...

class FullBufferBatch(BatchBuilder):
    """Flatten and shuffle an entire on-policy rollout into a single batch.

    Merges the time and environment dimensions, then applies a random
    permutation. Typically the first stage in a ``ComposedBatchBuilder``
    for PPO, followed by ``MiniBatch``.
    """

    def __init__(self, buffer_size, num_env):
        """
        Args:
            buffer_size: Number of timesteps in the rollout.
            num_env: Number of parallel environments.
        """
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
    """Split a flat batch into equal-sized minibatches.

    Reshapes the leading dimension into ``(num_minibatches, minibatch_size)``.
    The resulting array is iterated over by the updater's SGD loop.
    """

    def __init__(self, num_minibatches: int):
        """
        Args:
            num_minibatches: Number of minibatches to split the batch into.
                The batch size must be divisible by this value.
        """
        self.num_minibatches = num_minibatches

    def __call__(
        self, key: PRNGKey, buffer: Any
    ) -> Tuple[PRNGKey, Mapping[str, jnp.ndarray]]:

        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape((self.num_minibatches, -1) + x.shape[1:]), buffer
        )

        return minibatches


class NormalizeAdvantagesTargets(BatchBuilder):
    """Normalizes advantages (and optionally value targets) per minibatch."""

    def __init__(self, eps: float = 1e-8, normalize_targets: bool = True):
        """
        Args:
            eps: Small constant added to the standard deviation for numerical
                stability.
            normalize_targets: If ``True``, also normalise value targets in
                addition to advantages.
        """
        self.eps = eps
        self.normalize_targets = normalize_targets

    def __call__(self, key: PRNGKey, buffer: Any) -> Any:

        def normalize_minibatch(minibatch_data):
            advantages = minibatch_data["advantages"]
            adv_mean = jnp.mean(advantages)
            adv_std = jnp.std(advantages)
            normalized_advantages = (advantages - adv_mean) / (adv_std + self.eps)

            normalized_targets = minibatch_data["targets"]
            if self.normalize_targets:
                targets = minibatch_data["targets"]
                target_mean = jnp.mean(targets)
                target_std = jnp.std(targets)
                normalized_targets = (targets - target_mean) / (target_std + self.eps)

            return {
                **minibatch_data,
                "advantages": normalized_advantages,
                "targets": normalized_targets,
            }

        normalized_buffer = jax.vmap(normalize_minibatch)(buffer)

        return normalized_buffer

class ComposedBatchBuilder(BatchBuilder):
    """Apply a sequence of batch builders in order.

    Each builder's output is passed as input to the next. The full
    composed call is JIT-compiled. Typical PPO usage::

        ComposedBatchBuilder((
            FullBufferBatch(rollout_length, num_envs),
            MiniBatch(num_minibatches),
            NormalizeAdvantagesTargets(),
        ))
    """

    def __init__(self, batch_builders: Sequence[BatchBuilder]):
        """
        Args:
            batch_builders: Ordered sequence of batch builders to apply.
        """
        self.batch_builders = batch_builders
    @functools.partial(jax.jit, static_argnames=("self"))
    def __call__(
        self, key: PRNGKey, buffer: Any
    ) -> Tuple[PRNGKey, Mapping[str, jnp.ndarray]]:
        keys = jax.random.split(key, len(self.batch_builders))
        for i, batch_builder in enumerate(self.batch_builders):
            buffer = batch_builder(keys[i], buffer)

        return buffer


class UniformReplayBatch(BatchBuilder):
    """Sample a batch uniformly from a ReplayBuffer."""
    
    def __init__(self, batch_size: int):
        """
        Args:
            batch_size: Number of transitions to sample per update.
        """
        self.batch_size = batch_size
    
    def __call__(self, key: PRNGKey, buffer: Any) -> Mapping[str, jnp.ndarray]:
        """Sample transitions from replay buffer and convert to JAX arrays.
        
        Args:
            key: PRNG key (unused, but kept for interface consistency)
            buffer: ReplayBuffer instance
            
        Returns:
            Dictionary of JAX arrays with keys: obs, action, reward, next_obs, done
        """
        # Sample from the numpy buffer
        batch_np = buffer.sample(self.batch_size)
        # Convert to JAX arrays
        batch_jax = jax.tree_util.tree_map(jnp.array, batch_np)
        return batch_jax