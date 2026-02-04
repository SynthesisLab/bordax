"""Unit tests for batch builders.

Tests batch transformation operations and composition.
"""
import jax
import jax.numpy as jnp
import pytest
from bordax.data.batchbuilders import (
    FullBufferBatch,
    MiniBatch,
    NormalizeAdvantagesTargets,
    ComposedBatchBuilder,
    UniformReplayBatch,
)


@pytest.mark.unit
class TestFullBufferBatch:
    """Tests for FullBufferBatch."""

    def test_flatten_and_shuffle(self, rng_key):
        """(T, B, ...) -> shuffled (T*B, ...)"""
        builder = FullBufferBatch(buffer_size=8, num_env=4)

        # Create fake buffer: (time=8, batch=4, obs_dim=2)
        buffer = {
            "obs": jnp.ones((8, 4, 2)),
            "action": jnp.zeros((8, 4)),
        }

        batch = builder(rng_key, buffer)

        # Check flattening: (8*4, 2) = (32, 2)
        assert batch["obs"].shape == (32, 2), f"Expected (32, 2), got {batch['obs'].shape}"
        assert batch["action"].shape == (32,), f"Expected (32,), got {batch['action'].shape}"

        # Verify shuffling happened (probabilistic but very likely)
        # If we create ordered data and it stays ordered, shuffling didn't work
        ordered_buffer = {"obs": jnp.arange(32).reshape((8, 4, 1))}
        shuffled = builder(rng_key, ordered_buffer)
        is_shuffled = not jnp.array_equal(
            shuffled["obs"].flatten(), jnp.arange(32)
        )
        assert is_shuffled, "Data should be shuffled"


@pytest.mark.unit
class TestMiniBatch:
    """Tests for MiniBatch."""

    def test_reshape_to_minibatches(self, rng_key):
        """(N, ...) -> (num_mb, N/num_mb, ...)"""
        builder = MiniBatch(num_minibatches=4)

        # Create fake batch: (32, 2)
        buffer = {
            "obs": jnp.ones((32, 2)),
            "action": jnp.zeros((32,)),
        }

        minibatches = builder(rng_key, buffer)

        # Check reshaping: (4, 8, 2) for obs
        assert minibatches["obs"].shape == (4, 8, 2), \
            f"Expected (4, 8, 2), got {minibatches['obs'].shape}"
        assert minibatches["action"].shape == (4, 8), \
            f"Expected (4, 8), got {minibatches['action'].shape}"


@pytest.mark.unit
class TestNormalizeAdvantagesTargets:
    """Tests for advantage/target normalization."""

    def test_normalization(self, rng_key):
        """Advantages have ~zero mean, ~unit variance."""
        builder = NormalizeAdvantagesTargets(normalize_targets=False)

        # Create minibatch structure: (num_mb=2, mb_size=8)
        buffer = {
            "advantages": jax.random.normal(rng_key, (2, 8)) * 10 + 5,  # mean=5, std=10
            "targets": jnp.ones((2, 8)),
        }

        normalized = builder(rng_key, buffer)

        # Check per-minibatch normalization
        for i in range(2):
            adv_mean = jnp.mean(normalized["advantages"][i])
            adv_std = jnp.std(normalized["advantages"][i])

            assert jnp.abs(adv_mean) < 1e-6, f"Mean should be ~0, got {adv_mean}"
            assert jnp.abs(adv_std - 1.0) < 1e-6, f"Std should be ~1, got {adv_std}"

    def test_normalize_targets_flag(self, rng_key):
        """normalize_targets=True/False works."""
        key1, key2 = jax.random.split(rng_key)

        # Test with normalize_targets=False
        builder_no_norm = NormalizeAdvantagesTargets(normalize_targets=False)

        buffer = {
            "advantages": jnp.ones((2, 8)),
            "targets": jax.random.normal(key1, (2, 8)) * 10 + 5,  # Varied targets
        }
        original_targets = buffer["targets"].copy()

        result = builder_no_norm(rng_key, buffer)
        assert jnp.allclose(result["targets"], original_targets), \
            "Targets should not be normalized when flag is False"

        # Test with normalize_targets=True
        builder_with_norm = NormalizeAdvantagesTargets(normalize_targets=True)
        buffer2 = {
            "advantages": jnp.ones((2, 8)),
            "targets": jax.random.normal(key2, (2, 8)) * 10 + 5,  # Varied targets
        }
        result = builder_with_norm(rng_key, buffer2)

        # Check targets are normalized per minibatch
        for i in range(2):
            target_mean = jnp.mean(result["targets"][i])
            target_std = jnp.std(result["targets"][i])

            assert jnp.abs(target_mean) < 1e-5, f"Target mean should be ~0, got {target_mean}"
            assert jnp.abs(target_std - 1.0) < 1e-5, f"Target std should be ~1, got {target_std}"


@pytest.mark.unit
class TestComposedBatchBuilder:
    """Tests for batch builder composition."""

    def test_composition(self, rng_key):
        """Chain 3 builders, verify output correct."""
        # Pipeline: FullBufferBatch -> MiniBatch -> NormalizeAdvantagesTargets
        pipeline = ComposedBatchBuilder([
            FullBufferBatch(buffer_size=8, num_env=4),
            MiniBatch(num_minibatches=4),
            NormalizeAdvantagesTargets(normalize_targets=False),
        ])

        # Create buffer: (time=8, batch=4, obs_dim=2)
        buffer = {
            "obs": jnp.ones((8, 4, 2)),
            "advantages": jax.random.normal(rng_key, (8, 4)) * 10 + 5,
            "targets": jnp.ones((8, 4)),
        }

        result = pipeline(rng_key, buffer)

        # After FullBufferBatch: (32, ...) - flattened and shuffled
        # After MiniBatch: (4, 8, ...) - reshaped into 4 minibatches
        # After Normalize: advantages normalized per minibatch

        assert result["obs"].shape == (4, 8, 2), \
            f"Expected (4, 8, 2), got {result['obs'].shape}"

        # Check normalization happened
        for i in range(4):
            adv_mean = jnp.mean(result["advantages"][i])
            assert jnp.abs(adv_mean) < 1e-5, \
                f"Advantages should be normalized, got mean={adv_mean}"


@pytest.mark.unit
class TestUniformReplayBatch:
    """Tests for replay buffer sampling."""

    def test_samples_from_buffer(self, rng_key):
        """Returns batch_size JAX arrays from numpy buffer."""
        from bordax.data.buffer import ReplayBuffer
        import numpy as np

        # Create and populate a replay buffer
        buffer = ReplayBuffer(capacity=100, obs_shape=(4,), action_shape=())

        # Add some transitions manually
        transitions = {
            "obs": np.random.randn(20, 4).astype(np.float32),
            "action": np.random.randint(0, 2, size=20).astype(np.int32),
            "reward": np.random.randn(20).astype(np.float32),
            "next_obs": np.random.randn(20, 4).astype(np.float32),
            "done": np.random.randint(0, 2, size=20).astype(bool),
        }
        buffer.add(transitions)

        # Sample a batch
        builder = UniformReplayBatch(batch_size=8)
        batch = builder(rng_key, buffer)

        # Check output structure
        assert batch["obs"].shape == (8, 4), f"Expected (8, 4), got {batch['obs'].shape}"
        assert batch["action"].shape == (8,), f"Expected (8,), got {batch['action'].shape}"
        assert batch["reward"].shape == (8,), f"Expected (8,), got {batch['reward'].shape}"
        assert batch["next_obs"].shape == (8, 4), f"Expected (8, 4), got {batch['next_obs'].shape}"
        assert batch["done"].shape == (8,), f"Expected (8,), got {batch['done'].shape}"

        # Check data types are JAX arrays
        assert isinstance(batch["obs"], jnp.ndarray), "Should return JAX arrays"
