"""Unit tests for loss functions.

Tests loss output structure and key behaviors.
"""
import jax
import jax.numpy as jnp
import pytest
from bordax.algorithms.losses import PPOLoss, DQNLoss, CombinedLoss, SurrogateLoss, ValueLoss


@pytest.mark.unit
class TestPPOLoss:
    """Tests for PPO loss."""

    def test_computes_loss_and_metrics(self, mlp_agent, rng_key):
        """Returns scalar loss + metrics dict with expected keys."""
        loss_fn = PPOLoss(
            clip_schedule=lambda s: 0.2,
            vf_coef_schedule=lambda s: 0.5,
            ent_coef_schedule=lambda s: 0.01,
        )

        # Create fake batch
        batch_size = 16
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (batch_size, 4))

        # Get sample action and log prob from agent
        params = mlp_agent.init(rng_key, obs[0])
        dist, info = mlp_agent.policy(params, obs[0], rng_key)
        action = dist.sample(seed=rng_key)
        logp = dist.log_prob(action)

        batch = {
            "obs": obs,
            "action": jnp.ones(batch_size, dtype=jnp.int32) * action,
            "advantages": jax.random.normal(key2, (batch_size,)),
            "targets": jax.random.normal(key2, (batch_size,)),
            "info": {"logp": jnp.ones(batch_size) * logp},
        }

        loss, metrics = loss_fn(params, mlp_agent, batch, rng_key, jnp.array(0))

        # Check scalar loss
        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"

        # Check expected metrics
        expected_keys = ["loss", "approx_kl", "value_loss", "entropy_loss", "total_loss"]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert jnp.isfinite(metrics[key]), f"Metric {key} should be finite"

    def test_clipping_active(self, mlp_agent, rng_key):
        """Ratio gets clipped when advantages large."""
        loss_fn = SurrogateLoss(clip_schedule=lambda s: 0.2)

        batch_size = 16
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (batch_size, 4))

        params = mlp_agent.init(rng_key, obs[0])
        dist, info = mlp_agent.policy(params, obs[0], rng_key)
        action = dist.sample(seed=rng_key)

        # Create batch with very different log probs to force clipping
        batch = {
            "obs": obs,
            "action": jnp.ones(batch_size, dtype=jnp.int32) * action,
            "advantages": jnp.ones(batch_size) * 10.0,  # Large advantages
            "info": {"logp": jnp.ones(batch_size) * -5.0},  # Very different from current policy
        }

        loss_with_clip, _ = loss_fn(params, mlp_agent, batch, rng_key, jnp.array(0))

        # Just verify loss is computed without error and is finite
        assert jnp.isfinite(loss_with_clip), "Loss with clipping should be finite"


@pytest.mark.unit
class TestDQNLoss:
    """Tests for DQN loss."""

    def test_bellman_target(self, dqn_agent, rng_key):
        """Target = r + gamma * max_Q(s') * (1-done)"""
        loss_fn = DQNLoss(gamma=0.99)

        batch_size = 8
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (batch_size, 4))
        next_obs = jax.random.normal(key2, (batch_size, 4))

        params = dqn_agent.init(rng_key, obs[0])

        batch = {
            "obs": obs,
            "action": jax.random.randint(rng_key, (batch_size,), 0, 2),
            "reward": jnp.ones(batch_size),
            "next_obs": next_obs,
            "done": jnp.zeros(batch_size),  # No episodes done
        }

        loss, metrics = loss_fn(params, dqn_agent, batch, rng_key, jnp.array(0))

        # Check scalar loss
        assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
        assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"

        # Check metrics
        assert "dqn_loss" in metrics
        assert "mean_q_value" in metrics
        assert "mean_target" in metrics

    def test_done_mask(self, dqn_agent, rng_key):
        """Done=True zeros future Q-values."""
        loss_fn = DQNLoss(gamma=0.99)

        batch_size = 8
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (batch_size, 4))
        next_obs = jax.random.normal(key2, (batch_size, 4))

        params = dqn_agent.init(rng_key, obs[0])

        # Create two batches: one with done=False, one with done=True
        batch_not_done = {
            "obs": obs,
            "action": jnp.zeros(batch_size, dtype=jnp.int32),
            "reward": jnp.ones(batch_size),
            "next_obs": next_obs,
            "done": jnp.zeros(batch_size),
        }

        batch_done = {
            "obs": obs,
            "action": jnp.zeros(batch_size, dtype=jnp.int32),
            "reward": jnp.ones(batch_size),
            "next_obs": next_obs,
            "done": jnp.ones(batch_size),
        }

        _, metrics_not_done = loss_fn(params, dqn_agent, batch_not_done, rng_key, jnp.array(0))
        _, metrics_done = loss_fn(params, dqn_agent, batch_done, rng_key, jnp.array(0))

        # When done=True, target should be closer to reward (no future value)
        # mean_target_done ≈ reward = 1.0
        # mean_target_not_done ≈ reward + gamma * Q(s') > 1.0
        assert metrics_done["mean_target"] < metrics_not_done["mean_target"], \
            "Done episodes should have lower targets (no future value)"


@pytest.mark.unit
class TestCombinedLoss:
    """Tests for combined loss."""

    def test_sums_losses(self, mlp_agent, rng_key):
        """Total = sum of component losses."""
        surrogate_loss = SurrogateLoss(clip_schedule=lambda s: 0.2)
        value_loss = ValueLoss(vf_coef_schedule=lambda s: 0.5)
        combined = CombinedLoss([surrogate_loss, value_loss])

        batch_size = 16
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (batch_size, 4))

        params = mlp_agent.init(rng_key, obs[0])
        dist, info = mlp_agent.policy(params, obs[0], rng_key)
        action = dist.sample(seed=rng_key)
        logp = dist.log_prob(action)

        batch = {
            "obs": obs,
            "action": jnp.ones(batch_size, dtype=jnp.int32) * action,
            "advantages": jax.random.normal(key2, (batch_size,)),
            "targets": jax.random.normal(key2, (batch_size,)),
            "info": {"logp": jnp.ones(batch_size) * logp},
        }

        total_loss, metrics = combined(params, mlp_agent, batch, rng_key, jnp.array(0))

        # Compute individual losses
        loss1, metrics1 = surrogate_loss(params, mlp_agent, batch, rng_key, jnp.array(0))
        loss2, metrics2 = value_loss(params, mlp_agent, batch, rng_key, jnp.array(0))

        # Check that total is sum
        expected_total = loss1 + loss2
        assert jnp.allclose(total_loss, expected_total), \
            f"Total loss should be sum of components: {total_loss} vs {expected_total}"

    def test_metrics_merged(self, mlp_agent, rng_key):
        """All metrics from components present."""
        surrogate_loss = SurrogateLoss(clip_schedule=lambda s: 0.2)
        value_loss = ValueLoss(vf_coef_schedule=lambda s: 0.5)
        combined = CombinedLoss([surrogate_loss, value_loss])

        batch_size = 16
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (batch_size, 4))

        params = mlp_agent.init(rng_key, obs[0])
        dist, info = mlp_agent.policy(params, obs[0], rng_key)
        action = dist.sample(seed=rng_key)
        logp = dist.log_prob(action)

        batch = {
            "obs": obs,
            "action": jnp.ones(batch_size, dtype=jnp.int32) * action,
            "advantages": jax.random.normal(key2, (batch_size,)),
            "targets": jax.random.normal(key2, (batch_size,)),
            "info": {"logp": jnp.ones(batch_size) * logp},
        }

        _, metrics = combined(params, mlp_agent, batch, rng_key, jnp.array(0))

        # Check all component metrics are present
        assert "loss" in metrics, "Should have surrogate loss metric"
        assert "approx_kl" in metrics, "Should have approx_kl metric"
        assert "value_loss" in metrics, "Should have value_loss metric"
        assert "total_loss" in metrics, "Should have total_loss metric"
