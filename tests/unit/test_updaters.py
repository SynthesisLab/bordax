"""Unit tests for updaters.

Tests parameter updates and training mechanics.
"""
import jax
import jax.numpy as jnp
import optax
import pytest
from bordax.training.updaters import SGDUpdate, DQNUpdater
from bordax.algorithms.losses import PPOLoss, DQNLoss
from bordax.types import TrainingState


@pytest.mark.unit
class TestSGDUpdate:
    """Tests for SGD updater."""

    def test_updates_parameters(self, mlp_agent, rng_key):
        """Parameters change after update."""
        # Match the actual PPO optimizer construction
        adam = optax.inject_hyperparams(optax.adam)(learning_rate=1e-3)
        optimizer = optax.chain(optax.clip_by_global_norm(10.0), adam)

        loss_fn = PPOLoss(
            clip_schedule=lambda s: 0.2,
            vf_coef_schedule=lambda s: 0.5,
            ent_coef_schedule=lambda s: 0.01,
        )
        updater = SGDUpdate(optimizer=optimizer, loss_fn=loss_fn, num_sgd_steps=1)

        # Initialize params
        sample_obs = jnp.zeros((4,))
        params = mlp_agent.init(rng_key, sample_obs)
        training_state = updater.init(params)

        # Create fake minibatches (3 minibatches of 8 samples each)
        batch_size = 8
        num_minibatches = 3
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (num_minibatches, batch_size, 4))

        dist, info = mlp_agent.policy(params, obs[0, 0], rng_key)
        action = dist.sample(seed=rng_key)
        logp = dist.log_prob(action)

        minibatches = {
            "obs": obs,
            "action": jnp.ones((num_minibatches, batch_size), dtype=jnp.int32) * action,
            "advantages": jax.random.normal(key2, (num_minibatches, batch_size)),
            "targets": jax.random.normal(key2, (num_minibatches, batch_size)),
            "info": {"logp": jnp.ones((num_minibatches, batch_size)) * logp},
        }

        # Perform update
        new_training_state, metrics = updater(mlp_agent, minibatches, training_state, rng_key)

        # Check parameters changed
        old_params_flat = jax.tree_util.tree_leaves(training_state.params)
        new_params_flat = jax.tree_util.tree_leaves(new_training_state.params)

        params_changed = False
        for old_p, new_p in zip(old_params_flat, new_params_flat):
            if not jnp.allclose(old_p, new_p, atol=1e-6):
                params_changed = True
                break

        assert params_changed, "Parameters should change after update"

        # Check step incremented
        assert new_training_state.step == training_state.step + 1, \
            f"Step should increment: {training_state.step} -> {new_training_state.step}"

        # Check metrics returned
        assert "grad_norm" in metrics
        assert "update_norm" in metrics

    def test_gradient_clipping(self, mlp_agent, rng_key):
        """grad_clip limits gradient norm."""
        # Match the actual PPO optimizer construction
        adam = optax.inject_hyperparams(optax.adam)(learning_rate=1e-3)
        optimizer = optax.chain(optax.clip_by_global_norm(10.0), adam)

        loss_fn = PPOLoss(
            clip_schedule=lambda s: 0.2,
            vf_coef_schedule=lambda s: 0.5,
            ent_coef_schedule=lambda s: 0.01,
        )

        # Create updater with gradient clipping
        updater_with_clip = SGDUpdate(
            optimizer=optimizer,
            loss_fn=loss_fn,
            num_sgd_steps=1,
            grad_clip=1.0,
        )

        # Initialize params
        sample_obs = jnp.zeros((4,))
        params = mlp_agent.init(rng_key, sample_obs)
        training_state = updater_with_clip.init(params)

        # Create fake minibatches
        batch_size = 8
        num_minibatches = 2
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (num_minibatches, batch_size, 4))

        dist, info = mlp_agent.policy(params, obs[0, 0], rng_key)
        action = dist.sample(seed=rng_key)
        logp = dist.log_prob(action)

        minibatches = {
            "obs": obs,
            "action": jnp.ones((num_minibatches, batch_size), dtype=jnp.int32) * action,
            "advantages": jax.random.normal(key2, (num_minibatches, batch_size)) * 100,  # Large gradients
            "targets": jax.random.normal(key2, (num_minibatches, batch_size)),
            "info": {"logp": jnp.ones((num_minibatches, batch_size)) * logp},
        }

        # Perform update
        _, metrics = updater_with_clip(mlp_agent, minibatches, training_state, rng_key)

        # Check that clipped_grad_norm exists and is computed
        assert "clipped_grad_norm" in metrics, "Should have clipped_grad_norm metric"
        assert jnp.isfinite(metrics["clipped_grad_norm"]), "Clipped grad norm should be finite"


@pytest.mark.unit
class TestDQNUpdater:
    """Tests for DQN updater."""

    def test_target_network_updates(self, dqn_agent, rng_key):
        """Target copies q_network at step % freq == 0."""
        optimizer = optax.adam(learning_rate=1e-3)
        loss_fn = DQNLoss(gamma=0.99)
        target_update_freq = 3

        updater = DQNUpdater(
            optimizer=optimizer,
            loss_fn=loss_fn,
            target_update_freq=target_update_freq,
        )

        # Initialize params
        sample_obs = jnp.zeros((4,))
        params = dqn_agent.init(rng_key, sample_obs)
        training_state = updater.init(params)  # step = 0

        # Create fake batch
        batch_size = 8
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (batch_size, 4))
        next_obs = jax.random.normal(key2, (batch_size, 4))

        batch = {
            "obs": obs,
            "action": jax.random.randint(rng_key, (batch_size,), 0, 2),
            "reward": jnp.ones(batch_size),
            "next_obs": next_obs,
            "done": jnp.zeros(batch_size),
        }

        # Do first update at step=0 (target gets updated)
        training_state, _ = updater(dqn_agent, batch, training_state, rng_key)

        # Save target params after first update
        target_after_step0 = jax.tree_util.tree_leaves(training_state.params.target_network)

        # Do 2 more updates (steps 1, 2 - no target updates)
        training_state, _ = updater(dqn_agent, batch, training_state, rng_key)
        training_state, _ = updater(dqn_agent, batch, training_state, rng_key)

        # Target should still match what it was after step 0
        target_after_step2 = jax.tree_util.tree_leaves(training_state.params.target_network)

        target_unchanged_during_interval = all(
            jnp.allclose(p1, p2) for p1, p2 in zip(target_after_step0, target_after_step2)
        )
        assert target_unchanged_during_interval, "Target should not change between update intervals"

        # Do one more update (step 3 - target gets updated again)
        training_state, _ = updater(dqn_agent, batch, training_state, rng_key)

        # Now target should match the current q_network
        q_params = jax.tree_util.tree_leaves(training_state.params.q_network)
        target_params = jax.tree_util.tree_leaves(training_state.params.target_network)

        target_matches_q = all(
            jnp.allclose(q_p, t_p) for q_p, t_p in zip(q_params, target_params)
        )
        assert target_matches_q, "Target should match q_network after periodic update"

    def test_only_q_network_trained(self, dqn_agent, rng_key):
        """Target network unchanged during update (except at update frequency)."""
        optimizer = optax.adam(learning_rate=1e-3)
        loss_fn = DQNLoss(gamma=0.99)
        target_update_freq = 100  # Large value to avoid target update

        updater = DQNUpdater(
            optimizer=optimizer,
            loss_fn=loss_fn,
            target_update_freq=target_update_freq,
        )

        # Initialize params
        sample_obs = jnp.zeros((4,))
        params = dqn_agent.init(rng_key, sample_obs)
        training_state = updater.init(params)

        # Perform first update (this happens at step 0, which updates target)
        # So we need to perform this update first, then check subsequent updates
        batch_size = 8
        key1, key2 = jax.random.split(rng_key)
        obs = jax.random.normal(key1, (batch_size, 4))
        next_obs = jax.random.normal(key2, (batch_size, 4))

        batch = {
            "obs": obs,
            "action": jax.random.randint(rng_key, (batch_size,), 0, 2),
            "reward": jnp.ones(batch_size),
            "next_obs": next_obs,
            "done": jnp.zeros(batch_size),
        }

        # Do first update (step 0 -> 1, target gets updated at step 0)
        training_state, _ = updater(dqn_agent, batch, training_state, rng_key)

        # Store target params after first update
        target_after_first = training_state.params.target_network

        # Perform second update (step 1 -> 2, target should NOT update)
        new_training_state, _ = updater(dqn_agent, batch, training_state, rng_key)

        # Target network should be unchanged from previous
        new_target_params = new_training_state.params.target_network
        target_after_first_flat = jax.tree_util.tree_leaves(target_after_first)
        new_target_flat = jax.tree_util.tree_leaves(new_target_params)

        target_unchanged = True
        for prev_p, new_p in zip(target_after_first_flat, new_target_flat):
            if not jnp.allclose(prev_p, new_p):
                target_unchanged = False
                break

        assert target_unchanged, \
            f"Target network should remain unchanged at step {new_training_state.step} (not divisible by {target_update_freq})"

        # Q-network should have changed
        initial_q_params = training_state.params.q_network
        new_q_params = new_training_state.params.q_network
        initial_q_flat = jax.tree_util.tree_leaves(initial_q_params)
        new_q_flat = jax.tree_util.tree_leaves(new_q_params)

        q_changed = False
        for init_p, new_p in zip(initial_q_flat, new_q_flat):
            if not jnp.allclose(init_p, new_p, atol=1e-6):
                q_changed = True
                break

        assert q_changed, "Q-network should change after training step"
