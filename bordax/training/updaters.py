from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import optax
from bordax.types import TrainingState, PRNGKey, Params
from bordax.algorithms.losses import DQNLoss
from bordax.agents.base import DQNParameters

from typing import Tuple, Any, Callable, Mapping
import functools


class Updater(ABC):

    @abstractmethod
    def init(self, params: Params) -> TrainingState: ...

    @abstractmethod
    def __call__(self, agent, buffer, ts: TrainingState, key: PRNGKey) -> Tuple[TrainingState, Any]: ...

class SGDUpdate(Updater):
    def __init__(self, optimizer, loss_fn, num_sgd_steps: int = 1, grad_clip: float | None = None):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_sgd_steps = num_sgd_steps
        self.grad_clip = grad_clip

    def init(self, params):
        return TrainingState(optimizer_state=self.optimizer.init(params), params=params, step=jnp.array(0))

    def __call__(self, agent, buffer, ts, key):

        vg_loss = jax.value_and_grad(self.loss_fn, has_aux=True)
        minibatches = buffer
                    
        def sgd_step(carry, unused):
            optimizer_state, params, step_key = carry 
            
            def minibatch_step(carry, mini):
                optimizer_state, params, key = carry
                key, mini_loss_key = jax.random.split(key)
                (loss, metrics), grads = vg_loss(params, agent, mini, mini_loss_key, ts.step)
                grad_norm = optax.global_norm(grads) # this is pre-clipping norm

                if self.grad_clip is not None:
                    # estimate the clipping
                    clipped_grad_norm = jnp.minimum(grad_norm, self.grad_clip)
                else:
                    clipped_grad_norm = grad_norm

                params_update, new_optimizer_state = self.optimizer.update(grads, optimizer_state)
                new_params = optax.apply_updates(params, params_update)
                update_norm = optax.global_norm(params_update) # norm of the update applied (takes learning rate into account)

                metrics = dict(metrics)
                metrics["grad_norm"] = grad_norm
                metrics["clipped_grad_norm"] = clipped_grad_norm
                metrics["update_norm"] = update_norm
                if isinstance(new_optimizer_state, tuple):
                    metrics["lr"] = new_optimizer_state[1].hyperparams['learning_rate']
                    metrics["counter"] = new_optimizer_state[1].count
                else:
                    metrics["lr"] = new_optimizer_state.hyperparams['learning_rate']
                    metrics["counter"] = new_optimizer_state.count

                return (new_optimizer_state, new_params, key), metrics

            (new_optimizer_state, new_params, key), metrics = jax.lax.scan(
                minibatch_step, (optimizer_state, params, step_key), minibatches)

            metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)

            return (new_optimizer_state, new_params, key), metrics
        
        (new_optimizer_state, new_params, _), metrics = jax.lax.scan(
            sgd_step, (ts.optimizer_state, ts.params, key), length=self.num_sgd_steps
        )

        metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), metrics)

        return TrainingState(new_optimizer_state, new_params, ts.step + 1), metrics


class DQNUpdater(Updater):
    def __init__(self, optimizer: optax.GradientTransformation, loss_fn: DQNLoss, target_update_freq: int):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.target_update_freq = target_update_freq

    def init(self, params: DQNParameters) -> TrainingState:
        # params here will be DQNParameters(q_network=q_params, target_network=q_params)
        # We only optimize q_network, so initialize optimizer with q_network params
        return TrainingState(
            optimizer_state=self.optimizer.init(params.q_network),
            params=params, # Store the full DQNParameters
            step=jnp.array(0)
        )

    @functools.partial(jax.jit, static_argnames=("self", "agent"))
    def __call__(self, agent, batch: Mapping[str, jnp.ndarray], ts: TrainingState, key: PRNGKey) -> Tuple[TrainingState, Any]:
        # Calculate loss and gradients with respect to the full params
        def loss_wrapper(full_params):
            return self.loss_fn(full_params, agent, batch, key, ts.step)
        
        vg_loss = jax.value_and_grad(loss_wrapper, has_aux=True)
        (loss, metrics), grads = vg_loss(ts.params)

        # Apply gradients only to the online Q-network parameters
        updates, new_optimizer_state = self.optimizer.update(grads.q_network, ts.optimizer_state, ts.params.q_network)
        new_q_network_params = optax.apply_updates(ts.params.q_network, updates)

        # Update the full DQNParameters with new q_network params
        new_params = ts.params._replace(q_network=new_q_network_params)

        # Periodically update target network
        def update_target_fn(current_params):
            return current_params._replace(target_network=current_params.q_network)

        def no_update_target_fn(current_params):
            return current_params

        new_params = jax.lax.cond(
            ts.step % self.target_update_freq == 0,
            update_target_fn,
            no_update_target_fn,
            new_params
        )

        return TrainingState(
            optimizer_state=new_optimizer_state,
            params=new_params,
            step=ts.step + 1
        ), metrics
