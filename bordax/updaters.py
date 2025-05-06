from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import optax
from bordax.types import TrainingState


class Updater(ABC):

    @abstractmethod
    def init(self, params): ...

    @abstractmethod
    def __call__(self, agent, buffer, ts, key): ...

class SGDUpdate(Updater):
    def __init__(self, optimizer, loss_fn, num_sdg_steps: int = 1):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.num_sdg_steps = num_sdg_steps

    def init(self, params):
        return TrainingState(optimizer_state=self.optimizer.init(params), params=params, step=jnp.array(0))

    def __call__(self, agent, buffer, ts, key):

        vg_loss = jax.value_and_grad(self.loss_fn, has_aux=True)
        minibatches = buffer
                    
        def sdg_step(carry, unused):
            optimizer_state, params, step_key = carry 
            
            def minibatch_step(carry, mini):
                optimizer_state, params, key = carry
                key, mini_loss_key = jax.random.split(key)
                (loss, metrics), grads = vg_loss(params, agent, mini, mini_loss_key, ts.step)
                params_update, new_optimizer_state = self.optimizer.update(
                    grads, optimizer_state
                )
                new_params = optax.apply_updates(params, params_update)

                return (new_optimizer_state, new_params, key), metrics

            (new_optimizer_state, new_params, key), metrics = jax.lax.scan(
                minibatch_step, (optimizer_state, params, step_key), minibatches)

            return (new_optimizer_state, new_params, key), metrics
        
        (new_optimizer_state, new_params, _), metrics = jax.lax.scan(sdg_step, (ts.optimizer_state, ts.params, key), length=self.num_sdg_steps) 

        return TrainingState(new_optimizer_state, new_params, ts.step + 1), metrics
