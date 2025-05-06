from abc import ABC, abstractmethod
from bordax.types import Params, TrainState
import optax


class Optimizer(ABC):
    @abstractmethod
    def init(self, params: Params): ...

    @abstractmethod
    def step(self, train_state: TrainState, loss_grad: Params) -> TrainState: ...


class AdamUpdater(Optimizer):
    def __init__(self, lr: float = 3e-4):
        self.optim = optax.adam(lr)

    def init(self, params: Params):
        return TrainState(params, self.optim.init(params))

    def step(self, ts: TrainState, grads: Params):
        updates, new_opt_state = self.optim.update(grads, ts.opt_state)
        new_params = optax.apply_updates(ts.params, updates)
        return TrainState(new_params, new_opt_state)
