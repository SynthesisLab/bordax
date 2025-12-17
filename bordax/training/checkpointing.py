
from dataclasses import dataclass
import orbax.checkpoint as ocp
import jax

@dataclass
class CheckpointerConfig:
    save_path: str
    interval: int

class Checkpointer:
    def __init__(self, config: CheckpointerConfig):

        self.ckpt_path = config.save_path
        options = ocp.CheckpointManagerOptions(max_to_keep=10, create=True, save_interval_steps=config.interval)
        self.manager = ocp.CheckpointManager(
            self.ckpt_path,
            options=options
        )

    def save(self, model_state, epoch):
        self.manager.save(epoch, args=ocp.args.StandardSave(model_state))
        self.manager.wait_until_finished()

    def load(self, model_state, epoch):
        abstract_train_state = jax.tree.map(ocp.utils.to_shape_dtype_struct, model_state)
        restored_state = self.manager.restore(epoch, args=ocp.args.StandardRestore(abstract_train_state))
        return restored_state
