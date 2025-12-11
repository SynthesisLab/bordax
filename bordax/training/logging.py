from dataclasses import dataclass, field
import os
from typing import Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

@dataclass
class WandbConfig:
    project_name: Optional[str] = None
    entity: Optional[str] = None
    run_name: Optional[str] = None

@dataclass
class LoggerConfig:
    log_dir: str
    use_wandb: bool = False
    wandb: WandbConfig = field(default_factory=WandbConfig)

class Logger():
    
    def __init__(self, config: LoggerConfig):
        self.config = config
        self.log_dir = config.log_dir

        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self._metrics_log_path = os.path.join(self.log_dir, "metrics.csv") if self.log_dir else None
        self._metrics_log_file_exists = os.path.exists(self._metrics_log_path) if self._metrics_log_path else False
        self._evaluation_log_path = os.path.join(self.log_dir, "evaluation.csv") if self.log_dir else None
        self._evaluation_log_file_exists = os.path.exists(self._evaluation_log_path) if self._evaluation_log_path else False

        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            self._wandb_run = wandb.init( # type: ignore
                project=config.wandb.project_name,
                entity=config.wandb.entity,
                name=config.wandb.run_name,
            )


    def _log_dict(self, entry: dict, step: int, csv_path: Optional[str], file_exists_attr: str):
        if csv_path:
            write_header = not getattr(self, file_exists_attr)
            with open(csv_path, "a") as f:
                if write_header:
                    f.write(",".join(entry.keys()) + "\n")
                    setattr(self, file_exists_attr, True)
                f.write(",".join(str(v) for v in entry.values()) + "\n")
        if self.use_wandb:
            wandb.log(entry, step=step)  # type: ignore

    def log_metrics(self, metrics: dict, step: int):
        if metrics is None:
            return
        entry = {"step": step}
        entry.update(metrics)
        self._log_dict(entry, step, self._metrics_log_path, "_metrics_log_file_exists")

    def log_evaluation(self, evaluation: dict, step: int):
        if evaluation is None:
            return
        entry = {"step": step}
        entry.update(evaluation)
        self._log_dict(entry, step, self._evaluation_log_path, "_evaluation_log_file_exists")