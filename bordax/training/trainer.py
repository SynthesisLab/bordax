from dataclasses import dataclass

from bordax.agents.base import Agent
from bordax.environments.utils import EnvAdapter
from bordax.algorithms.base import Algorithm
from bordax.training.evaluation import Evaluator
from bordax.training.logging import Logger, LoggerConfig
from bordax.training.checkpointing import Checkpointer
from bordax.types import PRNGKey

from typing import Any, Callable, Optional, Tuple
import functools

from tqdm import tqdm

import jax
import numpy as np

@dataclass
class TrainerConfig:
    num_checkpoints: int
    epochs_per_checkpoint: int
    evaluation_episodes: int
    logger_config: Optional[LoggerConfig] = None
    chekpointer_config: Optional[Any] = None
    restore_checkpoint: Optional[int] = None  # Epoch number to restore from
    debug: bool = False
    # Off-policy specific config
    replay_buffer_capacity: Optional[int] = None  # If None, on-policy algorithm
    warmup_steps: Optional[int] = None  # Steps to collect before training
    # Evaluation config
    enable_evaluation: bool = True  # If False, skip evaluation entirely

# A trainer takes an environment, an agent architecture, and an algorithm (and a config)
class Trainer:
    def __init__(
        self,
        env: EnvAdapter,
        eval_env: EnvAdapter,
        agent: Agent,
        algo: Algorithm,
        config: TrainerConfig,
    ):
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.algo = algo
        self.config = config
        self.replay_buffer = None  # For off-policy algorithms
        self.evaluator = Evaluator(eval_env, agent, config)
        if config.logger_config:
            self.logs_enabled = True
            self.logger_config = config.logger_config
            self.logger = Logger(self.logger_config)
        else:
            self.logs_enabled = False
        if config.chekpointer_config:
            self.checkpoints_enabled = True
            self.checkpointer_config = config.chekpointer_config
            self.checkpointer = Checkpointer(self.checkpointer_config)
        else:
            self.checkpoints_enabled = False

    def init(self, key: PRNGKey):
        key, env_key, init_key = jax.random.split(key, 3)
        self.last_obs, self.last_env_state = self.env.reset(env_key)
        self.training_state = self.algo.init_training_state(
            self.agent, init_key, self.last_obs, self.env
        )

        if self.config.restore_checkpoint:
            restored_state = self.checkpointer.load(self.training_state, self.config.restore_checkpoint)
            self.training_state = restored_state

        # Evaluation environment must be single-environment (num_envs=1)
        assert self.eval_env.num_envs == 1, f"eval_env must have num_envs=1, got {self.eval_env.num_envs}"
        
        # Initialize replay buffer for off-policy algorithms
        if self.config.replay_buffer_capacity is not None:
            from bordax.data.buffer import ReplayBuffer
            obs_shape = self.env.obs_space().shape
            action_shape = self.env.action_space().shape
            self.replay_buffer = ReplayBuffer(
                capacity=self.config.replay_buffer_capacity,
                obs_shape=obs_shape,
                action_shape=action_shape
            )
            
            # Warmup: fill buffer with initial transitions
            if self.config.warmup_steps is not None and self.config.warmup_steps > 0:
                print(f"Warming up replay buffer with {self.config.warmup_steps} transitions...")
                for i in range(self.config.warmup_steps):
                    key, collect_key = jax.random.split(key)
                    (self.last_obs, self.last_env_state), self.replay_buffer = self.algo.collect(
                        collect_key, self.env, self.last_obs, self.last_env_state, 
                        self.replay_buffer, self.agent, self.training_state
                    )
                    if (i + 1) % 200 == 0 and self.config.debug:
                        print(f"  Warmup: {i+1}/{self.config.warmup_steps}, Buffer size: {len(self.replay_buffer)}")
                print(f"Buffer filled with {len(self.replay_buffer)} transitions\n")

    def _run_epoch(
        self,
        key: PRNGKey,
        train_step_fn: Optional[Callable],
    ) -> Tuple[PRNGKey, Any]:
        """Run a single training epoch."""
        if train_step_fn is not None:
            # JIT-compiled path (on-policy, jittable env)
            (
                key,
                self.training_state,
                _,
                self.last_obs,
                self.last_env_state,
            ), metrics = train_step_fn(
                key,
                self.training_state,
                None,
                self.last_obs,
                self.last_env_state,
            )
        else:
            # Non-JIT path (off-policy or non-jittable env)
            (
                key,
                self.training_state,
                self.replay_buffer,
                self.last_obs,
                self.last_env_state,
            ), metrics = self.algo.train_step(
                self.env,
                self.agent,
                key,
                self.training_state,
                self.replay_buffer,
                self.last_obs,
                self.last_env_state,
            )
        return key, metrics

    def _run_checkpoint(self, training_key: PRNGKey, evaluate_key: PRNGKey, ckpt: int, train_step_fn, epoch_rollouts):

        # On-policy with jittable environment
        for epoch in range(self.config.epochs_per_checkpoint):
            training_key, metrics = self._run_epoch(training_key, train_step_fn)

            if self.logs_enabled:
                # Log training metrics
                self.logger.log_metrics(
                    {f"train/{k}": float(v) for k, v in metrics.items()},
                    step=ckpt * self.config.epochs_per_checkpoint + epoch,
                )
                

        if self.config.enable_evaluation:
            eval_result = self.evaluator.evaluate(evaluate_key, self.training_state.params)
            eval_result = jax.tree_util.tree_map(np.asarray, eval_result)

            eval_returns = eval_result["return"]
            eval_lengths = eval_result["length"]
            done_info = eval_result.get("done_info", None)
            avg_return = float(np.mean(eval_returns))
            avg_length = float(np.mean(eval_lengths))
            if done_info is not None: # average additional info if available
                avg_done_info = {k: float(np.mean([info[k] for info in done_info])) for k in done_info[0]}
            else:
                avg_done_info = {}

            if self.logs_enabled:
                entry = {
                    "eval/avg_return": avg_return,
                    "eval/avg_length": avg_length,
                }
                entry.update({f"eval/done_info/{k}": v for k, v in avg_done_info.items()})
                self.logger.log_evaluation(
                    entry,
                    step=ckpt,
                )
        

    def run(self, key: PRNGKey):
        if self.config.debug:
            pbar = tqdm(
                initial=0 + (0 if self.config.restore_checkpoint is None else self.config.restore_checkpoint),
                total=self.config.num_checkpoints + (0 if self.config.restore_checkpoint is None else self.config.restore_checkpoint))
        else:
            pbar = None

        # Calculate total timesteps based on whether we have a replay buffer
        rollout_len = getattr(self.algo.collector, 'rollout_length', 1)
            
        print(
            "Total number of timesteps: ",
            self.config.num_checkpoints
            * self.config.epochs_per_checkpoint
            * rollout_len,
        )

        key, training_key, evaluate_key = jax.random.split(key, 3)

        # For on-policy algorithms with jittable envs, we can JIT the entire train_step
        # For off-policy algorithms, train_step internally handles the non-jittable buffer
        train_step = None
        if self.env.is_jittable and self.replay_buffer is None:
            train_step_fixed = functools.partial(
                self.algo.train_step, self.env, self.agent
            )
            train_step = jax.jit(train_step_fixed)

        epoch_rollouts = []

        for ckpt in range(self.config.num_checkpoints):
            training_key, ckpt_training_key = jax.random.split(training_key)
            evaluate_key, ckpt_evaluate_key = jax.random.split(evaluate_key)
            current_epoch = ckpt + (self.config.restore_checkpoint or 0)
            self._run_checkpoint(ckpt_training_key, ckpt_evaluate_key, current_epoch, train_step, epoch_rollouts)
            if self.checkpoints_enabled:
                self.checkpointer.save(self.training_state, current_epoch+1)

            if pbar is not None:
                pbar.update(1)

        return epoch_rollouts
