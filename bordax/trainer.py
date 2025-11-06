from dataclasses import dataclass
from gymnax import EnvParams
from bordax.agents.base import Agent
from bordax.environments.utils import EnvAdapter
from bordax.algorithms.base import Algorithm
from bordax.types import PRNGKey

from typing import Any, Mapping, Tuple, Optional
import functools

from tqdm import tqdm

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Any

@dataclass
class TrainerConfig:
    num_checkpoints: int
    epochs_per_checkpoint: int
    evaluation_episodes: int
    debug: bool
    save_model: bool
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

    def init(self, key: PRNGKey):
        key, env_key, init_key = jax.random.split(key, 3)
        self.last_obs, self.last_env_state = self.env.reset(env_key)
        self.training_state = self.algo.init_training_state(
            self.agent, init_key, self.last_obs, self.env
        )

        # Evaluation environment must be single-environment (num_envs=1)
        assert self.eval_env.num_envs == 1, f"eval_env must have num_envs=1, got {self.eval_env.num_envs}"
        
        # Initialize replay buffer for off-policy algorithms
        if self.config.replay_buffer_capacity is not None:
            from bordax.buffer import ReplayBuffer
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

    @functools.partial(jax.jit, static_argnames=("self"))
    def evaluate_jittable(self, keys, params):
        max_steps = self.eval_env.env_params.max_steps_in_episode

        def evaluate_one_episode(episode_key):
            run_key, reset_key = jax.random.split(episode_key)
            obs, env_state = self.eval_env.reset(reset_key)

            def step(
                carry: Tuple[PRNGKey, jnp.ndarray, Any, jnp.ndarray], _: Any
            ) -> Tuple[
                Tuple[PRNGKey, jnp.ndarray, Any, jnp.ndarray],
                Tuple[jnp.ndarray, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any],
            ]:
                step_key, obs, state, done = carry
                step_key, action_key, env_key = jax.random.split(step_key, 3)

                action, _ = self.agent.action(
                    params, obs, action_key, is_deterministic=True
                )
                n_obs, n_state, reward, done, info = self.eval_env.step(
                    env_key, state, action
                )

                # Ensure reward and done are jnp.ndarray
                reward = jnp.asarray(reward)
                done = jnp.asarray(done)

                new_carry = (step_key, n_obs, n_state, done)
                output = (obs, state, action, reward, done, info)

                return new_carry, output

            (final_carry, traj) = jax.lax.scan(
                f=step,
                init=(run_key, obs, env_state, jnp.array([False])),
                xs=None,
                length=max_steps,
            )

            obs_seq, state_seq, action_seq, reward_seq, done_seq, info_seq = traj
            obs_seq = jnp.squeeze(obs_seq, axis=1)
            state_seq = jax.tree_map(lambda s: jnp.squeeze(s, axis=1), state_seq)
            action_seq = jnp.squeeze(action_seq, axis=1)
            reward_seq = jnp.squeeze(reward_seq, axis=1)
            done_seq = jnp.squeeze(done_seq, axis=1)
            info_seq = jax.tree_map(lambda s: jnp.squeeze(s, axis=1), info_seq)

            return reward_seq, done_seq

        reward_seq, done_seq = jax.vmap(evaluate_one_episode)(keys)

        def summarize_episode(rewards, dones):
            num_steps = rewards.shape[0]
            indices = jnp.arange(num_steps)
            done_indices = jnp.asarray(jnp.where(dones, indices, num_steps), dtype=jnp.int32)
            first_done = jnp.min(done_indices)
            length = jnp.minimum(first_done + 1, num_steps)
            mask = indices < length
            total_reward = jnp.sum(jnp.where(mask, rewards, 0.0))
            return total_reward, length

        returns, lengths = jax.vmap(summarize_episode)(reward_seq, done_seq)
        return {
            "return": returns,
            "length": lengths,
        }

    def evaluate_non_jittable(self, keys, params):
        num_steps = self.eval_env.env_params.max_steps_in_episode
        num_envs = len(keys)
        episode_returns = np.zeros(num_envs, dtype=np.float32)
        episode_lengths = np.zeros(num_envs, dtype=np.int32)

        for episode, key in enumerate(keys):
            run_key, reset_key = jax.random.split(key, 2)

            obs, env_state = self.eval_env.reset(reset_key)

            total_reward = 0.0
            steps = 0

            for step in range(num_steps):
                action, _ = self.agent.action(
                    params, obs, run_key, is_deterministic=True
                )
                n_obs, n_state, reward, done, info = self.eval_env.step(
                    run_key, env_state, np.asarray(action)
                )

                reward = float(np.asarray(reward))
                done = bool(np.asarray(done))

                total_reward += reward
                steps += 1

                obs = n_obs
                env_state = n_state

                if done:
                    break

            episode_returns[episode] = total_reward
            episode_lengths[episode] = steps

        return {
            "return": episode_returns,
            "length": episode_lengths,
        }

    def evaluate(self, key: PRNGKey, params):
        evaluation_keys = jax.random.split(key, self.config.evaluation_episodes)
        if self.env.is_jittable:
            data = self.evaluate_jittable(evaluation_keys, params)
        else:
            data = self.evaluate_non_jittable(evaluation_keys, params)
        return data

    def run(self, key: PRNGKey):
        if self.config.debug:
            pbar = tqdm(total=self.config.num_checkpoints)
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
        all_metrics = []
        model_parameters = []

        for ckpt in range(self.config.num_checkpoints):
            metrics_accum = None
            metric_updates = 0
            # On-policy with jittable environment
            if self.env.is_jittable and self.replay_buffer is None:
                assert train_step is not None
                for epoch in range(self.config.epochs_per_checkpoint):
                    (
                        training_key,
                        self.training_state,
                        _,
                        self.last_obs,
                        self.last_env_state,
                    ), metrics = train_step(
                        training_key,
                        self.training_state,
                        None,
                        self.last_obs,
                        self.last_env_state,
                    )
                    if metrics is not None:
                        metrics_accum = (
                            metrics
                            if metrics_accum is None
                            else jax.tree_util.tree_map(lambda a, b: a + b, metrics_accum, metrics)
                        )
                        metric_updates += 1
            # Off-policy or non-jittable environment
            else:
                for epoch in range(self.config.epochs_per_checkpoint):
                    (
                        training_key,
                        self.training_state,
                        self.replay_buffer,
                        self.last_obs,
                        self.last_env_state,
                    ), metrics = self.algo.train_step(
                        self.env,
                        self.agent,
                        training_key,
                        self.training_state,
                        self.replay_buffer,
                        self.last_obs,
                        self.last_env_state,
                    )
                    if metrics is not None:
                        metrics_accum = (
                            metrics
                            if metrics_accum is None
                            else jax.tree_util.tree_map(lambda a, b: a + b, metrics_accum, metrics)
                        )
                        metric_updates += 1

            if self.config.enable_evaluation:
                eval_result = self.evaluate(evaluate_key, self.training_state.params)
                eval_result = jax.tree_util.tree_map(np.asarray, eval_result)
                epoch_rollouts.append(
                    {
                        "return": eval_result["return"].astype(np.float32).tolist(),
                        "length": eval_result["length"].astype(np.int32).tolist(),
                    }
                )
            
            if metrics_accum is not None:
                averaged_metrics = jax.tree_util.tree_map(
                    lambda x: x / metric_updates, metrics_accum
                )
                averaged_metrics = jax.device_get(averaged_metrics)
                all_metrics.append({k: float(v) for k, v in averaged_metrics.items()})

            model_parameters.append(self.training_state.params)

            if pbar is not None:
                pbar.update(1)

        return all_metrics, epoch_rollouts, model_parameters
