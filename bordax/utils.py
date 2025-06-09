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


# A trainer takes an environment, an agent architecture, and an algorithm (and a config)
class Trainer:
    def __init__(
        self,
        env: EnvAdapter,
        eval_env: EnvAdapter,
        agent: Agent,
        algo: Algorithm,
        config: Mapping[str, Any],
    ):
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        self.algo = algo
        self.config = config

    def init(self, key: PRNGKey):
        key, env_key, init_key = jax.random.split(key, 3)
        self.last_obs, self.last_env_state = self.env.reset(env_key)
        self.training_state = self.algo.init_training_state(
            self.agent, init_key, self.last_obs, self.env
        )

        assert self.eval_env.num_envs == 1

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
            print(obs_seq.shape)
            obs_seq = jnp.squeeze(obs_seq, axis=1)
            state_seq = jax.tree_map(lambda s: jnp.squeeze(s, axis=1), state_seq)
            action_seq = jnp.squeeze(action_seq, axis=1)
            reward_seq = jnp.squeeze(reward_seq, axis=1)
            done_seq = jnp.squeeze(done_seq, axis=1)
            info_seq = jax.tree_map(lambda s: jnp.squeeze(s, axis=1), info_seq)

            return {
                "obs": obs_seq,
                "state": state_seq,
                "action": action_seq,
                "reward": reward_seq,
                "done": done_seq,
                "info": info_seq,
            }

        data = jax.vmap(evaluate_one_episode)(keys)
        return data

    def evaluate_non_jittable(self, keys, params):
        num_steps = self.eval_env.env_params.max_steps_in_episode
        num_envs = len(keys)
        env_spec = dict(
            obs_shape=self.env.obs_space().shape, action_shape=self.env.action_space().shape
        )
        buffer = {
            "obs": np.zeros((num_envs, num_steps) + env_spec["obs_shape"]),
            "state": np.zeros((num_envs, num_steps) + env_spec["obs_shape"]),
            "action": np.zeros((num_envs, num_steps) + env_spec["action_shape"]),
            "reward": np.zeros((num_envs, num_steps)),
            "done": np.zeros((num_envs, num_steps), dtype=np.bool),
            "info": {"logp": np.zeros((num_envs, num_steps))},
        }

        for episode, key in enumerate(keys):
            run_key, reset_key = jax.random.split(key, 2)

            obs, env_state = self.eval_env.reset(reset_key)

            for step in range(num_steps):
                action, _ = self.agent.action(
                    params, obs, run_key, is_deterministic=True
                )
                n_obs, n_state, reward, done, info = self.eval_env.step(
                    run_key, env_state, np.asarray(action)
                )

                # Ensure reward and done are numpy arrays
                reward = np.asarray(reward)
                done = np.asarray(done)

                buffer["obs"][episode, step] = obs
                buffer["state"][episode, step] = env_state
                buffer["action"][episode, step] = action[0]
                buffer["reward"][episode, step] = reward
                buffer["done"][episode, step] = done
                for key in info:
                    buffer["info"][key][episode, step] = info[key]

                obs = n_obs
                env_state = n_state

                if done:
                    break

        return buffer

    def evaluate(self, key: PRNGKey, params):
        evaluation_keys = jax.random.split(key, self.config["evaluation_episodes"])
        if self.env.is_jittable:
            data = self.evaluate_jittable(evaluation_keys, params)
        else:
            data = self.evaluate_non_jittable(evaluation_keys, params)
        return data

    def run(self, key: PRNGKey):
        if self.config["debug"]:
            pbar = tqdm(total=self.config["num_checkpoints"])
        else:
            pbar = None

        print(
            "Total number of timesteps: ",
            self.config["num_checkpoints"]
            * self.config["epochs_per_checkpoint"]
            * self.algo.collector.rollout_len,
        )

        key, training_key, evaluate_key = jax.random.split(key, 3)

        if self.env.is_jittable:
            train_step_fixed = functools.partial(
                self.algo.train_step, self.env, self.agent
            )
            train_step = jax.jit(train_step_fixed)

        epoch_rollouts = []
        all_metrics = []
        model_parameters = []

        for ckpt in range(self.config["num_checkpoints"]):
            if self.env.is_jittable:
                for epoch in range(self.config["epochs_per_checkpoint"]):
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
            else:
                for epoch in range(self.config["epochs_per_checkpoint"]):
                    (
                        trainng_key,
                        self.training_state,
                        _,
                        self.last_obs,
                        self.last_env_state,
                    ), metrics = self.algo.train_step(
                        self.env,
                        self.agent,
                        training_key,
                        self.training_state,
                        None,
                        self.last_obs,
                        self.last_env_state,
                    )
                    all_metrics.append(metrics)

            epoch_rollouts.append(
                self.evaluate(evaluate_key, self.training_state.params)
            )

            model_parameters.append(self.training_state.params)


            if pbar is not None:
                pbar.update(1)

        return all_metrics, epoch_rollouts, model_parameters
