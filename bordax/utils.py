from bordax.agents.base import Agent
from bordax.environments.utils import EnvAdapter
from bordax.algorithms.base import Algorithm
from bordax.types import PRNGKey

from typing import Any, Mapping
import functools

from tqdm import tqdm

import jax
import jax.numpy as jnp

# A trainer takes an environment, an agent architecture, and an algorithm (and a config)
class Trainer:
    def __init__(
        self,
        env: EnvAdapter,
        eval_env: EnvAdapter,
        agent: Agent,
        algo: Algorithm,
        config: Mapping[str, Any]
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
    def evaluate(self, key: PRNGKey, params) -> jnp.ndarray:
        def evaluate_one_episode(episode_key):
            run_key, reset_key = jax.random.split(episode_key)
            obs, env_state = self.eval_env.reset(reset_key)

            def step(carry):
                step_key, obs, state, total_reward, done = carry
                step_key, action_key, env_key = jax.random.split(step_key, 3)
                action, _ = self.agent.action(
                    params, obs, action_key, is_deterministic=True
                )
                n_obs, n_state, reward, done, _ = self.eval_env.step(
                    env_key, state, action
                )
                return key, n_obs, n_state, total_reward + reward, done

            def cond(carry):
                _, _, _, _, done = carry
                return jnp.logical_not(jnp.any(done))

            _, _, _, total_reward, _ = jax.lax.while_loop(
                cond, step, (run_key, obs, env_state, jnp.array([0.0]), jnp.array([False]))
            )

            return total_reward

        keys = jax.random.split(key, self.config["evaluation_episodes"])

        rewards = jax.vmap(evaluate_one_episode)(keys)

        return rewards

    def run(self, key: PRNGKey):
        if self.config["debug"]:
            pbar = tqdm(total=self.config["num_checkpoints"])
        else:
            pbar = None

        print(
            "Total number of timesteps: ",
            self.config["num_checkpoints"]
            * self.config["epochs_per_checkpoint"]
            * 1024,
        )

        key, training_key, evaluate_key = jax.random.split(key, 3)

        if self.env.is_jittable:
            train_step_fixed = functools.partial(
                self.algo.train_step, self.env, self.agent
            )
            train_step = jax.jit(train_step_fixed)

        evaluations = []

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
                raise NotImplementedError

            rewards = self.evaluate(evaluate_key, self.training_state.params)
            rewards = jnp.squeeze(rewards, axis=-1)
            evaluations.append(rewards)

            if pbar is not None:
                pbar.update(1)

        return metrics, jnp.array(evaluations)
