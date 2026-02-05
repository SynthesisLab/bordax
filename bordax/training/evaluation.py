

import functools

import jax
import jax.numpy as jnp
import numpy as np

from typing import Tuple, Any
from bordax.types import PRNGKey


class Evaluator:
    def __init__(self, eval_env, agent, config):
        self.eval_env = eval_env
        self.agent = agent
        self.config = config

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
            state_seq = jax.tree.map(lambda s: jnp.squeeze(s, axis=1), state_seq)
            action_seq = jnp.squeeze(action_seq, axis=1)
            reward_seq = jnp.squeeze(reward_seq, axis=1)
            done_seq = jnp.squeeze(done_seq, axis=1)
            info_seq = jax.tree.map(lambda s: jnp.squeeze(s, axis=1), info_seq)

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

                reward = float(np.asarray(reward).squeeze())
                done = bool(np.asarray(done).squeeze())

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
        if self.eval_env.is_jittable:
            data = self.evaluate_jittable(evaluation_keys, params)
        else:
            data = self.evaluate_non_jittable(evaluation_keys, params)
        return data