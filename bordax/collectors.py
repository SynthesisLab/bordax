from bordax.agents.base import Agent, BlankAgent, MixtureAgent
from bordax.environments.utils import EnvAdapter, EnvState, EnvObs
from bordax.types import PRNGKey, Params, TrainingState

from typing import Any, Callable, Tuple
import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
import functools
import numpy as np


class Collector(ABC):

    @abstractmethod
    def __call__(
        self,
        key: PRNGKey,
        env: EnvAdapter,
        obs: EnvObs,
        env_state: EnvState,
        replay_buffer: Any,
        agent: Agent,
        ts: TrainingState,
    ) -> Tuple[Tuple[Any, EnvState], Any]: ...


class OnPolicyCollector(Collector):
    def __init__(
        self, rollout_length: int = 1024, gamma: float = 0.99, _lambda: float = 0.99
    ):
        self.rollout_length = rollout_length
        self.gamma = gamma
        self._lambda = _lambda

    @functools.partial(jax.jit, static_argnames=("self", "agent", "env"))
    def collect_jittable(self, key, env, obs, env_state, agent: Agent, params):
        init_obs, init_state = obs, env_state

        def one_step(carry, unused):
            key, obs, env_state = carry
            key, act_key, env_key = jax.random.split(key, 3)
            action, info = agent.action(params, obs, act_key)
            value = agent.value(params, obs)

            n_obs, n_env_state, reward, done, env_info = env.step(
                env_key, env_state, action
            )

            transition = dict(
                obs=obs,
                action=action,
                value=value,
                reward=reward,
                done=done,
                info=info,
            )

            return (key, n_obs, n_env_state), transition

        (key, last_obs, last_env_state), traj = jax.lax.scan(
            one_step,
            (key, init_obs, init_state),
            None,
            length=self.rollout_length,
        )

        return (key, last_obs, last_env_state), traj

    def collect_non_jittable(
        self, key, env: EnvAdapter, obs, env_state, agent: Agent, params
    ):
        # Since non-jittable environments are executed on CPU,
        # it makes sense to save data in numpy arrays directly
        # and eventually convert to JAX arrays and send them to device

        # It is, however, possibly a bottleneck as the data transfer for each step
        # between the device and the host may be slow

        env_spec = dict(
            obs_shape=env.obs_space().shape, action_shape=env.action_space().shape
        )
        
        buffer = {
            "obs": np.zeros(
                (self.rollout_length, env.num_envs) + env_spec["obs_shape"],
                dtype=np.float32,
            ),
            "action": np.zeros(
                (self.rollout_length, env.num_envs) + env_spec["action_shape"],
                dtype=np.int32,
            ),
            "value": np.zeros((self.rollout_length, env.num_envs), dtype=np.float32),
            "reward": np.zeros((self.rollout_length, env.num_envs), dtype=np.float32),
            "done": np.zeros((self.rollout_length, env.num_envs), dtype=np.bool_),
            "info": {
                "logp": np.zeros((self.rollout_length, env.num_envs), dtype=np.float32)
            },
        }

        for i in range(self.rollout_length):
            key, act_key, env_key = jax.random.split(key, 3)
            
            buffer["obs"][i] = obs
            
            
            action, action_info = agent.action(params, obs, act_key) # Agent methods return jax arrays
            value = agent.value(params, obs)

            n_obs, n_env_state, reward, done, env_info = env.step(
                act_key, env_state, np.asarray(action)
            )
            
            buffer["action"][i] = np.asarray(action)
            buffer["value"][i] = np.asarray(value)
            buffer["reward"][i] = np.asarray(reward)
            buffer["done"][i] = np.asarray(done)
            buffer["info"]["logp"][i] = np.asarray(action_info["logp"])
            
            obs = n_obs
            env_state = n_env_state

        # Convert everything to jax arrays
        traj = jax.tree_util.tree_map(jnp.asarray, buffer)

        return (obs, env_state), traj

    def __call__(self, key, env, obs, env_state, replay_buffer: Any, agent: Agent, ts: TrainingState):
        # Collect rollout (returns JAX arrays in both cases)
        if env.is_jittable:
            (key, last_obs, last_env_state), traj = self.collect_jittable(
                key, env, obs, env_state, agent, ts.params
            )
        else:
            (last_obs, last_env_state), traj = self.collect_non_jittable(
                key, env, obs, env_state, agent, ts.params
            )
        
        # Calculate GAE (traj is JAX arrays here)
        last_value = agent.value(ts.params, last_obs)
        values = agent.value(ts.params, traj["obs"])

        advantages, targets = jax.lax.stop_gradient(
            compute_gae(traj, last_value, values, self.gamma, self._lambda)
        )

        traj["advantages"] = advantages
        traj["targets"] = targets

        return (last_obs, last_env_state), traj


@jax.jit
def compute_gae(traj_batch, last_value, values, gamma, gae_lambda):

    def _get_advantages(gae_and_next_value, transition):

        gae, next_value = gae_and_next_value
        transition, value = transition
        done, reward = (
            transition["done"],
            transition["reward"],
        )

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae

        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_value), last_value),
        (traj_batch, values),
        reverse=True,
    )

    return advantages, advantages + values


class EpsGreedyCollector(Collector):
    def __init__(self, epsilon_schedule: Callable[[int], float], rollout_length: int = 1):
        self.epsilon_schedule = epsilon_schedule
        self.rollout_length = rollout_length

    @functools.partial(jax.jit, static_argnames=("self", "agent", "env"))
    def _jit_collect(self, key: PRNGKey, env: EnvAdapter, obs: EnvObs, 
                            env_state: EnvState, agent: Agent, params: Params, epsilon: float):
        def one_step(carry, unused):
            key, obs, env_state = carry
            key, explore_key, act_key, env_key = jax.random.split(key, 4)
            do_explore = jax.random.uniform(explore_key) < epsilon
            action, _ = agent.action(params, obs, act_key)
            if hasattr(env.action_space(), 'n'):
                random_action = jax.random.randint(act_key, action.shape, 0, env.action_space().n)
            else:
                random_action = jax.random.uniform(act_key, action.shape, 
                                                   minval=env.action_space().low,
                                                   maxval=env.action_space().high)
            
            action = jax.lax.select(do_explore, random_action, action)
            n_obs, n_env_state, reward, done, _ = env.step(env_key, env_state, action)
            
            transition = {
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': n_obs,
                'done': done
            }
            
            return (key, n_obs, n_env_state), transition
        
        (key, final_obs, final_state), transitions = jax.lax.scan(
            one_step,
            (key, obs, env_state),
            None,
            length=self.rollout_length
        )
        
        return (final_obs, final_state), transitions
    
    def _non_jittable_collect(self, key: PRNGKey, env: EnvAdapter, obs: EnvObs, 
                              env_state: EnvState, agent: Agent, params: Params):
        
        raise NotImplementedError("Non-jittable environments are not supported yet.")

    def __call__(self, key: PRNGKey, env: EnvAdapter, obs: EnvObs, env_state: EnvState, 
                 replay_buffer: Any, agent: Agent, ts: TrainingState) -> Tuple[Tuple[Any, EnvState], Any]:
        
        epsilon = self.epsilon_schedule(ts.step.item())
    
        if env.is_jittable:
            (obs, env_state), transitions = self._jit_collect(key, env, obs, env_state, agent, ts.params, epsilon)
        else:
            raise NotImplementedError("Non-jittable environments are not supported yet.")
        
        transitions_np = jax.tree_util.tree_map(np.asarray, transitions)

        for i in range(self.rollout_length):
            transition = jax.tree_util.tree_map(lambda x: x[i:i+1], transitions_np)
            replay_buffer.add(transition)

        return (obs, env_state), replay_buffer


class StochasticOffPolicyCollector(Collector):
    pass


class DeterministicOffPolicyCollector(Collector):
    pass
