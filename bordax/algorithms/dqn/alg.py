import flax.struct
import jax
from jax import numpy as jnp
import optax
import flashbax as fbx
from typing import Any, NamedTuple, Dict, Callable
from functools import partial
import time
from tqdm import trange

from gymnax.environments.environment import Environment as EnvGymnax
from gymnasium import Env as EnvGymnasium
from bordax.environments.pomdp.pomdp import BeliefWrapper
import gymnasium

from bordax.environments.utils import generate_unroll

PolicyParams = Any

import chex


@flax.struct.dataclass
class TimeStep:
    obs: Any
    action: Any
    reward: Any
    done: Any


@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: PolicyParams
    target_params: PolicyParams
    env_params: Any
    buffer_state: Any
    last_obs: Any
    last_state: Any
    timesteps: Any
    epsilon: Any


class DQNConfig(NamedTuple):
    seed: int = 0
    learning_rate: float = 1e-4
    num_checkpoints: int = 100
    epochs_per_checkpoint: int = 1
    steps_per_epoch: int = 1000
    batch_size: int = 128
    buffer_size: int = 10000
    target_update_freq: int = 500
    gamma: float = 0.99
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.98
    num_envs: int = 1
    env_jitable: bool = True


def loss_fn(params, target_params, policy, batch, gamma):
    q_values = policy.apply(params, batch.first.obs)
    q_values = q_values[jnp.arange(q_values.shape[0]), batch.first.action]

    next_q_values = policy.apply(target_params, batch.second.obs)
    target_q = batch.first.reward + gamma * jnp.max(next_q_values, axis=-1) * (
        1.0 - batch.first.done
    )

    loss = jnp.mean((q_values - target_q) ** 2)
    return loss

    # next_q_values = policy.apply(
    #     train_state.target_network_params, learn_batch.second.obs
    # )  # (batch_size, num_actions)
    # q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
    # target = (
    #     learn_batch.first.reward
    #     + (1 - learn_batch.first.done) * config["GAMMA"] * q_next_target
    # )


def gradient_update_fn(optimizer: optax.GradientTransformation, gamma: float, policy):
    def update(params, target_params, optimizer_state, batch):
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(params, target_params, policy, batch, gamma)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, updates)
        return params, optimizer_state, loss

    return jax.jit(update)


def training_step_fn(
    env, env_params, make_policy, update_fn, buffer, config: DQNConfig
):

    def training_step(carry, _):
        training_state, key = carry
        key_step, key_action, key_sample = jax.random.split(key, 3)

        policy = make_policy(training_state.params, deterministic=False)

        action = policy(training_state.last_obs, key_action, training_state.epsilon)

        obs, env_state, reward, done, info = env.step(
            key_step, training_state.last_state, action
        )
        timestep = TimeStep(
            obs=training_state.last_obs,
            action=action,
            reward=reward,
            done=done,
        )

        buffer_state = buffer.add(training_state.buffer_state, timestep)

        batch = buffer.sample(buffer_state, key_sample).experience

        new_params, new_opt_state, loss = update_fn(
            training_state.params,
            training_state.target_params,
            training_state.optimizer_state,
            batch,
        )

        # Periodic target network update
        is_learn_time = training_state.timesteps % config.target_update_freq == 0

        target_params = jax.lax.cond(
            is_learn_time, lambda: new_params, lambda: training_state.target_params
        )

        new_state = TrainingState(
            optimizer_state=new_opt_state,
            params=new_params,
            target_params=target_params,
            env_params=training_state.env_params,
            buffer_state=buffer_state,
            timesteps=training_state.timesteps + 1,
            last_obs=obs,
            last_state=env_state,
            epsilon=training_state.epsilon,
        )

        return (new_state, key_step), loss

    return jax.jit(training_step)


def training_epoch_fn(training_step, config: DQNConfig):
    def training_epoch(training_state, key):
        (training_state, _), losses = jax.lax.scan(
            training_step,
            (training_state, key),
            None,
            length=config.steps_per_epoch,
        )
        return training_state, jnp.mean(losses)

    return jax.jit(training_epoch)


def evaluate_fn(env, make_policy, n_envs, env_params):

    def evaluate_jittable(key, params):
        policy = make_policy(params, deterministic=True)

        def evaluate_one_episode(key):
            obs, env_state = env.reset(key, env_params)

            def step(carry):
                obs, state, total_reward, done = carry
                action = policy(obs, key, 0)

                if len(action.shape) > 0:
                    action = jnp.squeeze(action, axis=-1)
                n_obs, n_state, reward, done, _ = env.step(
                    key, state, action, env_params
                )
                return n_obs, n_state, total_reward + reward, done

            def cond(carry):
                obs, state, total_reward, done = carry
                return jnp.logical_not(done)

            _, _, total_reward, _ = jax.lax.while_loop(
                cond, step, (obs, env_state, 0.0, False)
            )

            return total_reward

        key_v = jax.random.split(key, n_envs)
        total_rewards = jax.vmap(evaluate_one_episode)(key_v)
        return total_rewards

    return jax.jit(evaluate_jittable)


def train(
    environment: EnvGymnax,
    env_params: Dict,
    policy_maker: Callable,
    make_inference_fn: Callable,
    config: DQNConfig = DQNConfig(),
):
    key = jax.random.key(config.seed)
    env = environment

    key_env, key_init, key_train, key_eval = jax.random.split(key, 4)
    reset_fn = env.reset
    obs, env_state = reset_fn(key_env, env_params)

    buffer = fbx.make_flat_buffer(
        config.buffer_size, config.target_update_freq, config.batch_size
    )

    buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

    rng = jax.random.PRNGKey(0)  # use a dummy rng here
    _action = env.action_space().sample(rng)
    _, _env_state = env.reset(rng, env_params)
    _obs, _, _reward, _done, _ = env.step(rng, _env_state, _action, env_params)
    _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)

    buffer_state = buffer.init(_timestep)

    actor = policy_maker(env, env_params=env_params)
    make_policy = make_inference_fn(actor)
    optimizer = optax.adam(config.learning_rate)
    update_fn = gradient_update_fn(optimizer, config.gamma, actor)
    training_step = training_step_fn(
        env, env_params, make_policy, update_fn, buffer, config
    )
    training_epoch = training_epoch_fn(training_step, config)
    evaluate = evaluate_fn(env, make_policy, 10, env_params)

    init_params = actor.init(key_init)
    epsilon = config.epsilon_start
    training_state = TrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        target_params=init_params,
        env_params=env_params,
        buffer_state=buffer_state,
        last_obs=obs,
        last_state=env_state,
        timesteps=0,
        epsilon=epsilon,
    )

    checkpoints = [evaluate(key_eval, training_state.params)]
    iterator = trange(config.num_checkpoints, desc="Checkpoints")

    for _ in iterator:
        for _ in range(config.epochs_per_checkpoint):

            training_state, _ = training_epoch(training_state, key_train)

        epsilon = max(config.epsilon_end, epsilon * config.epsilon_decay)
        training_state = training_state.replace(epsilon=epsilon)
        checkpoints.append(evaluate(key_eval, training_state.params))

    return training_state, jnp.array(checkpoints)
