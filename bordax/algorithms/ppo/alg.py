import flax.struct
import flax.typing
import jax
from jax import numpy as jnp
import optax
import flax

from bordax.algorithms.ppo.losses import compute_loss_fn
from bordax.algorithms.utils import compute_gae
from bordax.policies.utils import ActorCriticMaker, ActorCriticParams
from bordax.environments.utils import generate_unroll

from typing import Any, Dict, NamedTuple, Callable
from functools import partial

from gymnax.environments.environment import Environment as EnvGymnax
from gymnasium import Env as EnvGymnasium
from bordax.environments.pomdp.pomdp import BeliefWrapper

import gymnasium

import time
from tqdm import trange

PolicyParams = Any


class PPOConfig(NamedTuple):
    learning_rate: float
    num_checkpoints: int
    epochs_per_checkpoint: int
    seed: int
    unroll_length: int
    sgd_steps: int
    epoch_steps: int
    num_minibatches: int
    epsilon: float
    gamma: float
    vf_coef: float
    entropy_coef: float
    gae_lambda: float
    num_envs: int
    normalize_advantage: bool
    max_grad_norm: float
    debug: bool
    env_jitable: bool

@flax.struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    params: ActorCriticParams
    env_params: Any



def gradient_update_fn(
    loss_fn: Callable[..., float], optimizer: optax.GradientTransformation, has_aux=True
):
    def gradient_update(params, data, optimizer_state):
        vg_loss = jax.value_and_grad(loss_fn, has_aux=has_aux)
        value, grads = vg_loss(params, data)
        params_update, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(params, params_update)
        return value, params, optimizer_state

    return jax.jit(gradient_update)


def minibatch_step_fn(gradient_update_fn):

    def minibatch_step(carry, minibatch):
        optimizer_state, params, key = carry
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            minibatch,
            optimizer_state=optimizer_state,
        )

        return (optimizer_state, params, key), metrics

    return jax.jit(minibatch_step)


def sgd_step_fn(minibatch_step, config:PPOConfig):
    def sgd_step(carry, _, data):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        # flatten the batch from several environments
        batch_size = config.unroll_length * config.num_envs
        batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), data
                )
        # shuffling
        permutation = jax.random.permutation(key_perm, batch_size)
        shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
        
        # minibatches
        minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                )

        # do sgd
        (optimizer_state, params, _), metrics = jax.lax.scan(
            minibatch_step,
            (optimizer_state, params, key_grad),
            minibatches,
        )

        return (optimizer_state, params, key), metrics

    return jax.jit(sgd_step)

def training_step_fn(
    env, make_policy, make_value, sgd_step, config: PPOConfig
):
 
    def training_step(carry, _):
        training_state, obs, state, key = carry
        key_sgd, key_generate_unroll, key = jax.random.split(key, 3)

        # reconstruct the policy from the parameters
        policy = jax.jit(make_policy(training_state.params.actor_params))
        value = jax.jit(make_value(training_state.params.critic_params))

        # collect the data for the batch
        (last_obs, last_state), data = generate_unroll(key_generate_unroll, 
                                                    env, 
                                                    policy, 
                                                    obs, 
                                                    state, 
                                                    config.unroll_length, 
                                                    env_params=training_state.env_params, 
                                                    num_envs=config.num_envs)


        # calculate values (baseline)
        values = value(data.obs)
        last_value = value(last_obs)

        # calculate advantages
        advantages, targets = compute_gae(data, last_value, values, config.gamma, config.gae_lambda)
        batch = (data, advantages, targets)

        # perform gradient descent
        (optimizer_state, params, _), metrics = jax.lax.scan(
            partial(sgd_step, data=batch),
            (training_state.optimizer_state, training_state.params, key_sgd),
            length=config.sgd_steps,
        )

        training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            env_params=training_state.env_params,
        )

        return (training_state, last_obs, last_state, key), metrics

    return training_step


def training_epoch_fn(training_step, config: PPOConfig):

    if config.env_jitable:    
        def training_epoch(training_state, obs, state, key):
            (training_state, obs, state, key), metrics = jax.lax.scan(
                training_step,
                (training_state, obs, state, key),
                None,
                length=config.epoch_steps,
            )

            return training_state, obs, state, metrics
    else:
        def training_epoch(training_state, obs, state, key):
            for _ in range(config.epoch_steps):
                (training_state, obs, state, key), metrics = training_step(
                    (training_state, obs, state, key), None
                )

            return training_state, obs, state, metrics

    return training_epoch

def evaluate_fn(env, make_policy, n_envs, env_params):
    # evaluation of a jittable environment, for example gymnax
    def evaluate_jittable(key, params):
        policy = make_policy(params.actor_params, deterministic=True)

        def evaluate_one_episode(key):
            obs, env_state = env.reset(key, env_params)

            def step(carry):
                obs, state, total_reward, done = carry
                action, _ = policy(obs, key)
                if len(action.shape) > 0:
                    action = jnp.squeeze(action, axis=-1)
                n_obs, n_state, reward, done, _ = env.step(key, state, action, env_params)
                return n_obs, n_state, total_reward + reward, done
            
            def cond(carry):
                obs, state, total_reward, done = carry
                return jnp.logical_not(done)

            _, _, total_reward, _ = jax.lax.while_loop(cond, step, (obs, env_state, 0.0, False))

            return total_reward

        key_v = jax.random.split(key, n_envs)
        total_rewards = jax.vmap(evaluate_one_episode)(key_v)
        return total_rewards
    
    # evaluation of a non-jittable gymnasium environment
    def evaluate_non_jittable(key, params):
        policy = jax.jit(make_policy(params.actor_params, deterministic=True))

        def evaluate_one_episode(key):
            seed = jax.random.randint(key, (), 0, 2**8).item()
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            done = False

            while not done:
                action, _ = policy(obs, seed)
                obs, reward, terminated, truncated, info = env.step(action.item())
                done = terminated | truncated
                total_reward += reward
            return total_reward
        
        total_rewards = []
        
        # get the seeds from the key
        keys = jax.random.split(key, n_envs)
        for key in keys:
            total_rewards.append(evaluate_one_episode(key))
        return total_rewards

    if isinstance(env, EnvGymnax) or isinstance(env, BeliefWrapper):
        return jax.jit(evaluate_jittable)
    elif isinstance(env, EnvGymnasium):
        return evaluate_non_jittable
    else:
        raise NotImplementedError



def train(
    environment: EnvGymnax,
    env_params: Dict,
    policy_maker: ActorCriticMaker,
    make_inference_fn,
    config: PPOConfig = PPOConfig(
        learning_rate=2.5e-3,
        num_checkpoints=150,          # number of evaluations
        epochs_per_checkpoint=1,    # number of epochs between evaluations
        unroll_length=1024,         # length of the experience buffer
        sgd_steps=4,                # num of sgd passes through the same experience buffer
        epoch_steps=4,            # num of training steps per epoch
        num_minibatches=4,          # num of minibatches for each sgd pass
        num_envs=1,                 # num of parallel environments
        seed=0,
        epsilon=0.2,
        gamma=0.99,
        vf_coef=0.5,
        entropy_coef=0.01,
        gae_lambda=0.98,
        normalize_advantage=True,
        max_grad_norm=0.5,
        debug=True,
        env_jitable=True
    ),
):
    if config.debug:
        print("Training started")
        print("Architecture: ", policy_maker.__name__)
        if isinstance(environment, gymnasium.vector.VectorEnv):
            print("Environment: ", environment.spec.id)
        else:
            print("Environment: ", environment.name)
        print("Seed: ", config.seed) 

    # parallelization!
    key = jax.random.key(config.seed)

    env = environment
    if isinstance(env, gymnasium.vector.VectorEnv):
        validation_env = gymnasium.make(env.spec.id)
    elif isinstance(env, BeliefWrapper):
        validation_env = env
    elif isinstance(env, EnvGymnax):
        validation_env = env
        

    actor_critic = policy_maker(env, env_params=env_params)

    make_policy, make_value = make_inference_fn(actor_critic)

    optimizer = optax.adam(config.learning_rate)

    if config.max_grad_norm is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optimizer)

    loss_fn = partial(
        compute_loss_fn,
        actor_critic=actor_critic,
        epsilon=config.epsilon,
        vf_coef=config.vf_coef,
        entropy_coef=config.entropy_coef,
        normalize_advantage=config.normalize_advantage,
    )

    # loss_fn = jax.jit(loss_fn)

    # gradient update step calculates the loss by applying the actor-critic to a minibatch
    # and then updates the parameters of the policy according to the gradient
    gradient_update = gradient_update_fn(loss_fn, optimizer)

    # a minibatch step calculates the gradient update for a single minibatch
    # possible extension: batch normalization
    minibatch_step = minibatch_step_fn(gradient_update)

    # a sgd step runs several updates over the same rollout
    sgd_step = sgd_step_fn(minibatch_step, config)

    # a training step takes one rollout, calculates advantages runs and several steps of sgd on it
    training_step = training_step_fn(
        env, make_policy, make_value, sgd_step, config
    )

    # an epoch consists of several training steps
    training_epoch = training_epoch_fn(training_step, config)

    #
    # training initialiazation
    #

    key_env, key_init_params, key_training, key_eval = jax.random.split(key, 4)

    if isinstance(env, EnvGymnax) or isinstance(env, BeliefWrapper):
        reset_fn = jax.vmap(env.reset, in_axes=(0, None))
        key_envs = jax.random.split(key_env, config.num_envs)
        obs_v, env_state_v = reset_fn(key_envs, env_params)

    elif isinstance(env, gymnasium.vector.VectorEnv):
        obs_v, info = env.reset(seed=config.seed)
        obs_v = jnp.array(obs_v)
        env_state_v = jnp.array([])

    key_actor, key_critic = jax.random.split(key_init_params)

    init_policy_params = ActorCriticParams(
        actor_critic.actor.init(key_actor),
        actor_critic.critic.init(key_critic),
    )

    training_state = TrainingState(
        optimizer.init(init_policy_params), init_policy_params, env_params
    )


    evaluate = evaluate_fn(validation_env, make_policy, 5, env_params)
    
    print("Total number of timesteps: ", config.num_checkpoints * config.epochs_per_checkpoint * config.epoch_steps * config.num_envs * config.unroll_length)

    checkpoints = []
    checkpoints.append(evaluate(key_eval, training_state.params))

    if config.debug:
        iterator = trange(config.num_checkpoints, desc="Checkpoints")
    else:
        iterator = range(config.num_checkpoints)

    for it in iterator:
        # training epochs
        for i in range(config.epochs_per_checkpoint):
            training_state, obs_v, env_state_v, metrics = training_epoch(
                training_state, obs_v, env_state_v, key_training
            )

        # evaluation
        evaluation_returns = evaluate(key_eval, training_state.params)
        checkpoints.append(evaluation_returns)

    return training_state, jnp.array(checkpoints)