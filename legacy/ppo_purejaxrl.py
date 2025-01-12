## Adapted from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py

import os

import jax
import jax.numpy as jnp

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.core import frozen_dict
from flax.core.frozen_dict import FrozenDict

import numpy as np

from typing import Sequence, NamedTuple, List, Any
import functools

import optax
import distrax
import gymnax

from wrappers import *

from decision_tree import DTree

import time


# @functools.partial(jax.jit, static_argnums=(0, 1, 2))
def test_episode(network, params, env_name, rng):
    env, env_params = gymnax.make(env_name)
    rng, env_rng = jax.random.split(rng)
    obs, env_state = env.reset(env_rng, env_params)

    def _step(transition):
        obs, env_state, total_reward, done, rng = transition
        rng, rng_step = jax.random.split(rng)
        pi, _ = network.apply(params, obs)
        action = jnp.argmax(pi.probs)
        test_obs, test_env_state, reward, done, _ = env.step(
            rng_step, env_state, action, env_params
        )
        return (test_obs, test_env_state, total_reward + reward, done, rng)

    _, _, episode_reward, _, _ = jax.lax.while_loop(
        lambda transition: ~transition[3],
        _step,
        (obs, env_state, 0, False, rng),
    )

    return episode_reward


class MLP(nn.Module):
    layer_sizes: List[int]

    def setup(self):
        self.dense_layers = [
            nn.Dense(size, kernel_init=nn.initializers.orthogonal())
            for size in self.layer_sizes
        ]

    def __call__(self, x):
        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        return self.dense_layers[-1](x)


class NNActorCritic(nn.Module):
    actor_layer_sizes: List[int]

    def setup(self):
        self.actor = MLP(self.actor_layer_sizes)
        self.critic = MLP([16, 16, 1])

    def __call__(self, x):
        pi = distrax.Categorical(logits=self.actor(x))
        v = jnp.squeeze(self.critic(x), axis=-1)
        return pi, v


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    tree: DTree
    use_soft_sign: bool = False
    bias_mode: str = "internal"
    activation: str = "relu"

    def setup(self):
        # actor initialization
        n_nodes = self.tree.count_nodes()
        self.n_leaves = self.tree.count_leaves()
        self.n_actions = len(set(self.tree.get_labels()))
        reachability = self.tree.reachability(weighted=False)

        self.actor1 = nn.Dense(
            n_nodes,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=nn.initializers.zeros,
            use_bias=False if self.bias_mode == "external" else True,
            name="actor1",
        )

        self.actor2 = nn.Dense(  # 2
            2 * n_nodes,
            # kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor2",
        )
        self.actor3 = nn.Dense(  # 3
            self.n_leaves + (self.n_actions - (self.n_leaves % self.n_actions)),
            # kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="actor3",
        )

        # critic initialization

        self.critic1 = nn.Dense(
            16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic1",
        )
        self.critic2 = nn.Dense(
            16,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            name="critic2",
        )
        self.critic3 = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="critic3"
        )

    @nn.compact
    def __call__(self, x):

        if len(x.shape) == 1:
            x = jnp.array([x])

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = self.actor1(x)
        if self.use_soft_sign:
            actor_mean = nn.soft_sign(actor_mean)
        actor_mean = self.actor2(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = self.actor3(actor_mean)
        actor_mean = actor_mean.reshape((actor_mean.shape[0], -1, self.n_actions))
        actor_mean = actor_mean.max(axis=1)
        pi = distrax.Categorical(logits=actor_mean)

        critic = self.critic1(x)
        critic = activation(critic)
        critic = self.critic2(critic)
        critic = activation(critic)
        critic = self.critic3(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    if config["HIDE_OBS"]:
        env = HideObservationsWrapper(env)
    if config["EXTERNAL_BIAS"]:
        env = ExplicitBiasWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK

        match config["POLICY"]:
            case "dtsemnet":
                assert (
                    2 ** config["TREE_DEPTH"] >= env.action_space(env_params).n
                ), "The tree is to small for all actions"
                root = DTree.complete_tree(
                    config["TREE_DEPTH"], list(range(env.action_space(env_params).n))
                )
                n_nodes = root.count_nodes()
                n_leaves = root.count_leaves()
                n_actions = len(set(root.get_labels()))
                reachability = root.reachability(weighted=False)

                bias_mode = "external" if config["EXTERNAL_BIAS"] else "internal"

                network = ActorCritic(
                    env.action_space(env_params).n,
                    activation=config["ACTIVATION"],
                    use_soft_sign=config["SOFT_SIGN"],
                    tree=root,
                    bias_mode=bias_mode,
                )
            case "nn":
                network = NNActorCritic(config["NET_ARCH"])
            case _:
                raise ValueError(f"Unknown policy: {config['POLICY']}")

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        if config["HIDE_OBS"]:
            init_x = jnp.delete(init_x, 3)
        if config["POLICY"] == "dtsemnet":
            if bias_mode == "external":
                init_x = jnp.append(init_x, jnp.array([1.0]))

        network_params = network.init(_rng, init_x)

        updated_params = network_params

        # here we build the matrices that are used in DTSemNet
        match config["POLICY"]:
            case "dtsemnet":
                row_indices = np.arange(2 * n_nodes)
                col_indices = np.arange(n_nodes).repeat(2)
                tiles = np.tile([1.0, -1.0], n_nodes)
                matrix = np.zeros((2 * n_nodes, n_nodes), dtype=np.float32)
                matrix[row_indices, col_indices] = tiles
                updated_params["params"]["actor2"]["kernel"] = jnp.array(matrix.T)

                matrix = np.zeros(
                    (n_leaves + (n_actions - (n_leaves % n_actions)), 2 * n_nodes),
                    dtype=np.float32,
                )
                all_nodes = root.traverse()
                leaves = root.traverse_leaves()
                internal_nodes = root.traverse_internal_nodes()
                node_to_index = {node: i for i, node in enumerate(all_nodes)}
                leaf_to_index = {leaf: i for i, leaf in enumerate(leaves)}
                int_node_to_index = {node: i for i, node in enumerate(internal_nodes)}

                for int_node, id_node in int_node_to_index.items():
                    for leaf, id_leaf in leaf_to_index.items():
                        ind = reachability[node_to_index[int_node]][node_to_index[leaf]]
                        if ind == 1:
                            matrix[id_leaf][2 * id_node] = 1
                        elif ind == -1:
                            matrix[id_leaf][2 * id_node + 1] = 1
                        elif ind == 0:
                            matrix[id_leaf][2 * id_node] = 1
                            matrix[id_leaf][2 * id_node + 1] = 1

                updated_params["params"]["actor3"]["kernel"] = jnp.array(matrix.T)
            case "nn":
                pass

        network.apply(updated_params, init_x)

        match config["POLICY"]:
            case "dtsemnet":
                if config["ANNEAL_LR"]:
                    tx = optax.multi_transform(
                        {
                            "adam": optax.chain(
                                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                                optax.adam(learning_rate=linear_schedule, eps=1e-5),
                            ),
                            "zero": optax.set_to_zero(),
                        },
                        {
                            "params": {
                                "critic1": "adam",
                                "critic2": "adam",
                                "critic3": "adam",
                                "actor1": "adam",
                                "actor2": "zero",
                                "actor3": "zero",
                            }
                        },
                    )
                else:
                    tx = optax.multi_transform(
                        {
                            "adam": optax.chain(
                                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                                optax.adam(config["LR"], eps=1e-5),
                            ),
                            "zero": optax.set_to_zero(),
                        },
                        {
                            "params": {
                                "critic1": "adam",
                                "critic2": "adam",
                                "critic3": "adam",
                                "actor1": "adam",
                                "actor2": "zero",
                                "actor3": "zero",
                            }
                        },
                    )
            case "nn":
                if config["ANNEAL_LR"]:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(learning_rate=linear_schedule, eps=1e-5),
                    )
                else:
                    tx = optax.chain(
                        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                        optax.adam(config["LR"], eps=1e-5),
                    )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=updated_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)

                # try to do max instead of softmax, doesn't work now
                # action = jnp.argmax(pi.probs)

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                # advantages = (advantages - jnp.mean(advantages)) / (
                #     jnp.std(advantages) + 1e-8
                # )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Debugging mode
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, (metric, train_state.params)

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {
            "runner_state": runner_state,
            "metrics": metric,
            "parameters": updated_params,
        }

    return train


def test_weights(params_by_epoch, network, env_name, rng):
    # Define the function to test the weights
    def test_fn(params, rng):
        return test_episode(network=network, params=params, env_name=env_name, rng=rng)

    # Jit compile the function
    jitted_test_fn = jax.jit(test_fn)

    # Test each set of parameters in params_by_epoch
    results = []
    for params in params_by_epoch:
        param_result = []
        rng, test_rng = jax.random.split(rng)
        for i in range(30):
            rng, run_rng = jax.random.split(test_rng)
            score = jitted_test_fn(params, run_rng)
            param_result.append(score)
        results.append(param_result)

    return results


if __name__ == "__main__":
    original_config = {
        "LR": 0.024,
        "GAMMA": 0.99,
        "NUM_STEPS": 832,
        "NUM_ENVS": 1,
        "TOTAL_TIMESTEPS": 6e5,
        "UPDATE_EPOCHS": 14,
        "NUM_MINIBATCHES": 26,
        "GAE_LAMBDA": 0.86,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.02,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        #
        "ACTIVATION": "relu",
        # "ENV_NAME": "MountainCar-v0",
        "ENV_NAME": "CartPole-v1",
        "POLICY": "nn",  # dtsemnet or nn
        "NET_ARCH": [10, 2],
        "ANNEAL_LR": True,
        "SOFT_SIGN": False,
        "TREE_DEPTH": 4,
        "EXTERNAL_BIAS": False,
        "HIDE_OBS": False,
        "DEBUG": False,
        "SEED": 0,
    }

    config = {
        "LR": 0.002,
        "GAMMA": 0.99,
        "NUM_STEPS": 1024,
        "NUM_ENVS": 1,
        "TOTAL_TIMESTEPS": 6e5,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 16,
        "GAE_LAMBDA": 0.98,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.00,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        #
        "ACTIVATION": "relu",
        # "ENV_NAME": "MountainCar-v0",
        "ENV_NAME": "CartPole-v1",
        "POLICY": "dtsemnet",  # dtsemnet or nn
        "NET_ARCH": [3, 3],
        "ANNEAL_LR": True,
        "SOFT_SIGN": False,
        "TREE_DEPTH": 3,
        "EXTERNAL_BIAS": False,
        "HIDE_OBS": False,
        "DEBUG": False,
        "SEED": 0,
    }
    log_dir = f"logs/{config["POLICY"]}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save config to params.txt in the log directory
    with open(f"{log_dir}/params.txt", "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    seeds = list(range(300, 330))

    env, env_params = gymnax.make(config["ENV_NAME"])

    if config["POLICY"] == "dtsemnet":
        network = ActorCritic(
            env.action_space(env_params).n,
            activation=config["ACTIVATION"],
            use_soft_sign=config["SOFT_SIGN"],
            tree=DTree.complete_tree(config["TREE_DEPTH"], [0, 1]),
        )

    for seed in seeds:
        print(f"Seed {seed}")
        config.update({"SEED": seed})
        rng = jax.random.PRNGKey(config["SEED"])

        train_jit = jax.jit(make_train(config))
        start_time = time.time()
        out = train_jit(rng)
        runner_state = out["runner_state"]
        parameters = out["parameters"]
        metrics = out["metrics"][0]
        params = out["metrics"][1]

        inner_structure = jax.tree.structure(
            ["*" for _ in params["params"]["actor1"]["kernel"]]
        )
        outer_structure = jax.tree.structure(
            {
                "params": {
                    "actor1": {"bias": "*", "kernel": "*"},
                    "actor2": {"bias": "*", "kernel": "*"},
                    "actor3": {"bias": "*", "kernel": "*"},
                    "critic1": {"bias": "*", "kernel": "*"},
                    "critic2": {"bias": "*", "kernel": "*"},
                    "critic3": {"bias": "*", "kernel": "*"},
                }
            }
        )

        end_time = time.time()
        print(f"Seed running time: {end_time - start_time}")

        if config["POLICY"] == "dtsemnet":
            start_time = time.time()
            params_by_epoch = jax.tree.transpose(
                outer_treedef=outer_structure,
                inner_treedef=inner_structure,
                pytree_to_transpose=jax.tree.map(lambda x: [i for i in x], params),
            )

            params_by_epoch = [
                param
                for (i, param) in enumerate(params_by_epoch)
                if (i % len(params_by_epoch) // 100 == 0)
            ]

            results = test_weights(params_by_epoch, network, config["ENV_NAME"], rng)
            np.savetxt(f'{log_dir}/checkpoints/seed{config["SEED"]}_test.txt')
            end_time = time.time()

            print("Test time:", end_time - start_time)

        rewards_to_plot = metrics["returned_episode_returns"][
            metrics["returned_episode"]
        ]

        rew = np.array(rewards_to_plot)
        np.savetxt(f"{log_dir}/seed{config["SEED"]}_rewards.txt", rew, fmt="%.1f")
