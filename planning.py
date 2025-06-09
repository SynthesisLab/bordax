import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List
import pickle
from dataclasses import dataclass

from bordax.agents.base import MLPPolicyValue
from bordax.environments.utils import EnvAdapter, make_env
from bordax.types import PRNGKey


@dataclass
class RolloutWithActivations:
    """Container for a single episode with activations."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    activations: List[Dict[str, np.ndarray]]  # Keep as list of dicts
    episode_return: float
    episode_length: int


def flatten_activations(activations_jax, batch_idx=0):
    """Completely flatten nested activation dictionary for easier access."""
    activations_np = {}

    # Get final output
    if "__call__" in activations_jax:
        activations_np["final_output"] = np.asarray(
            activations_jax["__call__"][batch_idx]
        )

    for key in activations_jax:
        if key.startswith("dense_layers_"):
            layer_num = key.split("_")[-1]
            if (
                isinstance(activations_jax[key], dict)
                and "__call__" in activations_jax[key]
            ):
                activations_np[f"layer_{layer_num}"] = np.asarray(
                    activations_jax[key]["__call__"][batch_idx]
                )

    return activations_np


def collect_rollouts_with_activations(
    env: EnvAdapter,
    agent: MLPPolicyValue,
    params: Any,
    num_episodes: int,
    key: PRNGKey,
    verbose: bool = True,
) -> List[RolloutWithActivations]:
    """Collect episode rollouts with network activations."""

    assert env.num_envs == 1, "Use single environment for activation collection"

    max_steps = env.env_params.max_steps_in_episode
    rollouts = []

    @jax.jit
    def get_action_value_activations(params, obs, key):
        pi, activations = agent.policy_activations(params, obs)
        action = pi.mode()
        value = agent.value(params, obs)
        return action, value, activations

    obs_buffer = np.zeros((max_steps, *env.obs_space().shape), dtype=np.float32)
    action_buffer = np.zeros((max_steps,) + env.action_space().shape, dtype=np.int32)
    reward_buffer = np.zeros(max_steps, dtype=np.float32)
    done_buffer = np.zeros(max_steps, dtype=bool)
    value_buffer = np.zeros(max_steps, dtype=np.float32)

    if verbose:
        from tqdm import tqdm

        pbar = tqdm(total=num_episodes, desc="Collecting rollouts")

    for episode in range(num_episodes):
        key, reset_key, episode_key = jax.random.split(key, 3)

        obs, env_state = env.reset(reset_key)

        if hasattr(obs, "shape") and obs.shape[0] == 1:
            obs = np.asarray(obs[0])
        else:
            obs = np.asarray(obs)

        all_activations = []

        done = False
        step = 0

        while not done and step < max_steps:
            step_key, action_key = jax.random.split(episode_key, 2)
            episode_key = step_key

            # Store observation in pre-allocated buffer
            obs_buffer[step] = obs

            obs_jax = jnp.array(obs[None, ...])  # Add batch dimension
            action_jax, value_jax, activations_jax = get_action_value_activations(
                params, obs_jax, action_key
            )

            # Convert to numpy
            action = np.asarray(action_jax)
            value = float(value_jax[0])
            activations_np = flatten_activations(activations_jax, batch_idx=0)

            action_buffer[step] = action[0]
            value_buffer[step] = value
            all_activations.append(activations_np)

            obs_next, env_state, reward, done, _ = env.step(step_key, env_state, action)

            # Convert to numpy
            obs = np.asarray(obs_next[0] if obs_next.shape[0] == 1 else obs_next)
            reward = np.asarray(reward).item()
            done = bool(done)

            reward_buffer[step] = reward
            done_buffer[step] = done

            step += 1

        # Slice buffers to actual episode length
        episode_length = step

        # Create rollout object with sliced numpy arrays
        rollout = RolloutWithActivations(
            observations=obs_buffer[:episode_length].copy(),
            actions=action_buffer[:episode_length].copy(),
            rewards=reward_buffer[:episode_length].copy(),
            dones=done_buffer[:episode_length].copy(),
            values=value_buffer[:episode_length].copy(),
            activations=all_activations,
            episode_return=float(reward_buffer[:episode_length].sum()),
            episode_length=episode_length,
        )

        rollouts.append(rollout)

        if verbose:
            pbar.update(1)
            if episode % 100 == 0 and episode > 0:
                avg_return = np.mean([r.episode_return for r in rollouts[-100:]])
                pbar.set_postfix({"avg_return": f"{avg_return:.2f}"})

    if verbose:
        pbar.close()

    return rollouts


def save_rollouts_pickle(rollouts: List[RolloutWithActivations], filename: str):
    """Save rollouts using pickle - handles any structure."""
    import pickle
    import gzip

    # Use gzip for compression
    with gzip.open(filename, "wb") as f:
        pickle.dump(rollouts, f)


def load_rollouts_pickle(filename: str) -> List[RolloutWithActivations]:
    """Load pickled rollouts."""
    import pickle
    import gzip

    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


# Example usage with optimizations:
if __name__ == "__main__":

    eval_env = make_env("gymnasium/LunarLander-v3", num_envs=1)

    with open("full_model.pkl", "rb") as f:
        model = pickle.load(f)
        agent: MLPPolicyValue = model["agent"]
        params = model["params"]

    agent.policy_module.return_activations = True

    key = jax.random.PRNGKey(42)

    rollouts = collect_rollouts_with_activations(
        env=eval_env,
        agent=agent,
        params=params,
        num_episodes=10000,
        key=key,
        verbose=True,
    )

    save_rollouts_pickle(rollouts, "rollouts_with_activations.pkl")

    returns = [r.episode_return for r in rollouts]
    lengths = [r.episode_length for r in rollouts]
    print(f"\nCollected {len(rollouts)} episodes")
    print(f"Average return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"Average length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
