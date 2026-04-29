"""
Comparison of BordAX vs Stable-Baselines3 PPO performance.

This script trains PPO on CartPole-v1 using three configurations:
1. BordAX + Gymnax (fully JIT-compiled training loop)
2. BordAX + Gymnasium (only update step JIT-compiled)
3. Stable-Baselines3 + Gymnasium (PyTorch)

Key configuration details to ensure fair comparison:
- Same network architecture: [128, 128, 64] for both policy and value
- Same activation function: ReLU
- Same initialization: Orthogonal
- Same gradient clipping: max_norm=0.5
- Same evaluation frequency: every 2048 steps
- Same total timesteps: 51,200

Notable difference:
- BordAX normalizes both advantages AND value targets per minibatch
- SB3 only normalizes advantages (standard PPO)
"""
import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gymnasium as gym

# BordAX imports
import jax
from bordax.training.trainer import Trainer, TrainerConfig
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent

# Stable-Baselines3 imports
from stable_baselines3 import PPO as SB3_PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import torch.nn as nn


# Shared hyperparameters
ENV_NAME = "CartPole-v1"
NUM_ENVS = 4
ROLLOUT_TOTAL = 1_024  # Total env steps collected per checkpoint across all envs
ROLLOUT_PER_ENV = ROLLOUT_TOTAL // NUM_ENVS
TOTAL_TIMESTEPS = 51_200  # Overall env interactions across training
NUM_CHECKPOINTS = TOTAL_TIMESTEPS // ROLLOUT_TOTAL
EVAL_FREQ = TOTAL_TIMESTEPS // NUM_CHECKPOINTS
NUM_EVAL_EPISODES = 1
LEARNING_RATE = 1e-4  # Standard PPO learning rate
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
VF_COEF = 0.5
ENT_COEF = 0.01
MINIBATCH_SIZE = 64
NUM_MINIBATCHES = ROLLOUT_TOTAL // MINIBATCH_SIZE
NUM_SGD_EPOCHS = 10
# Multiple runs / seeds
SEEDS = [0, 1, 2, 3, 4]

# Network architecture (shared)
POLICY_LAYERS = [128, 128, 64]
VALUE_LAYERS = [128, 128, 64]


class EvalCallback(BaseCallback):
    """Callback for evaluating SB3 agent periodically."""
    
    def __init__(self, eval_env, eval_freq, n_eval_episodes=10):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.start_time = None
        
    def _on_training_start(self):
        self.start_time = time.time()
        
    def _on_step(self):
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(mean_reward)
            print(f"  Step {self.num_timesteps:6d}: {mean_reward:.2f} ± {std_reward:.2f}")
        return True


def train_bordax(seed: int):
    """Train using BordAX framework."""
    print("=" * 70)
    print(f" Training with BordAX (seed={seed})")
    print("=" * 70)
    
    # Environment
    env = make_env('gymnasium/CartPole-v1', {'init_config': {}, 'reset_config': {}}, num_envs=NUM_ENVS)
    eval_env = make_env('gymnasium/CartPole-v1', {'init_config': {}, 'reset_config': {}}, 1)
    
    # Agent
    agent_config = {
        "policy_layers": POLICY_LAYERS,
        "value_layers": VALUE_LAYERS,
    }
    agent = make_agent("mlp/mlp", env, agent_config)
    
    # Algorithm
    algo_config = {
        "lr": LEARNING_RATE,
        "rollout_length": ROLLOUT_PER_ENV,
        "gamma": GAMMA,
        "clip_schedule": lambda _: CLIP_RANGE,
        "vf_schedule": lambda _: VF_COEF,
        "ent_schedule": lambda _: ENT_COEF,
        "_lambda": GAE_LAMBDA,
        "num_minibatches": NUM_MINIBATCHES,
        "num_sgd_steps": NUM_SGD_EPOCHS,
    "num_envs": NUM_ENVS,
    }
    algorithm = make_algo("ppo", algo_config)
    
    # Calculate number of checkpoints
    num_checkpoints = TOTAL_TIMESTEPS // ROLLOUT_TOTAL
    
    # Training configuration
    training_config = TrainerConfig(
        num_checkpoints=num_checkpoints,
        epochs_per_checkpoint=1,
        evaluation_episodes=NUM_EVAL_EPISODES,
        debug=False,
        enable_evaluation=True,
    )
    
    trainer = Trainer(env, eval_env, agent, algorithm, training_config)
    key = jax.random.PRNGKey(seed)
    init_key, key = jax.random.split(key)
    
    print(f"\nHyperparameters:")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Rollout length (total): {ROLLOUT_TOTAL}")
    print(f"  Steps per env: {ROLLOUT_PER_ENV}")
    print(f"  Num checkpoints: {num_checkpoints}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Gamma: {GAMMA}")
    print(f"  GAE lambda: {GAE_LAMBDA}")
    print(f"  Clip range: {CLIP_RANGE}")
    print(f"  VF coef: {VF_COEF}")
    print(f"  Entropy coef: {ENT_COEF}")
    print(f"  Minibatches: {NUM_MINIBATCHES}")
    print(f"  SGD epochs: {NUM_SGD_EPOCHS}")
    print(f"  Evaluation episodes: {NUM_EVAL_EPISODES}")
    
    print(f"\nInitializing...")
    trainer.init(init_key)
    
    # Warmup: Run a few checkpoints to trigger all JIT compilation
    print(f"Warming up (JIT compilation)...")
    warmup_start = time.time()
    
    # Create a warmup trainer with just a few checkpoints
    warmup_config = TrainerConfig(
        num_checkpoints=3,  # Just 3 checkpoints to compile everything
        epochs_per_checkpoint=1,
        evaluation_episodes=NUM_EVAL_EPISODES,
        debug=False,
        enable_evaluation=False,  # No evaluation during warmup
    )
    warmup_trainer = Trainer(env, eval_env, agent, algorithm, warmup_config)
    warmup_trainer.init(init_key)
    
    warmup_key, key = jax.random.split(key)
    _ = warmup_trainer.run(warmup_key)
    
    # Force synchronization to ensure compilation is complete
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, 
        warmup_trainer.training_state
    )
    
    warmup_time = time.time() - warmup_start
    print(f"  Warmup complete, all functions compiled ({warmup_time:.2f}s)")
    
    # Now create the actual trainer for timed training
    print(f"\nTraining (timed, excluding warmup)...")
    actual_trainer = Trainer(env, eval_env, agent, algorithm, training_config)
    actual_trainer.init(init_key)
    
    start_time = time.time()
    data = actual_trainer.run(key)
    # Block until all computations are complete before stopping timer
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        data
    )
    training_time = time.time() - start_time
    
    eval_rewards = []
    eval_timesteps = []
    steps_per_checkpoint = training_config.epochs_per_checkpoint * ROLLOUT_TOTAL
    for idx, rollout in enumerate(data, start=1):
        if not rollout:
            continue
        returns = np.asarray(rollout["return"], dtype=np.float32)
        if returns.size == 0:
            continue
        eval_rewards.append(float(np.mean(returns)))
        eval_timesteps.append(idx * steps_per_checkpoint)

    final_reward = eval_rewards[-1] if eval_rewards else 0.0
    best_reward = max(eval_rewards) if eval_rewards else 0.0
    
    print(f"\n✓ Training complete!")
    print(f"  Time: {training_time:.2f}s")
    
    return {
        'timesteps': eval_timesteps,
        'rewards': eval_rewards,
        'training_time': training_time,
        'final_reward': final_reward,
        'best_reward': best_reward,
        'seed': seed,
    }


def train_bordax_gymnax(seed: int):
    """Train using BordAX framework with Gymnax (fully JIT-compiled)."""
    print("=" * 70)
    print(f" Training with BordAX + Gymnax [Full JIT] (seed={seed})")
    print("=" * 70)

    # Environment - use Gymnax for full JIT compilation
    env = make_env('gymnax/CartPole-v1', {'init_config': {}, 'reset_config': {}}, num_envs=NUM_ENVS)
    eval_env = make_env('gymnax/CartPole-v1', {'init_config': {}, 'reset_config': {}}, num_envs=1)

    # Agent
    agent_config = {
        "policy_layers": POLICY_LAYERS,
        "value_layers": VALUE_LAYERS,
    }
    agent = make_agent("mlp/mlp", env, agent_config)

    # Algorithm
    algo_config = {
        "lr": LEARNING_RATE,
        "rollout_length": ROLLOUT_PER_ENV,
        "gamma": GAMMA,
        "clip_schedule": lambda _: CLIP_RANGE,
        "vf_schedule": lambda _: VF_COEF,
        "ent_schedule": lambda _: ENT_COEF,
        "_lambda": GAE_LAMBDA,
        "num_minibatches": NUM_MINIBATCHES,
        "num_sgd_steps": NUM_SGD_EPOCHS,
        "num_envs": NUM_ENVS,
    }
    algorithm = make_algo("ppo", algo_config)

    # Calculate number of checkpoints
    num_checkpoints = TOTAL_TIMESTEPS // ROLLOUT_TOTAL

    # Training configuration
    training_config = TrainerConfig(
        num_checkpoints=num_checkpoints,
        epochs_per_checkpoint=1,
        evaluation_episodes=NUM_EVAL_EPISODES,
        debug=False,
        enable_evaluation=True,
    )

    trainer = Trainer(env, eval_env, agent, algorithm, training_config)
    key = jax.random.PRNGKey(seed)
    init_key, key = jax.random.split(key)

    print(f"\nHyperparameters:")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Rollout length (total): {ROLLOUT_TOTAL}")
    print(f"  Steps per env: {ROLLOUT_PER_ENV}")
    print(f"  Num checkpoints: {num_checkpoints}")
    print(f"  JIT scope: FULL TRAINING LOOP (env.is_jittable=True)")

    print(f"\nInitializing...")
    trainer.init(init_key)

    # Warmup: Run a few checkpoints to trigger all JIT compilation
    print(f"Warming up (JIT compilation)...")
    warmup_start = time.time()

    # Create a warmup trainer with just a few checkpoints
    warmup_config = TrainerConfig(
        num_checkpoints=3,  # Just 3 checkpoints to compile everything
        epochs_per_checkpoint=1,
        evaluation_episodes=NUM_EVAL_EPISODES,
        debug=False,
        enable_evaluation=False,  # No evaluation during warmup
    )
    warmup_trainer = Trainer(env, eval_env, agent, algorithm, warmup_config)
    warmup_trainer.init(init_key)

    warmup_key, key = jax.random.split(key)
    _ = warmup_trainer.run(warmup_key)

    # Force synchronization to ensure compilation is complete
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        warmup_trainer.training_state
    )

    warmup_time = time.time() - warmup_start
    print(f"  Warmup complete, all functions compiled ({warmup_time:.2f}s)")

    # Now create the actual trainer for timed training
    print(f"\nTraining (timed, excluding warmup)...")
    actual_trainer = Trainer(env, eval_env, agent, algorithm, training_config)
    actual_trainer.init(init_key)

    start_time = time.time()
    data = actual_trainer.run(key)
    # Block until all computations are complete before stopping timer
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x,
        data
    )
    training_time = time.time() - start_time

    eval_rewards = []
    eval_timesteps = []
    steps_per_checkpoint = training_config.epochs_per_checkpoint * ROLLOUT_TOTAL
    for idx, rollout in enumerate(data, start=1):
        if not rollout:
            continue
        returns = np.asarray(rollout["return"], dtype=np.float32)
        if returns.size == 0:
            continue
        eval_rewards.append(float(np.mean(returns)))
        eval_timesteps.append(idx * steps_per_checkpoint)

    final_reward = eval_rewards[-1] if eval_rewards else 0.0
    best_reward = max(eval_rewards) if eval_rewards else 0.0

    print(f"\n✓ Training complete!")
    print(f"  Time: {training_time:.2f}s")

    return {
        'timesteps': eval_timesteps,
        'rewards': eval_rewards,
        'training_time': training_time,
        'final_reward': final_reward,
        'best_reward': best_reward,
        'seed': seed,
    }


def train_sb3(seed: int):
    """Train using Stable-Baselines3."""
    print("\n" + "=" * 70)
    print(f" Training with Stable-Baselines3 (seed={seed})")
    print("=" * 70)
    
    # Environment
    def make_env_fn():
        return gym.make(ENV_NAME)

    def make_eval_env_fn():
        return Monitor(gym.make(ENV_NAME))
    
    env = DummyVecEnv([make_env_fn for _ in range(NUM_ENVS)])
    eval_env = DummyVecEnv([make_eval_env_fn])
    try:
        env.seed(seed)
        eval_env.seed(seed + 10_000)
    except Exception:
        pass
    
    # Custom network architecture matching BordAX
    policy_kwargs = dict(
        net_arch=dict(
            pi=POLICY_LAYERS,  # Policy network
            vf=VALUE_LAYERS,   # Value network
        ),
        activation_fn=nn.ReLU,  # BordAX uses ReLU by default
        ortho_init=True,  # Match BordAX's orthogonal initialization
    )
    
    print(f"\nHyperparameters:")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Rollout length (total): {ROLLOUT_TOTAL}")
    print(f"  Steps per env: {ROLLOUT_PER_ENV}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Gamma: {GAMMA}")
    print(f"  GAE lambda: {GAE_LAMBDA}")
    print(f"  Clip range: {CLIP_RANGE}")
    print(f"  VF coef: {VF_COEF}")
    print(f"  Entropy coef: {ENT_COEF}")
    print(f"  Batch size (total): {ROLLOUT_TOTAL}")
    print(f"  Minibatch size: {ROLLOUT_TOTAL // NUM_MINIBATCHES}")
    print(f"  SGD epochs: {NUM_SGD_EPOCHS}")
    print(f"  Evaluation episodes: {NUM_EVAL_EPISODES}")
    
    # Create PPO model
    # Optional: ensure Python/NumPy/torch rngs are seeded consistently
    try:
        import random, numpy as _np, torch as _torch
        random.seed(seed)
        _np.random.seed(seed)
        _torch.manual_seed(seed)
    except Exception:
        pass

    model = SB3_PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=ROLLOUT_PER_ENV,
        batch_size=MINIBATCH_SIZE,
        n_epochs=NUM_SGD_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        vf_coef=VF_COEF,
        ent_coef=ENT_COEF,
        max_grad_norm=0.5,  # Match BordAX's gradient clipping
        normalize_advantage=True,  # SB3 default, BordAX also normalizes per minibatch
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )
    
    # Warmup: Run a few steps to ensure PyTorch is fully initialized
    print(f"\nWarming up (PyTorch initialization)...")
    initial_policy_state = copy.deepcopy(model.policy.state_dict())
    initial_optimizer_state = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=ROLLOUT_TOTAL, reset_num_timesteps=True)
    model.policy.load_state_dict(initial_policy_state)
    model.policy.optimizer.load_state_dict(initial_optimizer_state)
    if model.ep_info_buffer is not None:
        model.ep_info_buffer.clear()
    if model.ep_success_buffer is not None:
        model.ep_success_buffer.clear()
    model._last_obs = None
    model._episode_num = 0
    model.num_timesteps = 0
    model._total_timesteps = 0
    print(f"  Warmup complete")
    
    print(f"\nTraining (timed)...")
    eval_callback = EvalCallback(eval_env, eval_freq=EVAL_FREQ, n_eval_episodes=NUM_EVAL_EPISODES)

    start_time = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, reset_num_timesteps=True, callback=eval_callback)
    training_time = time.time() - start_time
    eval_rewards = [float(r) for r in eval_callback.evaluations_results]
    eval_timesteps = [int(t) for t in eval_callback.evaluations_timesteps]
    final_reward = eval_rewards[-1] if eval_rewards else 0.0
    best_reward = max(eval_rewards) if eval_rewards else 0.0
    
    print(f"\n✓ Training complete!")
    print(f"  Time: {training_time:.2f}s")
    
    env.close()
    eval_env.close()
    
    return {
        'timesteps': eval_timesteps,
        'rewards': eval_rewards,
        'training_time': training_time,
        'final_reward': final_reward,
        'best_reward': best_reward,
        'seed': seed,
    }


def _mean_ci(values: np.ndarray):
    """Return mean and 95% CI half-width along axis=0 for a 1D array."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    mean = float(np.mean(values))
    if n <= 1:
        return mean, 0.0
    std = float(np.std(values, ddof=1))
    hw = 1.96 * std / np.sqrt(n)
    return mean, hw


def _aggregate_curves(runs: list):
    """Aggregate multiple runs with (timesteps, rewards) pairs.

    Returns:
        steps_sorted: list of common timesteps across all runs (intersection)
        mean: np.ndarray of means per timestep
        ci_hw: np.ndarray of 95% CI half-width per timestep
    """
    if not runs:
        return [], np.array([]), np.array([])

    # Build list of sets of timesteps
    step_sets = [set(run['timesteps']) for run in runs]
    common_steps = sorted(list(set.intersection(*step_sets))) if len(step_sets) > 1 else sorted(list(step_sets[0]))
    if not common_steps:
        # Fallback: align by index using the minimum run length
        min_len = min(len(run['timesteps']) for run in runs if run['timesteps'])
        idx_steps = list(range(min_len))
        vals = np.array([run['rewards'][:min_len] for run in runs], dtype=np.float64)
        mean = np.mean(vals, axis=0)
        std = np.std(vals, ddof=1, axis=0) if vals.shape[0] > 1 else np.zeros_like(mean)
        ci_hw = 1.96 * std / np.sqrt(vals.shape[0]) if vals.shape[0] > 1 else np.zeros_like(mean)
        return idx_steps, mean, ci_hw

    # Collect rewards per common timestep
    vals_per_step = []
    for t in common_steps:
        vals_t = []
        for run in runs:
            # Find reward at exact t; skip if not present
            step_to_reward = {s: r for s, r in zip(run['timesteps'], run['rewards'])}
            if t in step_to_reward:
                vals_t.append(step_to_reward[t])
        vals_per_step.append(vals_t)

    vals_arr = np.array([np.array(v, dtype=np.float64) for v in vals_per_step], dtype=object)
    means = np.array([np.mean(v) if len(v) > 0 else np.nan for v in vals_arr], dtype=np.float64)
    stds = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals_arr], dtype=np.float64)
    ns = np.array([len(v) for v in vals_arr], dtype=np.int32)
    with np.errstate(divide='ignore', invalid='ignore'):
        ci_hw = np.where(ns > 1, 1.96 * stds / np.sqrt(ns), 0.0)
    # Remove any NaN entries (if some steps had no rewards collected)
    filtered = [(t, m, c) for t, m, c, n in zip(common_steps, means, ci_hw, ns) if np.isfinite(m) and n > 0]
    if not filtered:
        return [], np.array([]), np.array([])
    steps_sorted, mean_vals, ci_vals = zip(*filtered)
    return list(steps_sorted), np.array(mean_vals), np.array(ci_vals)


def plot_comparison(bordax_gymnax_runs: list, bordax_gymnasium_runs: list, sb3_runs: list, output_dir: str):
    """Create comparison plots with averages and 95% CI across runs."""
    sns.set_theme(style="darkgrid")

    # Compute aggregated curves
    bg_steps, bg_mean, bg_ci = _aggregate_curves(bordax_gymnax_runs)
    b_steps, b_mean, b_ci = _aggregate_curves(bordax_gymnasium_runs)
    s_steps, s_mean, s_ci = _aggregate_curves(sb3_runs)

    # Prepare time aggregates
    bg_times = np.array([run['training_time'] for run in bordax_gymnax_runs], dtype=np.float64)
    b_times = np.array([run['training_time'] for run in bordax_gymnasium_runs], dtype=np.float64)
    s_times = np.array([run['training_time'] for run in sb3_runs], dtype=np.float64)
    bg_time_mean, bg_time_ci = _mean_ci(bg_times)
    b_time_mean, b_time_ci = _mean_ci(b_times)
    s_time_mean, s_time_ci = _mean_ci(s_times)

    # Create figure
    has_any_eval = (len(bg_steps) > 0) or (len(b_steps) > 0) or (len(s_steps) > 0)
    if not has_any_eval:
        print("\n⊗ Skipping training curves plot (no evaluation data)")
        fig, ax_time = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig, (ax_curve, ax_time) = plt.subplots(1, 2, figsize=(16, 5))
        # Plot mean ± CI curves
        if len(bg_steps) > 0:
            ax_curve.plot(bg_steps, bg_mean, color='#2ca02c', label='BordAX+Gymnax (Full JIT)', linewidth=2)
            ax_curve.fill_between(bg_steps, bg_mean - bg_ci, bg_mean + bg_ci, color='#2ca02c', alpha=0.25)
        if len(b_steps) > 0:
            ax_curve.plot(b_steps, b_mean, color='#1f77b4', label='BordAX+Gymnasium', linewidth=2)
            ax_curve.fill_between(b_steps, b_mean - b_ci, b_mean + b_ci, color='#1f77b4', alpha=0.25)
        if len(s_steps) > 0:
            ax_curve.plot(s_steps, s_mean, color='#ff7f0e', label='Stable-Baselines3', linewidth=2)
            ax_curve.fill_between(s_steps, s_mean - s_ci, s_mean + s_ci, color='#ff7f0e', alpha=0.25)
        ax_curve.set_xlabel('Environment Steps', fontsize=12)
        ax_curve.set_ylabel('Average Reward', fontsize=12)
        ax_curve.set_title('Training Performance (mean ± 95% CI)', fontsize=14, fontweight='bold')
        ax_curve.legend(fontsize=10)
        ax_curve.grid(True, alpha=0.3)

    # Bar chart with error bars for training time
    frameworks = ['BordAX\n+Gymnax\n(Full JIT)', 'BordAX\n+Gymnasium', 'SB3']
    means = [bg_time_mean, b_time_mean, s_time_mean]
    cis = [bg_time_ci, b_time_ci, s_time_ci]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']
    x = np.arange(len(frameworks))
    bars = ax_time.bar(x, means, yerr=cis, color=colors, alpha=0.7, edgecolor='black', capsize=8)
    ax_time.set_xticks(x)
    ax_time.set_xticklabels(frameworks)
    ax_time.set_ylabel('Training Time (seconds)', fontsize=12)
    ax_time.set_title('Training Time (mean ± 95% CI)', fontsize=14, fontweight='bold')
    ax_time.grid(True, alpha=0.3, axis='y')
    for i, (bar, m, c) in enumerate(zip(bars, means, cis)):
        height = bar.get_height()
        ax_time.text(bar.get_x() + bar.get_width()/2., height + (c if c else 0) * 0.1 + 0.2,
                     f'{m:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Comparison plot saved to '{plot_path}'")


def _print_summary(name: str, runs: list, total_timesteps: int):
    """Print summary statistics for a set of runs."""
    times = [r['training_time'] for r in runs]
    mean_t, ci_t = _mean_ci(np.array(times))
    print(f"\n{name}:")
    print(f"  Runs: {len(runs)}  Seeds: {SEEDS}")
    print(f"  Training time: {mean_t:.2f}s ± {ci_t:.2f}s (95% CI)")
    if any(r['rewards'] for r in runs):
        finals = [r['final_reward'] for r in runs if r['rewards']]
        bests = [r['best_reward'] for r in runs if r['rewards']]
        f_mean, f_ci = _mean_ci(np.array(finals)) if finals else (0.0, 0.0)
        b_mean, b_ci = _mean_ci(np.array(bests)) if bests else (0.0, 0.0)
        print(f"  Final reward: {f_mean:.2f} ± {f_ci:.2f}")
        print(f"  Best reward:  {b_mean:.2f} ± {b_ci:.2f}")
    print(f"  Throughput (mean): {total_timesteps / mean_t:.0f} steps/s")
    return mean_t, ci_t


def main():
    """Run comparison between BordAX (Gymnax & Gymnasium) and SB3 across multiple seeds."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"runs/comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(" BordAX vs Stable-Baselines3 PPO Comparison")
    print(" Environment: CartPole-v1")
    print(" Configurations:")
    print("   1. BordAX + Gymnax   (Full JIT-compiled training loop)")
    print("   2. BordAX + Gymnasium (Only update step JIT-compiled)")
    print("   3. Stable-Baselines3 (PyTorch)")
    print("=" * 70)

    # Train with all frameworks for multiple seeds
    bordax_gymnax_runs = []
    bordax_gymnasium_runs = []
    sb3_runs = []
    for i, seed in enumerate(SEEDS, start=1):
        print(f"\n{'='*70}")
        print(f" Run {i}/{len(SEEDS)} (seed={seed})")
        print(f"{'='*70}")
        bg_res = train_bordax_gymnax(seed)
        b_res = train_bordax(seed)
        s_res = train_sb3(seed)
        bordax_gymnax_runs.append(bg_res)
        bordax_gymnasium_runs.append(b_res)
        sb3_runs.append(s_res)

    # Print summary
    print("\n" + "=" * 70)
    print(" Summary (across runs)")
    print("=" * 70)

    bg_mean_t, _ = _print_summary("BordAX + Gymnax (Full JIT)", bordax_gymnax_runs, TOTAL_TIMESTEPS)
    b_mean_t, _ = _print_summary("BordAX + Gymnasium", bordax_gymnasium_runs, TOTAL_TIMESTEPS)
    s_mean_t, _ = _print_summary("Stable-Baselines3", sb3_runs, TOTAL_TIMESTEPS)

    print("\n" + "-" * 70)
    print(" Speedup Comparison (vs SB3)")
    print("-" * 70)
    speedup_gymnax = s_mean_t / bg_mean_t if bg_mean_t > 0 else float('nan')
    speedup_gymnasium = s_mean_t / b_mean_t if b_mean_t > 0 else float('nan')
    print(f"  BordAX+Gymnax (Full JIT): {speedup_gymnax:.1f}x faster than SB3")
    print(f"  BordAX+Gymnasium:         {speedup_gymnasium:.1f}x faster than SB3")
    print("-" * 70)

    # Create comparison plot
    plot_comparison(bordax_gymnax_runs, bordax_gymnasium_runs, sb3_runs, output_dir)

    # Save results
    results = {
        'bordax_gymnax_runs': bordax_gymnax_runs,
        'bordax_gymnasium_runs': bordax_gymnasium_runs,
        'sb3_runs': sb3_runs,
        'seeds': SEEDS,
        'hyperparameters': {
            'total_timesteps': TOTAL_TIMESTEPS,
            'learning_rate': LEARNING_RATE,
            'num_envs': NUM_ENVS,
            'rollout_total': ROLLOUT_TOTAL,
            'rollout_per_env': ROLLOUT_PER_ENV,
            'gamma': GAMMA,
            'gae_lambda': GAE_LAMBDA,
            'clip_range': CLIP_RANGE,
            'vf_coef': VF_COEF,
            'ent_coef': ENT_COEF,
            'num_minibatches': NUM_MINIBATCHES,
            'num_sgd_epochs': NUM_SGD_EPOCHS,
        }
    }

    import pickle
    results_path = os.path.join(output_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to '{results_path}'")

    print(f"\n✓ Output directory: {output_dir}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
