# BordAX

<div align="center">

**A High-Performance JAX Framework for Programmatic Reinforcement Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.8.0+-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/SynthesisLab/bordax/actions/workflows/ci.yml/badge.svg)](https://github.com/SynthesisLab/bordax/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/SynthesisLab/bordax/branch/main/graph/badge.svg)](https://codecov.io/gh/SynthesisLab/bordax)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://synthesislab.github.io/bordax)

</div>

---

## Overview

BordAX is a research-focused framework for **Programmatic Reinforcement Learning (PRL)** that combines the speed of JAX with support for structured, interpretable policies including neural networks, boolean functions, and decision trees.

### Key Features

- **High Performance** — Fully JIT-compiled training pipelines leveraging JAX's XLA compilation
- **Modular Architecture** — Clean separation between agents, algorithms, environments, and training
- **Multiple Policy Types** — MLPs, boolean functions (HyperBool), and decision trees (DTSemNet)
- **Flexible Algorithms** — Built-in PPO (on-policy) and DQN (off-policy) with easy extensibility
- **Environment Agnostic** — Supports both Gymnax (JIT-compiled) and Gymnasium environments
- **Production Ready** — Checkpointing, logging, WandB integration, and comprehensive tests

---

## Performance

BordAX achieves high performance through:

- **Full JIT compilation** for jittable environments (Gymnax)
- **Vectorized environments** via `jax.vmap`
- **Efficient loops** using `jax.lax.scan`
- **Pure functional design** compatible with XLA optimization

### JIT Compilation Strategy

| Environment | Algorithm | JIT Scope |
|-------------|-----------|-----------|
| Gymnax | On-policy | Entire `train_step` |
| Gymnax | Off-policy | `update` only |
| Gymnasium | Any | `update` only |

### Benchmark: BordAX vs Stable-Baselines3

PPO on CartPole-v1 with identical hyperparameters (5 seeds, 51k timesteps):

| Framework | Training Time | Throughput | Speedup |
|-----------|--------------|------------|---------|
| **BordAX + Gymnax** (Full JIT) | 3.73s ± 0.06s | **13,709 steps/s** | **2.8x** |
| **BordAX + Gymnasium** | 5.41s ± 0.11s | 9,468 steps/s | 1.9x |
| Stable-Baselines3 | 10.40s ± 0.05s | 4,923 steps/s | 1.0x |

<p align="center">
  <img src="https://github.com/SynthesisLab/bordax/blob/main/imgs/comparison.png?raw=true" alt="Benchmark Comparison" width="800"/>
</p>

With Gymnax (fully JIT-compiled), BordAX is **2.8x faster** than Stable-Baselines3.
Even with Gymnasium (Python environment), BordAX is **1.9x faster**.

*Measured on Apple M3 Pro (2023) with JAX 0.9.0, Stable-Baselines3 2.8.0, PyTorch 2.11.0.*

Run the benchmark yourself:
```bash
uv sync --extra benchmark
uv run python compare_sb3.py
```


---

## Installation

```bash
# Clone the repository
git clone https://github.com/SynthesisLab/bordax.git
cd bordax

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .

# With optional dependencies (WandB, visualization)
pip install -e ".[all]"
```

### Verify Installation

```bash
python -c "from bordax.training.trainer import Trainer; print('BordAX installed successfully')"
```

---

## Quick Start

### Train PPO on CartPole

```bash
python train_ppo.py
```

- Solves CartPole-v1 (reward = 500) in ~400k steps
- Training time: ~18 seconds on CPU
- Throughput: ~23,000 steps/s

<p align="center">
  <img src="https://github.com/SynthesisLab/bordax/blob/main/imgs/evaluation_rewards_ppo.png?raw=true" alt="PPO Training" width="700"/>
</p>

### Train DQN on CartPole

```bash
python train_dqn.py
```

- Solves CartPole-v1 in ~50k steps
- Training time: ~36 seconds on CPU
- Includes 1,000 step warmup phase

<p align="center">
  <img src="https://github.com/SynthesisLab/bordax/blob/main/imgs/evaluation_rewards_dqn.png?raw=true" alt="DQN Training" width="700"/>
</p>

### Custom Training Script

```python
import jax
from bordax.training.trainer import Trainer, TrainerConfig
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent

# Setup environments
env = make_env("gymnax/CartPole-v1", {"init_config": {}, "reset_config": {}}, num_envs=4)
eval_env = make_env("gymnax/CartPole-v1", {"init_config": {}, "reset_config": {}}, num_envs=1)

# Create agent with MLP policy and value networks
agent = make_agent("mlp/mlp", env, {
    "policy_layers": [64, 64],
    "value_layers": [64, 64],
})

# Configure PPO algorithm
algorithm = make_algo("ppo", {
    "lr": 3e-4,
    "rollout_length": 2048,
    "gamma": 0.99,
    "_lambda": 0.95,
    "clip_schedule": lambda _: 0.2,
    "vf_schedule": lambda _: 0.5,
    "ent_schedule": lambda _: 0.01,
    "num_minibatches": 16,
    "num_sgd_steps": 10,
})

# Setup trainer
config = TrainerConfig(
    num_checkpoints=100,
    epochs_per_checkpoint=1,
    evaluation_episodes=32,
    debug=True,
)

trainer = Trainer(env, eval_env, agent, algorithm, config)

# Train
key = jax.random.PRNGKey(0)
init_key, train_key = jax.random.split(key)
trainer.init(init_key)
eval_data = trainer.run(train_key)
```

---

## Architecture

BordAX uses a modular pipeline architecture that cleanly separates concerns:

```
Trainer
  └─> Algorithm (Collector + BatchBuilder + Updater)
       ├─> Collector: Generates environment transitions
       ├─> BatchBuilder: Constructs training batches
       └─> Updater: Computes gradients and updates parameters
```

### Core Components

| Component | Purpose | Examples |
|-----------|---------|----------|
| **Agent** | Defines policy and value networks | `MLPPolicyValue`, `BooleanPolicyValue`, `DTPolicy`, `DQNAgent` |
| **Algorithm** | Bundles training pipeline components | `ppo_algo()`, `dqn_algo()` |
| **Collector** | Generates transitions via env interaction | `OnPolicyCollector`, `EpsGreedyCollector` |
| **BatchBuilder** | Transforms data into training batches | `FullBufferBatch`, `MiniBatch`, `UniformReplayBatch` |
| **Updater** | Updates parameters using gradients | `SGDUpdate`, `DQNUpdater` |
| **Trainer** | Orchestrates full training loop | `Trainer` |

### Supported Algorithms

| Algorithm | Type | Collector | Batch Strategy |
|-----------|------|-----------|----------------|
| **PPO** | On-policy | `OnPolicyCollector` | `FullBufferBatch` → `MiniBatch` |
| **DQN** | Off-policy | `EpsGreedyCollector` | `UniformReplayBatch` |

---

## Policy Representations

### Standard Neural Networks

**MLP Policy-Value** (`mlp/mlp`):
```python
agent = make_agent("mlp/mlp", env, {
    "policy_layers": [128, 128, 64],
    "value_layers": [128, 128, 64],
})
```

### Programmatic Policies

**HyperBool** — Boolean function-based policies (`mlp/bool`):
```python
agent = make_agent("mlp/bool", env, {
    "n": 4,  # Number of boolean variables
    "value_layers": [128, 64, 32],
})
```

**DTSemNet** — Decision tree policies (`mlp/dt`):
```python
agent = make_agent("mlp/dt", env, {
    "tree_depth": 4,
    "value_layers": [64, 64],
})
```

### DQN Agent

**Q-Network** (`dqn/mlp`):
```python
agent = make_agent("dqn/mlp", env, {
    "q_layers": [64, 64],
})
```

---

## Project Structure

```
bordax/
├── bordax/
│   ├── agents/              # Agent implementations
│   │   ├── base.py          # MLPPolicyValue, BooleanPolicyValue, DTPolicy, DQNAgent
│   │   ├── components.py    # Neural modules (MLP, DTSemNet, BooleanFunction)
│   │   └── utils.py         # make_agent() factory
│   ├── algorithms/          # RL algorithms
│   │   ├── base.py          # Algorithm class, ppo_algo(), dqn_algo()
│   │   ├── losses.py        # PPOLoss, DQNLoss
│   │   └── utils.py         # make_algo() factory
│   ├── data/                # Data collection and batching
│   │   ├── collectors.py    # OnPolicyCollector, EpsGreedyCollector
│   │   ├── batchbuilders.py # Batch transformations
│   │   └── buffer.py        # ReplayBuffer
│   ├── environments/        # Environment adapters
│   │   └── utils.py         # EnvAdapter, make_env()
│   ├── training/            # Training infrastructure
│   │   ├── trainer.py       # Main Trainer class
│   │   ├── evaluation.py    # Evaluator
│   │   ├── logging.py       # Logger with WandB support
│   │   ├── checkpointing.py # Model checkpointing (Orbax)
│   │   └── updaters.py      # SGDUpdate, DQNUpdater
│   └── types.py             # Core type definitions
├── tests/                   # Test suite (48 tests, 77% coverage)
│   ├── unit/                # Fast component tests
│   ├── integration/         # Pipeline tests
│   └── slow/                # Learning verification tests
├── train_ppo.py             # PPO training example
├── train_dqn.py             # DQN training example
└── compare_sb3.py           # Stable-Baselines3 benchmark
```

---

## Testing

BordAX has a comprehensive test suite with **48 tests** achieving **77% code coverage**.

```bash
# install test dependencies
uv sync --extra dev

# run all tests (excluding slow)
uv run python -m pytest tests/ -m "not slow" -v

# run slow learning tests
uv run python -m pytest tests/ -m slow -v

# run with coverage
uv run python -m pytest tests/ --cov=bordax --cov-report=term-missing
```

### Test Categories

| Category | Tests | Purpose |
|----------|-------|---------|
| **Unit** | 44 | Fast component tests |
| **Integration** | 2 | Full pipeline verification |
| **Slow** | 2 | Learning verification |


---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| JAX | >=0.8.0 | Core computation |
| Flax | >=0.12.0 | Neural networks |
| Optax | >=0.2.6 | Optimizers |
| Gymnax | >=0.0.9 | JAX environments |
| Gymnasium | >=1.2.0 | Standard environments |
| Distrax | >=0.1.7 | Distributions |
| Orbax | >=0.11.32 | Checkpointing |

Optional: WandB (experiment tracking), Matplotlib/Seaborn (visualization)

---

## Restoring from Checkpoints

```bash
# Restore last checkpoint and continue training
python train_ppo.py --restore-last
```

---

## License

BordAX is released under the [MIT License](LICENSE).

## Support

For questions, bug reports, and feature requests, please [open a GitHub Issue](https://github.com/SynthesisLab/bordax/issues). See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Acknowledgments

BordAX builds on excellent work from the JAX ecosystem:

- [JAX](https://github.com/google/jax) — High-performance numerical computing
- [Flax](https://github.com/google/flax) — Neural network library
- [Gymnax](https://github.com/RobertTLange/gymnax) — JAX-compatible RL environments
- [Optax](https://github.com/deepmind/optax) — Gradient processing and optimization
- [Distrax](https://github.com/deepmind/distrax) — Probability distributions

---

<div align="center">
<sub>Built with JAX for speed and interpretability</sub>
</div>
