# BordAX

<div align="center">

**A High-Performance JAX Framework for Programmatic Reinforcement Learning**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.38-orange.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## Overview

BordAX is a research-focused framework for **Programmatic Reinforcement Learning (PRL)** that combines the speed of JAX with support for structured, interpretable policies.

### Key Features

- 🚀 **High Performance**: Fully JIT-compiled training pipelines leveraging JAX's XLA compilation
- 🧩 **Modular Architecture**: Clean separation between agents, algorithms, environments, and training logic
- 🎯 **Multiple Policy Types**: Support for MLPs, boolean functions (HyperBool), and decision trees (DTSemNet)
- 🔄 **Flexible Algorithms**: Built-in PPO (on-policy) and DQN (off-policy) with easy extensibility
- 🔧 **Extensible**: Simple APIs for adding new agents, algorithms, and environments

---

## Installation

### Setup

```bash
# Clone the repository
git clone https://github.com/SynthesisLab/bordax.git
cd bordax

# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Test

Verify your installation:

```bash
python -c "from bordax.trainer import Trainer; print('✓ BordAX installed successfully')"
```

---

## Quick Start

### Training PPO on CartPole

```bash
python train_ppo.py
```

This will:
- Train an agent with MLP policy using PPO on CartPole-v1
- Save results to `runs/ppo_YYYYMMDD_HHMMSS/`
- Generate training plots (rewards, policy loss, value loss, entropy)

Expected results:
- Solves CartPole-v1 (reward = 500) in ~100k steps
- Training time: ~6 seconds on CPU

![PPO Training Rewards](https://github.com/SynthesisLab/bordax/blob/main/imgs/evaluation_rewards_ppo.png?raw=true)

### Training DQN on CartPole

```bash
python train_dqn.py
```

Expected results:
- Solves CartPole-v1 (reward = 500) in ~50k steps
- Training time: ~30 seconds on CPU

![DQN Training Rewards](https://github.com/SynthesisLab/bordax/blob/main/imgs/evaluation_rewards_dqn.png?raw=true)

### Custom Training Script

```python
from bordax.trainer import Trainer, TrainerConfig
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent
import jax

# Setup environment
env = make_env("gymnax/CartPole-v1", {}, num_envs=1)
eval_env = make_env("gymnax/CartPole-v1", {}, num_envs=1)

# Create agent
agent = make_agent("mlp/mlp", env, {
    "policy_layers": [64, 64],
    "value_layers": [64, 64],
})

# Configure algorithm
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
    save_model=True,
)

trainer = Trainer(env, eval_env, agent, algorithm, config)

# Initialize and train
key = jax.random.PRNGKey(0)
init_key, train_key = jax.random.split(key)
trainer.init(init_key)

metrics, eval_data, model_params = trainer.run(train_key)
```

---

## Architecture

BordAX uses a modular pipeline architecture:

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
| **Agent** | Defines policy and value networks | `MLPPolicyValue`, `DQNAgent` |
| **Algorithm** | Bundles training pipeline components | `ppo_algo()`, `dqn_algo()` |
| **Collector** | Generates transitions via environment interaction | `OnPolicyCollector`, `EpsGreedyCollector` |
| **BatchBuilder** | Transforms data into training batches | `MiniBatch`, `UniformReplayBatch` |
| **Updater** | Updates parameters using loss functions | `SGDUpdate`, `DQNUpdater` |
| **Trainer** | Orchestrates full training loop | `Trainer` |

### Supported Algorithms

- **PPO**
- **DQN**
---

## Project Structure

```
bordax/
├── bordax/                   # Main package
│   ├── agents/               # Agent definitions
│   │   ├── base.py           # Base classes and implementations
│   │   ├── components.py     # Neural network modules
│   │   └── utils.py          # Agent factory
│   ├── algorithms/           # RL algorithms
│   │   ├── base.py           # Algorithm implementations
│   │   ├── losses.py         # Algorithm-specific losses
│   │   └── utils.py          # Algorithm factory
│   ├── environments/         # Environment adapters (Gymnax, Gymnasium)
│   ├── batchbuilders.py      # Batch construction
│   ├── buffer.py             # Replay buffer
│   ├── collectors.py         # Data collection strategies
│   ├── trainer.py            # Training pipeline orchestration
│   ├── types.py              # Type definitions
│   └── updaters.py           # Model parameter updates
├── train_ppo.py              
├── train_dqn.py              
├── requirements.txt          
└── README.md                 
```

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

**HyperBool** (Boolean function-based):
```python
agent = make_agent("boolean/mlp", env, {
    "n": 4,  # Number of boolean variables
    "value_layers": [128, 64, 32],
})
```

**DTSemNet** (Decision trees):
```python
agent = make_agent("dt/mlp", env, {
    "tree_depth": 4,
    "value_layers": [64, 64],
})
```

---

## Training Outputs

Each training run creates a timestamped directory in `runs/`:

```
runs/ppo_20250113_120000/
├── best_model.pkl              # Best model parameters
├── metrics.pkl                 # Training metrics
├── evaluation_rewards.npy      # Evaluation rewards per checkpoint
├── evaluation_rewards.png      # Performance over time
├── policy_loss.png             # Policy loss curve
├── value_loss.png              # Value loss curve
└── entropy_loss.png            # Entropy curve
```

---

## Advanced Usage

### Custom Agent

```python
from bordax.agents.base import Agent

class MyAgent(Agent):
    def init(self, key, sample_obs):
        # Initialize parameters
        return params
    
    def policy(self, params, obs, key):
        # Return distribution and info
        return distribution, info
    
    def value(self, params, obs):
        # Return value estimate
        return value

```

### Custom Algorithm

```python
from bordax.algorithms.base import Algorithm

def my_algo(lr=1e-4, **kwargs):
    return Algorithm(
        collector=MyCollector(...),
        batch_builder=MyBatchBuilder(...),
        updater=MyUpdater(...),
    )
```

### Environment Adapters

BordAX supports both Gymnax and Gymnasium environments:

```python
# Gymnax (JAX-native, fastest)
env = make_env("gymnax/CartPole-v1", {}, num_envs=1)

# Gymnasium (Python, more environments)
env = make_env("gymnasium/CartPole-v1", {}, num_envs=1)
```

---


## License

BordAX is released under the [MIT License](LICENSE).

```
Copyright (c) 2025 SynthesisLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## Acknowledgments

BordAX builds on excellent work from the JAX ecosystem:

- [JAX](https://github.com/google/jax): High-performance numerical computing
- [Flax](https://github.com/google/flax): Neural network library
- [Gymnax](https://github.com/RobertTLange/gymnax): JAX-compatible RL environments
- [Optax](https://github.com/deepmind/optax): Gradient processing and optimization
- [Distrax](https://github.com/deepmind/distrax): Probability distributions

</div>