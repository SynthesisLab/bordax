# BordAX

**A High-Performance JAX Framework for Programmatic Reinforcement Learning**

BordAX is a research-focused framework for Programmatic Reinforcement Learning (PRL) that combines the speed of JAX with support for structured, interpretable policies including neural networks, boolean functions, and decision trees.

## Key Features

- **High Performance** — Fully JIT-compiled training pipelines leveraging JAX's XLA compilation
- **Modular Architecture** — Clean separation between agents, algorithms, environments, and training
- **Multiple Policy Types** — MLPs, boolean functions (HyperBool), and decision trees (DTSemNet)
- **Flexible Algorithms** — Built-in PPO (on-policy) and DQN (off-policy) with easy extensibility
- **Environment Agnostic** — Supports both Gymnax (JIT-compiled) and Gymnasium environments

## Quick Start

```bash
pip install bordax
```

```python
import jax
from bordax.training.trainer import Trainer, TrainerConfig
from bordax.algorithms.utils import make_algo
from bordax.environments.utils import make_env
from bordax.agents.utils import make_agent

env = make_env("gymnax/CartPole-v1", {}, num_envs=4)
eval_env = make_env("gymnax/CartPole-v1", {}, num_envs=1)
agent = make_agent("mlp/mlp", env, {"policy_layers": [64, 64]})
algo = make_algo("ppo", {"lr": 3e-4, "rollout_length": 256})
config = TrainerConfig(num_checkpoints=100, epochs_per_checkpoint=1, evaluation_episodes=32)

trainer = Trainer(env, eval_env, agent, algo, config)
trainer.init(jax.random.PRNGKey(0))
trainer.run(jax.random.PRNGKey(1))
```

## Links

- [GitHub](https://github.com/SynthesisLab/bordax)
- [Issues](https://github.com/SynthesisLab/bordax/issues)
- [CONTRIBUTING](https://github.com/SynthesisLab/bordax/blob/main/CONTRIBUTING.md)
