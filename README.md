# BordAX: A JAX-Based Framework for Programmatic Reinforcement Learning

BordAX is a **high-performance, JAX-based framework** for **Programmatic Reinforcement Learning (PRL)**.  
It enables efficient training and evaluation of **structured policies**, including:
- **HyperBool**: A boolean function-based policy representation.
- **DTSemNet**: A decision tree model.


---

## Features
- **Fast training**: Achieves a *97× speed-up* compared to PyTorch-based implementations.
- **Modular architecture**: Decouples *policy representations, environments, and learning algorithms*, allowing easy integration of new policies and learning algorithms.
- **Programmatic policy support**: Implements *boolean function-based policies* and *differentiable decision trees*.
- **JAX-native**: Built with *Flax* for neural networks and *Gymnax* and *Gymnasium* for reinforcement learning environments.

---

## Usage

BordAX requires **Python 3.12+**. You can install the dependencies it via:

```bash
git clone https://github.com/your-org/bordax.git
cd bordax
pip install -r requirements.txt 
```

In order to run the CartPole experiments presented in the paper, run 
```
python ./test.py
```