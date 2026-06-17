from bordax.algorithms.base import Algorithm, ppo_algo, dqn_algo

ALGO_REGISTRY = {
    "ppo": ppo_algo,
    "dqn": dqn_algo,
}


def make_algo(algo_name: str, algo_config: dict = {}) -> Algorithm:
    """Create an algorithm by name.

    Args:
        algo_name: Algorithm identifier. Supported values:

            - ``"ppo"`` — Proximal Policy Optimization (on-policy)
            - ``"dqn"`` — Deep Q-Network (off-policy)

        algo_config: Dict of hyperparameters forwarded to the algorithm
            factory function. See ``ppo_algo`` and ``dqn_algo`` for the
            accepted keys.

    Returns:
        A configured ``Algorithm`` instance.

    Raises:
        ValueError: If ``algo_name`` is not in the registry.
    """
    try:
        alg = ALGO_REGISTRY[algo_name]
    except KeyError:
        raise ValueError(
            f"Algo {algo_name} is not supported. Supported algos are: {list(ALGO_REGISTRY.keys())}"
        )
    return alg(**algo_config)
