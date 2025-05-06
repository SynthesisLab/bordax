from bordax.algorithms.base import Algorithm, ppo_algo

ALGO_REGISTRY = {
    "ppo": ppo_algo,
}


def make_algo(algo_name: str, algo_config: dict = {}) -> Algorithm:
    try:
        alg = ALGO_REGISTRY[algo_name]
    except KeyError:
        raise ValueError(
            f"Algo {algo_name} is not supported. Supported algos are: {list(ALGO_REGISTRY.keys())}"
        )
    return alg(**algo_config)
