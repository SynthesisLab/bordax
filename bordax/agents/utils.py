from bordax.agents.base import Agent, BlankAgent, MLPPolicyValue, MLPPolicyValueContinuous, DQNAgent
from bordax.environments.utils import EnvAdapter

AGENT_REGISTRY = {
    "blank/blank": BlankAgent,
    "mlp/mlp": MLPPolicyValue,
    "mlp/dt": MLPPolicyValue,
    "mlp/bool": MLPPolicyValue,
    "mlp-continuous/mlp": MLPPolicyValueContinuous,
    "dqn/mlp": DQNAgent,
}

def make_agent(agent_name: str, env: EnvAdapter, agent_config: dict = {}) -> Agent:
    """Create an agent by name.

    Args:
        agent_name: Identifier in the form ``"policy/value"``. Supported values:

            - ``"mlp/mlp"`` — MLP policy with MLP value (discrete actions)
            - ``"mlp/dt"`` — DTSemNet decision-tree policy with MLP value
            - ``"mlp/bool"`` — HyperBool boolean policy with MLP value
            - ``"mlp-continuous/mlp"`` — MLP policy with MLP value (continuous actions)
            - ``"dqn/mlp"`` — DQN Q-network agent
            - ``"blank/blank"`` — uniform random agent (baseline)

        env: Environment adapter used to infer observation and action spaces.
        agent_config: Dict of hyperparameters passed to the agent constructor.
            Required keys depend on the agent type (e.g. ``policy_layers``,
            ``value_layers`` for MLP agents; ``q_layers`` for DQN).

    Returns:
        An initialised ``Agent`` instance.

    Raises:
        ValueError: If ``agent_name`` is not in the registry.
    """
    try:
        cls = AGENT_REGISTRY[agent_name]
    except KeyError:
        raise ValueError(f"Agent {agent_name} is not supported. Supported agents are: {list(AGENT_REGISTRY.keys())}")
    
    # DQNAgent doesn't need the policy_architecture parameter
    if agent_name.startswith("dqn/"):
        return cls(agent_config, env)
    
    return cls(agent_config, env, agent_name.split('/')[1])
