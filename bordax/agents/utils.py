from bordax.agents.base import Agent, BlankAgent, MLPPolicyValue

AGENT_REGISTRY = {
    'blank': BlankAgent,
    'mlp': MLPPolicyValue,
}

def make_agent(agent_name: str, agent_config: dict = {}) -> Agent:
    try:
        cls = AGENT_REGISTRY[agent_name]
    except KeyError:
        raise ValueError(f"Agent {agent_name} is not supported. Supported agents are: {list(AGENT_REGISTRY.keys())}")
    return cls(**agent_config)