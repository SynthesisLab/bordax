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
    try:
        cls = AGENT_REGISTRY[agent_name]
    except KeyError:
        raise ValueError(f"Agent {agent_name} is not supported. Supported agents are: {list(AGENT_REGISTRY.keys())}")
    
    # DQNAgent doesn't need the policy_architecture parameter
    if agent_name.startswith("dqn/"):
        return cls(agent_config, env)
    
    return cls(agent_config, env, agent_name.split('/')[1])
