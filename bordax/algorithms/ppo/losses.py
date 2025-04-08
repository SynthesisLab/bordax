import jax.numpy as jnp

from bordax.policies.utils import PolicyValue


def ppo_loss(
    params,  # policy parameters
    batch,  # rollout
    policy_value: PolicyValue,  #
    epsilon: float,  #
    vf_coef: float = 0.5,  # these are fixed for the whole training process
    entropy_coef: float = 1e-4,  #
    normalize_advantage: bool = True,  #
):

    policy = policy_value.policy
    value = policy_value.value

    (rollout, advantages, targets) = batch

    pi, _ = policy.get_distribution(params.policy_params, rollout.obs)
    values = value.get_value(params.value_params, rollout.obs)
    log_prob = pi.log_prob(rollout.action)

    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # policy loss
    ratio = jnp.exp(log_prob - rollout.log_prob)
    surrogate_loss1 = ratio * advantages
    surrogate_loss2 = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # value loss
    value_loss = 0.5 * jnp.mean(jnp.square(values - targets))

    # entropy loss
    entropy_loss = pi.entropy().mean()

    total_loss = policy_loss + (vf_coef * value_loss) - (entropy_coef * entropy_loss)

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
    }


def ppo_explain_loss(
    params,  # policy parameters
    batch,  # rollout
    layer_entropy_coef,
    policy_value: PolicyValue,  #
    epsilon: float,  #
    vf_coef: float = 0.5,  # these are fixed for the whole training process
    entropy_coef: float = 1e-4,  #
    normalize_advantage: bool = True,  #
):

    policy = policy_value.policy
    value = policy_value.value

    (rollout, advantages, targets) = batch

    pi, layer_distributions = policy.get_distribution(params.policy_params, rollout.obs)
    values = value.get_value(params.value_params, rollout.obs)
    log_prob = pi.log_prob(rollout.action)

    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # policy loss
    ratio = jnp.exp(log_prob - rollout.log_prob)
    surrogate_loss1 = ratio * advantages
    surrogate_loss2 = jnp.clip(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # value loss
    value_loss = 0.5 * jnp.mean(jnp.square(values - targets))

    # entropy loss
    entropy_loss = pi.entropy().mean()

    # entropy loss for pieces layers activation
    entropy_activation_loss = jnp.array(
        [pi.entropy().mean() for pi in layer_distributions]
    ).sum()

    total_loss = (
        policy_loss
        + (vf_coef * value_loss)
        - (entropy_coef * entropy_loss)
        + (layer_entropy_coef * entropy_activation_loss)
    )

    return total_loss, {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy_loss": entropy_loss,
        "activation_loss": entropy_activation_loss,
    }
