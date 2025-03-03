import jax.numpy as jnp

from bordax.policies.utils import ActorCritic


def compute_loss_fn(
    params,  # policy parameters
    batch,  # rollout
    actor_critic: ActorCritic,  #
    epsilon: float,  #
    vf_coef: float = 0.5,  # these are fixed for the whole training process
    entropy_coef: float = 1e-4,  #
    normalize_advantage: bool = True,  #
):

    actor = actor_critic.actor
    critic = actor_critic.critic

    (rollout, advantages, targets) = batch

    pi = actor.apply(params.actor_params, rollout.obs)
    values = critic.apply(params.critic_params, rollout.obs)
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
