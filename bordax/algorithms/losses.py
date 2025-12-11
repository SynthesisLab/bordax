from bordax.agents.base import Agent, DQNAgent, DQNParameters
from bordax.types import PRNGKey, Params

import jax
import optax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from typing import Mapping, Tuple, Callable, Sequence, cast


class LossFn(ABC):
    @abstractmethod
    def __call__(
        self,
        params: Params,
        agent: Agent,
        batch: Mapping[str, jnp.ndarray],
        key: PRNGKey,
        step: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]: ...


class SurrogateLoss(LossFn):

    def __init__(self, clip_schedule: Callable[[jnp.ndarray], float]):
        self.clip_schedule = clip_schedule

    def __call__(self, params, agent, batch, key, step):
        eps_clip = self.clip_schedule(step)

        obs, act, old_logp, adv = (
            batch["obs"],
            batch["action"],
            jax.lax.stop_gradient(batch["info"]["logp"]),
            jax.lax.stop_gradient(batch["advantages"]),
        )

        pi, info = agent.policy(params, obs, key)
        logp = pi.log_prob(act)

        if act.ndim > 1:
            logp = jnp.sum(logp, axis=-1)

        ratio = jnp.exp(logp - old_logp)

        surrogate_loss1 = ratio * adv
        surrogate_loss2 = jnp.clip(ratio, 1 - eps_clip, 1 + eps_clip) * adv

        loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
        approx_kl = jnp.mean(old_logp - logp)
        metrics = {
            "loss": loss,
            "approx_kl": approx_kl,
        }
        return loss, metrics


class ValueLoss(LossFn):

    def __init__(self, vf_coef_schedule: Callable[[jnp.ndarray], float]):
        self.vf_coef_schedule = vf_coef_schedule

    def __call__(self, params, agent, batch, key, step):
        obs, targets = (
            batch["obs"],
            jax.lax.stop_gradient(batch["targets"]),
        )
        values = agent.value(params, obs)
        loss = self.vf_coef_schedule(step) * jnp.mean(jnp.square(values - targets))
        metrics = {
            "value_loss": loss,
        }
        return loss, metrics


class EntropyLoss(LossFn):

    def __init__(self, ent_coef_schedule: Callable[[jnp.ndarray], float]):
        self.ent_coef_schedule = ent_coef_schedule

    def __call__(self, params, agent, batch, key, step):
        obs = batch["obs"]
        pi, info = agent.policy(params, obs, key)
        loss = -self.ent_coef_schedule(step) * pi.entropy().mean()
        metrics = {
            "entropy_loss": loss,
        }
        return loss, metrics


class DQNLoss(LossFn):
    def __init__(self, gamma: float, applied_loss: Callable = optax.squared_error): # the canonical choice is Huber loss, but MSE can also be used
        self.gamma = gamma
        self.applied_loss = applied_loss

    def __call__(self, params: DQNParameters, agent: Agent, batch: Mapping[str, jnp.ndarray], key: PRNGKey, step: jnp.ndarray):

        assert isinstance(agent, DQNAgent), "DQN loss can only be used with a DQNAgent"
        agent = cast(DQNAgent, agent)

        obs, action, reward, next_obs, done = (
            batch["obs"],
            batch["action"],
            batch["reward"],
            batch["next_obs"],
            batch["done"],
        )

        # Calculate current Q-values using agent's q_network
        q_values = agent.q_network.apply(params.q_network, obs)
        if isinstance(q_values, tuple):
            q_values = q_values[0]
        
        # Handle both scalar and vector actions
        if action.ndim == 1:
            action = action[..., None]
        q_values_taken = jnp.take_along_axis(q_values, action, axis=-1).squeeze(-1)

        # Calculate target Q-values using agent's target_network
        target_q_values_next_state = agent.target_network.apply(params.target_network, next_obs)
        if isinstance(target_q_values_next_state, tuple):
            target_q_values_next_state = target_q_values_next_state[0]
        max_target_q_next_state = jnp.max(target_q_values_next_state, axis=-1)
        
        # Compute target for Bellman equation
        target = reward + self.gamma * max_target_q_next_state * (1 - done)

        # Compute the loss
        loss = self.applied_loss(q_values_taken, target).mean()
        
        metrics = {
            "dqn_loss": loss,
            "mean_q_value": jnp.mean(q_values_taken),
            "mean_target": jnp.mean(target),
        }
        return loss, metrics


class CombinedLoss(LossFn):
    def __init__(self, losses: Sequence[LossFn]):
        self.losses = losses

    def __call__(self, params, agent, batch, key, step):
        total_loss = 0
        metrics = {}
        for loss in self.losses:
            loss_value, loss_metrics = loss(params, agent, batch, key, step)
            total_loss += loss_value
            metrics.update({k: v for k, v in loss_metrics.items() if k not in metrics})
        metrics["total_loss"] = total_loss
        return total_loss, metrics


class PPOLoss(LossFn):
    def __init__(
        self,
        clip_schedule: Callable[[jnp.ndarray], float],
        vf_coef_schedule: Callable[[jnp.ndarray], float],
        ent_coef_schedule: Callable[[jnp.ndarray], float],
    ):
        self.loss_fn = CombinedLoss(
            (
                SurrogateLoss(clip_schedule),
                ValueLoss(vf_coef_schedule),
                EntropyLoss(ent_coef_schedule),
            )
        )

    def __call__(self, params, agent, batch, key, step):
        return self.loss_fn(params, agent, batch, key, step)
