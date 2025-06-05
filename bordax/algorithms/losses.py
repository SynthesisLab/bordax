from bordax.agents.base import Agent
from bordax.types import PRNGKey, Params

import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod
from typing import Mapping, Tuple, Callable, Sequence


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

        pi, info = agent.policy(params, obs)
        logp = pi.log_prob(act)

        ratio = jnp.exp(logp - old_logp)

        surrogate_loss1 = ratio * adv
        surrogate_loss2 = jnp.clip(ratio, 1 - eps_clip, 1 + eps_clip) * adv

        loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
        metrics = {
            "loss": loss,
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
        pi, info = agent.policy(params, obs)
        loss = -self.ent_coef_schedule(step) * pi.entropy().mean()
        metrics = {
            "entropy_loss": loss,
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
