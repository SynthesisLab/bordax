import jax
import optax


from typing import Any, Callable, NamedTuple
import functools

from bordax.agents.base import Agent
from bordax.environments.utils import EnvAdapter, EnvState, EnvObs
from bordax.data.batchbuilders import BatchBuilder
from bordax.types import PRNGKey, TrainingState
from bordax.data.collectors import Collector
from bordax.training.updaters import Updater

from bordax.data.batchbuilders import UniformReplayBatch
from bordax.training.updaters import DQNUpdater
from bordax.algorithms.losses import DQNLoss

from bordax.algorithms.losses import PPOLoss
from bordax.data.batchbuilders import (
    FullBufferBatch,
    MiniBatch,
    NormalizeAdvantagesTargets,
    ComposedBatchBuilder,
)
from bordax.data.collectors import (
    OnPolicyCollector,
    EpsGreedyCollector,
)
from bordax.training.updaters import SGDUpdate

# The algorithm, e.g., PPO or VPG or DQN
# The algorithms vary in how they
# - collect the rollout (mainly on-policy or off-policy) and add it to the replay buffer
# - form batches from the buffer
# - update the parameters of the model (do they use the loss? what is the optimizer if yes?)


class Algorithm(NamedTuple):
    """A training algorithm composed of a collector, batch builder, and updater.

    Attributes:
        collector: Generates transitions by interacting with the environment.
        batch_builder: Transforms collected data into training batches.
        updater: Applies gradient updates to the network parameters.
    """

    collector: Collector
    batch_builder: BatchBuilder
    updater: Updater

    def init_training_state(
        self, agent: Agent, key: PRNGKey, sample_obs: Any, env: EnvAdapter
    ) -> TrainingState:
        """Initialise the training state for a given agent.

        Args:
            agent: The agent whose parameters are initialised.
            key: JAX random key.
            sample_obs: A sample observation used to infer network input shapes.
            env: The training environment (used by some updaters).

        Returns:
            A ``TrainingState`` containing initial parameters and optimizer state.
        """
        params = agent.init(key, sample_obs)
        return self.updater.init(params)

    def collect(
        self,
        key: PRNGKey,
        env: EnvAdapter,
        obs: EnvObs,
        env_state: EnvState,
        replay_buffer: Any,
        agent: Agent,
        ts: TrainingState,
    ):
        """Collect experience from the environment.

        Delegates to ``self.collector``. For on-policy algorithms the
        returned buffer contains the freshly collected rollout; for
        off-policy algorithms transitions are appended to the existing
        replay buffer which is returned.

        Returns:
            Tuple of ``((obs, env_state), replay_buffer)``.
        """
        return self.collector(key, env, obs, env_state, replay_buffer, agent, ts)

    @functools.partial(jax.jit, static_argnames=("self", "agent"))
    def update(self, agent: Agent, batch: Any, ts: TrainingState, key: PRNGKey):
        """JIT-compiled parameter update step.

        Args:
            agent: Agent providing loss function access.
            batch: Training batch produced by the batch builder.
            ts: Current training state.
            key: JAX random key.

        Returns:
            Tuple of ``(new_training_state, metrics_dict)``.
        """
        return self.updater(
            agent,
            batch,
            ts,
            key,
        )

    def train_step(
        self,
        env: EnvAdapter,
        agent: Agent,
        key: PRNGKey,
        ts: TrainingState,
        replay_buffer: Any,
        obs: EnvObs,
        env_state: EnvState,
    ):
        """Run one full training iteration: collect → batch → update.

        This method is JIT-compiled by the ``Trainer`` when the environment
        is jittable and the algorithm is on-policy.

        Returns:
            Tuple of ``((key, ts, replay_buffer, obs, env_state), metrics)``.
        """
        key, collect_key, batch_key, update_key = jax.random.split(key, 4)

        (obs, env_state), replay_buffer = self.collect(
            collect_key, env, obs, env_state, replay_buffer, agent, ts
        ) 
        # the collector also updates the replay buffer:
        # for on-policy, it returns the new buffer with the collected rollout; 
        # for off-policy, it adds the new transitions to the existing buffer

        batch = self.batch_builder(batch_key, replay_buffer)
        ts, metrics = self.update(agent, batch, ts, update_key)

        return (key, ts, replay_buffer, obs, env_state), metrics


def ppo_algo(
    rollout_length: int = 1024,
    gamma: float = 0.99,
    _lambda: float = 0.85,
    lr: float = 0.001,
    clip_schedule=lambda _: 0.2,
    vf_schedule=lambda _: 0.5,
    ent_schedule=lambda _: 0.01,
    num_minibatches=16,
    num_sgd_steps=1,
    num_envs: int = 1,
    grad_clip: float = 0.5,
    **kwargs
):
    """Create a PPO algorithm.

    Args:
        rollout_length: Number of environment steps collected per epoch
            per environment. Must be divisible by ``num_minibatches``.
        gamma: Discount factor for returns.
        _lambda: GAE lambda for advantage estimation.
        lr: Adam learning rate.
        clip_schedule: Callable ``(step) -> clip_ratio``. Defaults to
            constant 0.2.
        vf_schedule: Callable ``(step) -> vf_coef``. Defaults to 0.5.
        ent_schedule: Callable ``(step) -> ent_coef``. Defaults to 0.01.
        num_minibatches: Number of minibatches to split each rollout into.
        num_sgd_steps: Number of SGD passes over the data per epoch.
        num_envs: Number of parallel environments (used for batch reshaping).
        grad_clip: Global gradient norm clipping threshold.

    Returns:
        A configured ``Algorithm`` for PPO.
    """

    assert (
        rollout_length % num_minibatches == 0
    ), "Rollout length must be divisible by number of minibatches"

    schedule = optax.constant_schedule(lr)
    adam = optax.inject_hyperparams(optax.adam)(learning_rate=schedule)
    optimizier = optax.chain(optax.clip_by_global_norm(grad_clip), adam)    

    return Algorithm(
        OnPolicyCollector(rollout_length, gamma, _lambda),
        ComposedBatchBuilder(
            (
                FullBufferBatch(rollout_length, num_envs),
                MiniBatch(num_minibatches),
                NormalizeAdvantagesTargets(normalize_targets=False),
            ),
        ),
        SGDUpdate(
            optimizer=optimizier,
            loss_fn=PPOLoss(
                clip_schedule=clip_schedule,
                vf_coef_schedule=vf_schedule,
                ent_coef_schedule=ent_schedule,
            ),
            num_sgd_steps=num_sgd_steps,
            grad_clip=grad_clip,
        ),
    )

def dqn_algo(
    epsilon_schedule: Callable[[int], float] = lambda t: 0.1,
    rollout_length: int = 1,
    batch_size: int = 32,
    gamma: float = 0.99,
    lr: float = 1e-4,
    target_update_freq: int = 1000,
    applied_loss: Callable = optax.squared_error,
    **kwargs
):
    """Create a DQN algorithm.

    Args:
        epsilon_schedule: Callable ``(step) -> epsilon`` controlling the
            exploration rate over time.
        rollout_length: Number of environment steps collected per update.
            Typically 1 for standard DQN.
        batch_size: Number of transitions sampled from the replay buffer
            per update.
        gamma: Discount factor for Bellman targets.
        lr: Adam learning rate for the Q-network.
        target_update_freq: Number of training steps between target network
            hard updates.
        applied_loss: Element-wise loss applied to TD errors (e.g.
            ``optax.squared_error`` or ``optax.huber_loss``).

    Returns:
        A configured ``Algorithm`` for DQN.
    """
    
    return Algorithm(
        EpsGreedyCollector(epsilon_schedule=epsilon_schedule, rollout_length=rollout_length),
        UniformReplayBatch(batch_size),
        DQNUpdater(
            optimizer=optax.adam(lr),
            loss_fn=DQNLoss(gamma=gamma, applied_loss=applied_loss),
            target_update_freq=target_update_freq,
        ),
    )