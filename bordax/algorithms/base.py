import jax
import optax


from typing import Any, NamedTuple
import functools

from bordax.agents.base import Agent
from bordax.environments.utils import EnvAdapter, EnvState, EnvObs
from bordax.batchbuilders import BatchBuilder
from bordax.types import PRNGKey, TrainingState
from bordax.collectors import Collector
from bordax.updaters import Updater

from bordax.algorithms.losses import PPOLoss
from bordax.batchbuilders import FullBufferBatch, MiniBatch, ComposedBatchBuilder
from bordax.collectors import (
    OnPolicyCollector,
    EpsGreedyCollector,
    StochasticOffPolicyCollector,
    DeterministicOffPolicyCollector,
)
from bordax.updaters import SGDUpdate

# The algorithm, e.g., PPO or VPG or DQN
# The algorithms vary in how they
# - collect the rollout (mainly on-policy or off-policy) and add it to the replay buffer
# - form batches from the buffer
# - update the parameters of the model (do they use the loss? what is the optimizer if yes?)


class Algorithm(NamedTuple):
    collector: Collector
    batch_builder: BatchBuilder
    updater: Updater

    def init_training_state(
        self, agent: Agent, key: PRNGKey, sample_obs: Any, env: EnvAdapter
    ) -> TrainingState:

        params = agent.init(key, sample_obs, env.action_space())
        return self.updater.init(params)

    def collect(
        self,
        key: PRNGKey,
        env: EnvAdapter,
        obs: EnvObs,
        env_state: EnvState,
        agent: Agent,
        ts: TrainingState,
    ):
        return self.collector(key, env, obs, env_state, agent, ts.params)

    @functools.partial(
        jax.jit, static_argnames=("self", "agent"), donate_argnames=("batch")
    )
    def update(self, agent: Agent, batch: Any, ts: TrainingState, key: PRNGKey):
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
        buffer: Any,
        obs: EnvObs,
        env_state: EnvState,
    ):
        key, collect_key, batch_key, update_key = jax.random.split(key, 4)

        (obs, env_state), buffer = self.collect(
            collect_key, env, obs, env_state, agent, ts
        )

        buffer["advantages"] = (buffer["advantages"] - buffer["advantages"].mean()) / (
            buffer["advantages"].std() + 1e-8
        ) # TODO: make normalization (and the calculation) optional

        batch = self.batch_builder(batch_key, buffer)
        ts, metrics = self.update(agent, batch, ts, update_key)

        return (key, ts, None, obs, env_state), metrics


def ppo_algo(
    rollout_length: int = 1024,
    gamma: float = 0.99,
    _lambda: float = 0.85,
    lr: float = 0.001,
    clip_schedule=lambda _: 0.2,
    vf_schedule=lambda _: 0.5,
    ent_schedule=lambda _: 0.01,
    num_minibatches=16,
    num_sdg_steps=5,
    **kwargs
):

    assert (
        rollout_length % num_minibatches == 0
    ), "Rollout length must be divisible by number of minibatches"
    return Algorithm(
        OnPolicyCollector(rollout_length, gamma, _lambda),
        ComposedBatchBuilder(
            (
                FullBufferBatch(rollout_length, 1),
                MiniBatch(rollout_length // num_minibatches),
            ),
        ),
        SGDUpdate(
            optimizer=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr)),
            loss_fn=PPOLoss(clip_schedule, ent_schedule, vf_schedule),
            num_sdg_steps=num_sdg_steps,
        ),
    )


# a2c_algo = Algorithm(
#     OnPolicyCollector(roullout_length=5, gamma=0.99),
#     FullBufferBatch(),
#     A2CLoss(vf_coef=0.5, ent_coef=0.01),
#     AdamUpdater(3e-4),
# )

# dqn_algo = Algorithm(
#     EpsGreedyCollector(epsilon=0.1, gamma=0.99),
#     UniformReplayBatch(32),
#     DQNLoss(n_step=1, delta=1.0),
#     TargetNetUpdater(AdamUpdater(1e-4), tau=None, update_interval=1000),  # hard update
# )

# ddqn_algo = Algorithm(
#     EpsGreedyCollector(epsilon=0.1, gamma=0.99),
#     UniformReplayBatch(32),
#     DoubleDQNLoss(n_step=1),
#     TargetNetUpdater(AdamUpdater(1e-4), update_interval=1000),
# )

# sac_algo = Algorithm(
#     StochasticOffPolicyCollector(temp=1.0, gamma=0.99),
#     UniformReplayBatch(256),
#     SACLoss(alpha_autotune=True, target_entropy=-1.0),
#     SACUpdater(actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, tau=0.005),
# )

# td3_algo = Algorithm(
#     DeterministicOffPolicyCollector(noise_std=0.1, noise_clip=0.5),
#     UniformReplayBatch(256),
#     TD3Loss(target_noise=0.2, target_clip=0.5, policy_delay=2),
#     PolyakUpdater(actor_lr=3e-4, critic_lr=3e-4, tau=0.005),
# )
