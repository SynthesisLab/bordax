import jax
import optax


from typing import Any, Callable, NamedTuple
import functools

from bordax.agents.base import Agent
from bordax.environments.utils import EnvAdapter, EnvState, EnvObs
from bordax.batchbuilders import BatchBuilder
from bordax.types import PRNGKey, TrainingState
from bordax.collectors import Collector
from bordax.updaters import Updater

from bordax.batchbuilders import UniformReplayBatch
from bordax.updaters import DQNUpdater
from bordax.algorithms.losses import DQNLoss

from bordax.algorithms.losses import PPOLoss
from bordax.batchbuilders import (
    FullBufferBatch,
    MiniBatch,
    NormalizeAdvantagesTargets,
    ComposedBatchBuilder,
)
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
        return self.collector(key, env, obs, env_state, replay_buffer, agent, ts)

    @functools.partial(jax.jit, static_argnames=("self", "agent"))
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
        replay_buffer: Any,
        obs: EnvObs,
        env_state: EnvState,
    ):
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
    **kwargs
):

    assert (
        rollout_length % num_minibatches == 0
    ), "Rollout length must be divisible by number of minibatches"
    return Algorithm(
        OnPolicyCollector(rollout_length, gamma, _lambda),
        ComposedBatchBuilder(
            (
                FullBufferBatch(rollout_length, num_envs),
                MiniBatch(num_minibatches),
                NormalizeAdvantagesTargets(),
            ),
        ),
        SGDUpdate(
            optimizer=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(lr)),
            loss_fn=PPOLoss(
                clip_schedule=clip_schedule,
                vf_coef_schedule=vf_schedule,
                ent_coef_schedule=ent_schedule,
            ),
            num_sgd_steps=num_sgd_steps,
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
        epsilon: Exploration rate for epsilon-greedy policy
        rollout_length: Number of steps to collect before updating (typically 1 for DQN)
        batch_size: Number of transitions to sample from replay buffer
        gamma: Discount factor
        lr: Learning rate for the Q-network optimizer
        target_update_freq: How often to update target network (in training steps)
        applied_loss: Loss function to use (e.g., Huber loss or MSE)
    
    Returns:
        Algorithm instance for DQN
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


# a2c_algo = Algorithm(
#     OnPolicyCollector(roullout_length=5, gamma=0.99),
#     FullBufferBatch(),
#     A2CLoss(vf_coef=0.5, ent_coef=0.01),
#     AdamUpdater(3e-4),
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
