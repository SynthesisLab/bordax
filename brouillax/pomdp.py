import gymnax
import gymnax.environments
import gymnax.environments.spaces
import jax
import jax.numpy as jnp
from brouillax.utils import POMDP as prePOMDP
from brouillax.parser import parse
import distrax

from flax import struct

from typing import Optional

from functools import partial
import functools

@struct.dataclass
class EnvPOMDPState(gymnax.environments.EnvState):
    state: int

@struct.dataclass
class EnvBeliefState(gymnax.environments.EnvState):
    belief: jnp.ndarray
    state: EnvPOMDPState

@struct.dataclass
class EnvParams(gymnax.environments.EnvParams):
    max_steps_in_episode: int = 10


class POMDP(gymnax.environments.environment.Environment):

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
    
    @property
    def name(self) -> str:
        return f"brouillax/{self._name}-obs"

    @property
    def num_action(self) -> int:
        return self.n_actions

    def action_space(self, params: Optional[EnvParams] = None):
        return gymnax.environments.spaces.Discrete(self.n_actions)
    
    def observation_space(self, params: EnvParams):
        return gymnax.environments.spaces.Discrete(self.n_obs)
    
    def state_space(self, params: EnvParams):
        return gymnax.environments.spaces.Discrete(self.n_states)

    def __init__(self, pomdp: prePOMDP, env_name: str):
        self._pomdp = pomdp
        self._name = env_name

        self.states = self._pomdp.states
        self.actions = self._pomdp.actions
        self.obs = self._pomdp.obs

        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.n_obs = len(self.obs)

        # self.action_space = gymnax.environments.spaces.Discrete(self.n_actions)
        # self.observation_space = gymnax.environments.spaces.Discrete(self.n_obs)

        self.state_to_index = {}
        self.action_to_index = {}
        self.observation_to_index = {}
        self.state_to_index = self._pomdp.statesinv
        self.action_to_index = self._pomdp.actionsinv
        self.observation_to_index = self._pomdp.obsinv

        # Convert start probabilities (state)
        self.start_probs = jnp.zeros((self.n_states))
        for s, p in self._pomdp.start.items():
            self.start_probs = self.start_probs.at[s].set(p)

        # Convert transition probabilities (state, action, state')

        self.transitions = jnp.zeros((self.n_states, self.n_actions, self.n_states))
        for s0, actions in self._pomdp.trans.items():
            for a, states in actions.items():
                for s1, p in states.items():
                    self.transitions = self.transitions.at[s0, a, s1].set(p)

        # Convert observation function  (action, state', obs)
        self.obs_fun = jnp.zeros((self.n_actions, self.n_states, self.n_obs))
        for a_idx in self.action_to_index.values():
            for s_idx in self.state_to_index.values():
                for o, p in self._pomdp.obsfun[a_idx][s_idx].items():
                    self.obs_fun = self.obs_fun.at[a_idx, s_idx, o].set(p)

        # Convert rewards (action, )
        self.reward = jnp.zeros(
            (self.n_actions, self.n_states, self.n_states, self.n_obs)
        )
        for (a_idx, s_idx, s0_idx, obs_idx), reward in self._pomdp.reward.items():
            self.reward = self.reward.at[a_idx, s_idx, s0_idx, obs_idx].set(reward)

    def reset_env(self, key, params):
        distr = distrax.Categorical(probs=self.start_probs)
        state_key, obs_key = jax.random.split(key)
        state = distr.sample(seed=state_key)

        distr = distrax.Categorical(probs=self.obs_fun[0, state])
        obs = distr.sample(seed=obs_key)

        return obs, EnvPOMDPState(0, state)

    def step_env(self, key, state: EnvPOMDPState, action, params):
        distr = distrax.Categorical(probs=self.transitions[state.state][action])

        state_key, obs_key = jax.random.split(key)

        new_state = distr.sample(seed=state_key)

        distr = distrax.Categorical(probs=self.obs_fun[action][new_state])
        obs = distr.sample(seed=obs_key)

        reward = self.reward[action][state.state][new_state][obs]

        new_state = EnvPOMDPState(state.time + 1, new_state)

        done = self.is_terminal(new_state, params)

        return obs, new_state, reward, done, {}

    @partial(jax.jit, static_argnums=(0,))
    def is_terminal(self, state: EnvPOMDPState, params: EnvParams):
        done_steps = state.time >= params.max_steps_in_episode
        return done_steps


class BeliefPOMDP(gymnax.environments.environment.Environment):

    # state of a belief wrapper is
    # (belief, (state, counter))

    @property
    def default_params(self):
        return EnvParams()

    @property
    def name(self) -> str:
        return f"broullax/{self._env.name}-belief"

    def observation_space(self, params: gymnax.EnvParams):
        return gymnax.environments.spaces.Box(
            low=0, high=1, shape=(self._env.n_states,), dtype=jnp.float32
        )

    def num_actions(self):
        return self._env.num_actions

    def action_space(self, params: Optional[gymnax.EnvParams] = None):
        return self._env.action_space()

    def state_space(self, params: Optional[gymnax.EnvParams] = None):
        if params is None:
            params = self.default_params
        # Ensure params is of type EnvParams from this module
        if not isinstance(params, EnvParams):
            params = EnvParams(**vars(params))
        return self._env.state_space(params)

    def __init__(self, env: POMDP):
        self._env = env

        self.start_belief = jnp.zeros((self._env.n_states))
        for s in self._env.state_to_index.values():
            self.start_belief = self.start_belief.at[s].set(self._env.start_probs[s])

        # action, observation, initial state, final state
        self.belief_matrix = jnp.zeros(
            (
                self._env.n_actions,
                self._env.n_obs,
                self._env.n_states,
                self._env.n_states,
            )
        )
        for a_idx in self._env.action_to_index.values():
            for o_idx in self._env.observation_to_index.values():
                # eta is just a normalization, so it can be applied at the computation time
                for sp in self._env.state_to_index.values():
                    for s in self._env.state_to_index.values():
                        self.belief_matrix = self.belief_matrix.at[
                            a_idx, o_idx, sp, s
                        ].set(
                            self._env.obs_fun[a_idx, sp, o_idx]
                            * self._env.transitions[s, a_idx, sp]
                        )

    def reset_env(self, key, params: EnvParams):
        _, state = self._env.reset(key, self.default_params)
        current_belief = self.start_belief

        return current_belief, EnvBeliefState(time = 0, belief=current_belief, state=state)

    def step_env(self, key, state: EnvBeliefState, action, params: EnvParams):
        obs, new_state, reward, done, _ = self._env.step(
            key, state.state, action, self.default_params
        )
        new_belief = self._belief(obs, action, state.belief)
        return new_belief, EnvBeliefState(new_state.time, new_belief, new_state), reward, done, {"signal": obs}

    @functools.partial(jax.jit, static_argnums=(0,))
    def _belief(self, obs, action, current_belief):
        belief = self.belief_matrix[action, obs] @ current_belief
        return jnp.divide(belief, jnp.linalg.norm(belief, ord=1))


def make(env_name: str):
    with open(f"environments/{env_name}.POMDP", "r") as f:
        prepomdp = parse(f.read())

    env = BeliefPOMDP(POMDP(prepomdp, env_name))
    return env, env.default_params
