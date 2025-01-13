import gymnax.utils
import jax
import jax.numpy as jnp
from bordax.environments.pomdp.utils import POMDP
import distrax

import gymnax

from functools import partial

class jPOMDP(gymnax.environments.environment.Environment):

    def __init__(self, pomdp: POMDP):
        self.init_pomdp = pomdp

        self.states = self.init_pomdp.states
        self.actions = self.init_pomdp.actions
        self.obs = self.init_pomdp.obs

        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.n_obs = len(self.obs)
        
        self.state_to_index = {}
        self.action_to_index = {}
        self.observation_to_index = {}
        self.state_to_index = self.init_pomdp.statesinv
        self.action_to_index = self.init_pomdp.actionsinv
        self.observation_to_index = self.init_pomdp.obsinv
        
        # Convert start probabilities (state)
        self.start_probs = jnp.zeros((self.n_states))
        for s, p in self.init_pomdp.start.items():
            self.start_probs = self.start_probs.at[s].set(p)

        # Convert transition probabilities (state, action, state')

        self.transitions = jnp.zeros((self.n_states, self.n_actions, self.n_states))
        for s0_idx in self.state_to_index.values():
            for a_idx in self.action_to_index.values():
                for s1_idx in self.state_to_index.values():
                    p = self.init_pomdp.trans[s0_idx][a_idx][s1_idx]
                    self.transitions = self.transitions.at[s0_idx, a_idx, s1_idx].set(p)

        # Convert observation function  (action, state', obs)
        self.obs_fun = jnp.zeros((self.n_actions, self.n_states, self.n_obs))
        for a_idx in self.action_to_index.values():
            for s_idx in self.state_to_index.values():
                for o, p in self.init_pomdp.obsfun[a_idx][s_idx].items():
                    self.obs_fun = self.obs_fun.at[a_idx, s_idx, o].set(p)

        # Convert rewards (action, )
        self.reward = jnp.zeros((self.n_actions, self.n_states))
        for (a_idx, s_idx), reward in self.init_pomdp.reward.items():
            self.reward = self.reward.at[a_idx, s_idx].set(reward)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params):
        distr = distrax.Categorical(probs=self.start_probs)
        state_key, obs_key = jax.random.split(key)
        state = distr.sample(seed=state_key)
        
        distr = distrax.Categorical(probs=self.obs_fun[0, state])
        obs = distr.sample(seed=obs_key)
        
        return obs, state
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params):
        distr = distrax.Categorical(probs = self.transitions[state][action])

        state_key, obs_key = jax.random.split(key)

        new_state = distr.sample(seed=state_key)

        distr = distrax.Categorical(probs = self.obs_fun[action][new_state])
        obs = distr.sample(seed=obs_key)

        reward = self.reward[(action, state)]

        done = False

        return obs, new_state, reward, done

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

class BeliefWrapper(GymnaxWrapper):
    def __init__(self, env: jPOMDP):
        super().__init__(env)
        self._env = env

        self.start_belief = jnp.zeros((self._env.n_states))
        for s in self._env.state_to_index.values():
            self.start_belief = self.start_belief.at[s].set(self._env.start_probs[s])


        # action, observation, initial state, final state
        self.belief_matrix = jnp.zeros((self._env.n_actions, self._env.n_obs, self._env.n_states, self._env.n_states))
        for a_idx in self._env.action_to_index.values():
            for o_idx in self._env.observation_to_index.values():
                # eta is just a normalization, so it can be applied at the computation time
                for sp in self._env.state_to_index.values():
                    for s in self._env.state_to_index.values():
                        self.belief_matrix = self.belief_matrix.at[a_idx, o_idx, sp, s].set(
                            self._env.obs_fun[a_idx, sp, o_idx] * self._env.transitions[s, a_idx, sp]
                        )

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params):
        obs, state = self._env.reset(key, params)
        current_belief = self.start_belief
        
        return current_belief, (current_belief, state)
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, action, params):
        current_belief, state = state
        obs, new_state, reward, done = self._env.step(key, state, action, params)
        new_belief = self._belief(obs, action, current_belief)
        return new_belief, (new_belief, new_state), reward, done, {"signal": obs}
    
    @partial(jax.jit, static_argnums=(0,))
    def _belief(self, obs, action, current_belief):
        belief = self.belief_matrix[action, obs] @ current_belief 
        return jnp.divide(belief, jnp.linalg.norm(belief, ord=1))
    