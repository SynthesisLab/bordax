import numpy as np
from typing import Dict, Tuple

class ReplayBuffer:
    """
    A simple ring buffer for storing and sampling transitions for off-policy RL.
    This implementation is based on NumPy and is not designed to be JAX-jittable.
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]):
        """
        Initializes the replay buffer.

        Args:
            capacity: The maximum number of transitions to store.
            obs_shape: The shape of a single observation.
            action_shape: The shape of a single action.
        """
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self.observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

        self._ptr = 0
        self._size = 0

    def add(self, rollout: Dict[str, np.ndarray]):
        """
        Adds a batch of transitions to the buffer.
        The input arrays in the rollout dictionary are expected to have the same leading dimension.
        Required keys: 'obs', 'action', 'reward', 'next_obs', 'done'.
        """
        num_transitions = rollout['obs'].shape[0]
        indices = np.arange(self._ptr, self._ptr + num_transitions) % self.capacity

        self.observations[indices] = rollout['obs']
        self.actions[indices] = rollout['action']
        self.rewards[indices] = rollout['reward']
        self.next_observations[indices] = rollout['next_obs']
        self.dones[indices] = rollout['done']

        self._ptr = (self._ptr + num_transitions) % self.capacity
        self._size = min(self._size + num_transitions, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Samples a batch of transitions from the buffer.

        Args:
            batch_size: The number of transitions to sample.

        Returns:
            A dictionary containing the sampled transitions.
        """
        if self._size < batch_size:
            raise ValueError(f"Not enough samples in the buffer to sample {batch_size} transitions. "
                             f"Current size: {self._size}")

        indices = np.random.randint(0, self._size, size=batch_size)
        return {
            'obs': self.observations[indices],
            'action': self.actions[indices],
            'reward': self.rewards[indices],
            'next_obs': self.next_observations[indices],
            'done': self.dones[indices],
        }

    def __len__(self) -> int:
        return self._size