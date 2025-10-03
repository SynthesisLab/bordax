import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from typing import List


class MLP(nn.Module):
    layer_sizes: List[int]
    """Simple fully-connected MLP used for policy/value heads.

    The module constructs a sequence of Dense layers using `layer_sizes`.
    The final layer is returned without an activation.
    """

    def setup(self):
        self.dense_layers = [
            nn.Dense(size, kernel_init=nn.initializers.orthogonal())
            for size in self.layer_sizes
        ]

    def __call__(self, x):
        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        return self.dense_layers[-1](x)


class MLP_dtsemnet(nn.Module):
    tree_depth: int
    action_dim: int
    """A decision-tree-inspired dense module.

    This module builds an internal representation derived from a binary
    tree structure of depth `tree_depth` and maps inputs to `action_dim`
    outputs. It is an experimental architecture used as an alternative
    policy head.
    """

    def setup(self):
        self.weights = nn.Dense(
            (2 ** (self.tree_depth) - 1),
            kernel_init=nn.initializers.orthogonal(),
            bias_init=nn.initializers.uniform(),
        )

    def __call__(self, x):
        """Compute the forward pass for the tree-based representation.

        The implementation supports both single-example inputs (1D) and
        batched inputs (2D). Returns an array shaped (batch, action_dim).
        """

        if len(x.shape) == 1:
            x = jnp.array([x])

        x = self.weights(x)

        n_nodes = 2 ** (self.tree_depth) - 1
        n_leaves = n_nodes + 1

        row_indices = jnp.arange(2 * n_nodes)
        col_indices = jnp.arange(n_nodes).repeat(2)
        tiles = jnp.tile(jnp.array([1.0, -1.0]), n_nodes)
        matrix = jnp.zeros((2 * n_nodes, n_nodes), dtype=jnp.float32)
        matrix = matrix.at[row_indices, col_indices].set(tiles)

        x = nn.relu(x @ matrix.T)

        tree_representation = jnp.ones((n_leaves, 2 * n_nodes))
        for i in range(n_leaves):
            virtual_index = i + n_nodes
            relevant_indices = jnp.zeros(self.tree_depth - 1)
            replacement = jnp.ones(2 * n_nodes)
            for j in range(self.tree_depth):
                new_virtual_index = (virtual_index - 1) // 2
                relevant_indices = relevant_indices.at[self.tree_depth - j].set(
                    new_virtual_index
                )
                if virtual_index % 2 == 0:
                    replacement_tile = jnp.array([0, 1])
                else:
                    replacement_tile = jnp.array([1, 0])
                virtual_index = new_virtual_index
                replacement = replacement.at[
                    2 * virtual_index : 2 * virtual_index + 2
                ].set(replacement_tile)
            tree_representation = tree_representation.at[i].set(replacement)

        appendice = jnp.zeros(
            ((self.action_dim - (n_leaves % self.action_dim)), 2 * n_nodes)
        )
        tree_representation = jnp.concatenate((tree_representation, appendice), axis=0)

        x = x @ tree_representation.T

        x = x.reshape((x.shape[0], -1, self.action_dim))
        x = x.max(axis=1)

        return x


class MLP_boolean(nn.Module):
    n: int
    action_dim: int
    """Boolean-function-inspired dense module.

    The module constructs a mapping from inputs to outputs by interpreting
    the learned dense layer as coefficients over the truth table of all
    boolean functions with `n` inputs. The outputs are reduced per
    `action_dim` using a max operation.
    """

    def setup(self):
        self.weights = nn.Dense(
            self.n,
            kernel_init=nn.initializers.orthogonal(),
            bias_init=nn.initializers.uniform(),
        )

    def __call__(self, x):

        if len(x.shape) == 1:
            x = jnp.array([x])

        x = self.weights(x)

        numbers = np.arange(2**self.n)

        binary_strings = [np.binary_repr(num, width=self.n) for num in numbers]

        function_representation = np.array(
            [[1 if char == "1" else -1 for char in binary] for binary in binary_strings]
        )
        function_representation = jnp.array(function_representation)

        x = x @ function_representation.T

        x = x.reshape((x.shape[0], -1, self.action_dim))
        x = x.max(axis=1)

        return x
