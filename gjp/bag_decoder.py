from typing import Any, Sequence

import jax
from flax import linen as nn

from .model import MLP


class GraphBagDecoder(nn.module):
    max_nodes: int
    node_bag_mlp_size: Sequence[int]
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self._mlp_kwargs = self.mlp_kwargs
        if self._mlp_kwargs is None:
            self._mlp_kwargs = {}

        self.node_bag_mlp = MLP(self.node_bag_mlp_size[:-1] + (self.node_bag_mlp_size[-1] * self.max_nodes,), **self._mlp_kwargs)

        def _init_nodes(sub_x):
            # Strip off n_node, n_edges
            sub_x = sub_x[:-2]
            init_nodes = self.node_bag_mlp(sub_x)
            return init_nodes.reshape((self.max_num_nodes, self.node_bag_mlp_size[-1]))

        self.init_nodes_fn = jax.vmap(_init_nodes, in_axes=0)

    def __call__(self, x):
        """
        First dimension of x is the batch of graphs.
        Second dimension -2 is the n_node specification, and -1 is the n_edge_specification.
        """

        n_node = x[:, -2]
        n_edge = x[:, -1]

        init_nodes = self.init_nodes_fn(x)
