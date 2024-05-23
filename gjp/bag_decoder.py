from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn
from mpg import MessagePassingGraph

from .model import MLP


class InitialNodeBag(nn.Module):
    max_nodes: int
    mlp_size: Sequence[int]
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self._mlp_kwargs = self.mlp_kwargs
        if self._mlp_kwargs is None:
            self._mlp_kwargs = {}

        self.node_bag_mlp = MLP(self.mlp_size[:-1] + (self.mlp_size[-1] * self.max_nodes,), **self._mlp_kwargs)

        def _init_nodes(sub_x):
            # Strip off n_node, n_edges
            n_node = sub_x[-2]
            sub_x = sub_x[:-2]
            init_nodes = self.node_bag_mlp(sub_x)
            init_nodes = init_nodes.reshape((self.max_nodes, self.mlp_size[-1]))
            logic = jnp.arange(self.max_nodes) < n_node
            init_nodes = init_nodes * logic[:, None]
            return init_nodes

        self.init_nodes_fn = jax.vmap(_init_nodes, in_axes=0)

    def __call__(self, x):
        """
        First dimension of x is the batch of graphs.
        Second dimension -2 is the n_node specification, and -1 is the n_edge_specification.
        """

        init_nodes = self.init_nodes_fn(x)
        return init_nodes


class InitialBagEdges(nn.Module):
    max_nodes: int
    mlp_size: Sequence[int]
    multi_edge_repeat: int = 1
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self._mlp_kwargs = self.mlp_kwargs
        if self._mlp_kwargs is None:
            self._mlp_kwargs = {}

        self.init_edge_mlp = MLP(self.mlp_size[:-1] + (int(self.max_nodes) * int(self.max_nodes) * int(self.multi_edge_repeat * self.mlp_size[-1]),), **self._mlp_kwargs)

        def _init_edge_features(sub_x):
            n_node = sub_x[-2]
            sub_x = sub_x[:-2]

            features = self.init_edge_mlp(sub_x).reshape(int(self.max_nodes) * int(self.max_nodes) * int(self.multi_edge_repeat), self.mlp_size[-1])
            return features

        self.init_edge_features = jax.vmap(_init_edge_features)

        def _init_full_edges(sub_x):
            n_node = sub_x[-2]
            edge_range = jnp.arange(self.max_nodes, dtype=int)

            senders = jnp.repeat(edge_range, self.max_nodes)
            receivers = jnp.repeat(edge_range.reshape(1, self.max_nodes), self.max_nodes, axis=0).flatten()

            senders = jnp.repeat(senders, self.multi_edge_repeat)
            receivers = jnp.repeat(receivers, self.multi_edge_repeat)

            logic = jnp.logical_or(senders >= n_node, receivers >= n_node)
            senders = senders * ~logic + (self.max_nodes - 1) * logic
            receivers = receivers * ~logic + (self.max_nodes - 1) * logic

            # Sort invalid edges to the end
            sort_idx = jnp.argsort(senders + receivers)
            senders = senders[sort_idx]
            receivers = receivers[sort_idx]

            return senders.flatten(), receivers.flatten()

        self.init_full_edges = jax.vmap(_init_full_edges)

    def __call__(self, x):
        features = self.init_edge_features(x)
        senders, receivers = self.init_full_edges(x)

        # Shift senders and receivers, such that they work on the batched sub-graphs
        senders = senders + jnp.arange(x.shape[0], dtype=int)[:, None] * int(self.max_nodes)
        receivers = receivers + jnp.arange(x.shape[0], dtype=int)[:, None] * int(self.max_nodes)

        return senders.flatten(), receivers.flatten(), features.reshape((x.shape[0] * int(self.max_nodes) ** 2 * int(self.multi_edge_repeat), self.mlp_size[-1]))


class InitialGraphBagDecoder(nn.Module):
    max_nodes: int
    init_node_stack: Sequence[int]
    init_edge_stack: Sequence[int]
    message_passing_stack: Sequence[Sequence[int]]
    multi_edge_repeat: int = 1
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self.init_edges = InitialBagEdges(self.max_nodes, self.init_edge_stack, self.multi_edge_repeat, self.mlp_kwargs)
        self.init_nodes = InitialNodeBag(self.max_nodes, self.init_node_stack, self.mlp_kwargs)

        self.init_mpg = MessagePassingGraph(
            node_stack=self.message_passing_stack + ((1,),), edge_stack=self.message_passing_stakc + ((1,),), attention_stack=self.message_passing_stakc + ((1,),), global_stack=None, mean_aggregate=True, mlp_kwargs=self.mlp_kwargs
        )

    def __call__(self, x):
        n_node = x[:, -2]
        n_node = jnp.vstack((n_node, self.max_nodes - n_node)).flatten(order="F")

        n_edge_max: int = int(self.max_nodes) ** 2 * int(self.multi_edge_repeat)
        n_edge = x[:, -1]
        n_edge = jnp.vstack((n_edge, n_edge_max - n_edge)).flatten(order="F")

        node_features = self.init_nodes(x)
        senders, receivers, edge_features = self.init_edges(x)

        initial_graph = jraph.GraphsTuple(
            node=node_features,
            edges=edge_features,
            senders=senders,
            receivers=receivers,
            n_node=n_node,
            n_edge=n_edge,
            globals=x,
        )
        return initial_graph
