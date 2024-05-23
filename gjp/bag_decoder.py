from typing import Any, Sequence

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn

from .model import MLP
from .mpg import MessagePassingGraph


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
            senders = senders * ~logic + (n_node) * logic
            receivers = receivers * ~logic + (n_node) * logic

            # Sort invalid edges to the end
            sort_idx = jnp.argsort(senders + receivers, stable=True)
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
            node_stack=self.message_passing_stack + ((1,),), edge_stack=self.message_passing_stack + ((1,),), attention_stack=self.message_passing_stack + ((1,),), global_stack=None, mean_aggregate=True, mlp_kwargs=self.mlp_kwargs
        )

    def __call__(self, x):
        n_node = x[:, -2]

        n_edge_max: int = int(self.max_nodes) ** 2 * int(self.multi_edge_repeat)
        n_edge_tmp = (n_node) ** 2 * int(self.multi_edge_repeat)
        new_n_edge = jnp.vstack((n_edge_tmp, n_edge_max - n_edge_tmp)).flatten(order="F")

        node_features = self.init_nodes(x)
        node_features = node_features.reshape((node_features.shape[0] * node_features.shape[1], node_features.shape[2]))
        senders, receivers, edge_features = self.init_edges(x)

        new_n_node = jnp.vstack((n_node, self.max_nodes - n_node)).flatten(order="F")
        initial_graph = jraph.GraphsTuple(
            nodes=node_features,
            edges=edge_features,
            senders=senders.astype(int),
            receivers=receivers.astype(int),
            n_node=new_n_node.astype(int),
            n_edge=new_n_edge.astype(int),
            globals=jnp.repeat(x, 2, axis=0),
        )

        out_graph = self.init_mpg(initial_graph)

        return out_graph


class GraphBagDecoder(nn.Module):
    max_nodes: int
    init_node_stack: Sequence[int]
    init_edge_stack: Sequence[int]
    init_mpg_stack: Sequence[Sequence[int]]
    final_mpg_stack: Sequence[Sequence[int]]
    multi_edge_repeat: int = 1
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self.max_edges = self.max_nodes**2 * self.multi_edge_repeat

        self.init_graph_decoder = InitialGraphBagDecoder(
            max_nodes=self.max_nodes,
            init_node_stack=self.init_node_stack,
            init_edge_stack=self.init_edge_stack,
            message_passing_stack=self.init_mpg_stack,
            multi_edge_repeat=self.multi_edge_repeat,
            mlp_kwargs=self.mlp_kwargs,
        )

        def _reduce_graph(nodes, edges, senders, receivers, n_node, double_n_edge, n_edge):
            new_nodes = jax.nn.softmax(nodes) * (jnp.arange(self.max_nodes) < n_node[0]).astype(int)
            new_nodes = new_nodes / jnp.sum(new_nodes)
            new_edges = jax.nn.softmax(edges) * (jnp.arange(self.max_edges) < double_n_edge[0]).astype(int)
            new_edges = new_edges / jnp.sum(new_edges)

            edge_sort_idx = jnp.argsort(new_edges, stable=True)

            new_edges = new_edges[edge_sort_idx]

            new_senders = senders[edge_sort_idx]
            new_receivers = receivers[edge_sort_idx]

            edge_logic = jnp.arange(self.max_edges) < n_edge
            new_senders = new_senders * edge_logic + ~edge_logic * (jnp.ones(self.max_edges) * n_node[0])
            new_receivers = new_receivers * edge_logic + ~edge_logic * (jnp.ones(self.max_edges) * n_node[0])

            new_n_edge = jnp.asarray((n_edge, self.max_edges - n_edge))

            return new_nodes, new_edges, new_senders.astype(int), new_receivers.astype(int), new_n_edge.astype(int)

        self.reduce_graph = jax.vmap(_reduce_graph)

        self.final_mpg = MessagePassingGraph(node_stack=self.final_mpg_stack, edge_stack=self.final_mpg_stack, attention_stack=self.final_mpg_stack, global_stack=None, mean_aggregate=True, mlp_kwargs=self.mlp_kwargs)

    def __call__(self, x):
        n_node = x[:, -2]
        n_edge = x[:, -1]

        tmp_graphs = self.init_graph_decoder(x)

        # We know last dimensions of edges and nodes is 1
        graph_nodes = tmp_graphs.nodes.reshape(x.shape[0], self.max_nodes)
        graph_edges = tmp_graphs.edges.reshape(x.shape[0], self.max_edges)

        graph_senders = tmp_graphs.senders.reshape(x.shape[0], self.max_edges)
        graph_receivers = tmp_graphs.receivers.reshape(x.shape[0], self.max_edges)
        double_n_node = tmp_graphs.n_node.reshape(x.shape[0], 2)
        double_n_edge = tmp_graphs.n_edge.reshape(x.shape[0], 2)

        reduced_nodes, reduced_edges, reduced_senders, reduced_receivers, reduced_n_edge = self.reduce_graph(graph_nodes, graph_edges, graph_senders, graph_receivers, double_n_node, double_n_edge, n_edge)

        tmp_graphs._replace(nodes=reduced_nodes.reshape(reduced_nodes.size, 1), edges=reduced_edges.reshape(reduced_edges.size, 1), senders=reduced_senders.flatten(), receivers=reduced_receivers.flatten(), n_edge=reduced_n_edge.flatten())

        final_graphs = self.final_mpg(tmp_graphs)
        return final_graphs
