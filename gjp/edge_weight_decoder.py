from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn


@dataclass
class FullyConnectedGraph:
    max_nodes: int
    multi_edge_repeat: int = 1

    def __post_init__(self):
        self.setup()

    def setup(self):
        self.max_edges = int(self.max_nodes) ** 2 * int(self.multi_edge_repeat)

        def _init_full_edges(sub_x):
            n_node = sub_x[-2]
            num_edges = n_node * n_node * self.multi_edge_repeat

            edge_range = jnp.arange(self.max_nodes, dtype=int)

            senders = jnp.repeat(edge_range, int(self.max_nodes))
            receivers = jnp.repeat(edge_range.reshape(1, self.max_nodes), int(self.max_nodes), axis=0).flatten()

            senders = jnp.repeat(senders, int(self.multi_edge_repeat))
            receivers = jnp.repeat(receivers, int(self.multi_edge_repeat))

            logic = jnp.logical_or(senders >= n_node, receivers >= n_node)
            senders = senders * ~logic + (n_node) * logic
            receivers = receivers * ~logic + (n_node) * logic

            # Sort invalid edges to the end
            sort_idx = jnp.argsort(senders + receivers, stable=True)
            senders = senders[sort_idx]
            receivers = receivers[sort_idx]

            return senders.flatten(), receivers.flatten(), jnp.asarray([n_node, self.max_nodes - n_node], dtype=int), jnp.asarray([num_edges, self.max_edges - num_edges])

        self.init_full_edges = jax.vmap(_init_full_edges, in_axes=0)

    def __call__(self, x):
        senders, receivers, n_node, n_edge = self.init_full_edges(x)

        senders = senders + jnp.arange(x.shape[0], dtype=int)[:, None] * int(self.max_nodes)
        receivers = receivers + jnp.arange(x.shape[0], dtype=int)[:, None] * int(self.max_nodes)

        graph = jraph.GraphsTuple(nodes=None, edges=None, senders=senders.flatten().astype(int), receivers=receivers.flatten().astype(int), n_edge=n_edge.flatten().astype(int), n_node=n_node.flatten().astype(int), globals=None)
        return graph


def make_graph_fully_connected(graph, multi_edge_repeat):
    max_nodes = jnp.max(graph.n_node)

    graph_generator = FullyConnectedGraph(max_nodes, multi_edge_repeat)
    input_data = jnp.vstack([graph.n_node, graph.n_edge], dtype=int).transpose()

    new_graph = graph_generator(input_data)

    cumsum_end = jnp.cumsum(graph.n_node, dtype=int)
    cumsum_start = jnp.concatenate([[0], cumsum_end[:-1]])

    idx = jnp.vstack([cumsum_start, cumsum_end])
    print(idx)

    return new_graph
