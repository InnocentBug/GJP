from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jraph
import numpy as np
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
    cumsum_start = jnp.concatenate([jnp.zeros(1), cumsum_end[:-1]])

    idx = jnp.vstack([cumsum_start, cumsum_end]).transpose().astype(int)

    def _scatter_nodes(sub_idx):
        new_node = jax.lax.dynamic_slice_in_dim(operand=graph.nodes, start_index=sub_idx[0], slice_size=max_nodes)

        slice_size = sub_idx[1] - sub_idx[0]
        roll_shift = (sub_idx[0] + max_nodes) - graph.nodes.shape[0]
        roll_shift = roll_shift * (roll_shift > 0).astype(int)
        roll_shift = max_nodes - roll_shift
        new_node = jnp.roll(new_node, roll_shift, axis=0)

        logic = jnp.arange(max_nodes) < slice_size
        new_node = new_node * logic[:, None]
        return new_node

    scatter_nodes = jax.vmap(_scatter_nodes, in_axes=0)
    new_nodes = scatter_nodes(idx)

    unbatch = jraph.unbatch(graph)
    unbatch_new = jraph.unbatch(new_graph)
    edge_weight_list = []

    for i in range(len(unbatch)):
        g = unbatch[i]
        ng = unbatch_new[2 * i]
        edge_weight = np.zeros(max_nodes**2 * multi_edge_repeat)
        new_edges = np.zeros((g.n_node[0] ** 2 * multi_edge_repeat,) + g.edges.shape[1:])

        sorted_senders = g.senders
        sorted_receivers = g.receivers
        sorted_edges = g.edges

        search_map = {}
        for old_idx in range(g.senders.shape[0]):
            send_idx = int(sorted_senders[old_idx])
            recv_idx = int(sorted_receivers[old_idx])
            try:
                start_search = search_map[(send_idx, recv_idx)]
            except KeyError:
                start_search = 0

            new_match_idx = None
            for new_idx in range(start_search, ng.senders.shape[0]):
                if ng.senders[new_idx] == send_idx and ng.receivers[new_idx] == recv_idx:
                    new_match_idx = new_idx
                    break

            assert new_match_idx is not None
            assert new_match_idx < g.n_node[0] ** 2 * multi_edge_repeat
            search_map[(send_idx, recv_idx)] = new_match_idx + 1

            new_edges[new_match_idx] = sorted_edges[old_idx]
            edge_weight[new_match_idx] += 1

        unbatch_new[2 * i] = ng._replace(edges=jnp.asarray(new_edges))
        unbatch_new[2 * i + 1] = unbatch_new[2 * i + 1]._replace(edges=jnp.zeros((unbatch_new[2 * i + 1].n_edge[0],) + g.edges.shape[1:]))

        edge_weight_list.append(edge_weight)

    new_graph = jraph.batch(unbatch_new)

    new_graph = new_graph._replace(nodes=new_nodes.reshape((new_nodes.shape[0] * new_nodes.shape[1], new_nodes.shape[2])), globals=jnp.repeat(graph.globals, 2, axis=0))

    return new_graph, jnp.concatenate(edge_weight_list)


def make_graph_sparse(graph, edge_weights):
    max_node = graph.n_node[0] + graph.n_node[1]
    num_graphs = graph.n_node.shape[0] // 2
    unbatch = jraph.unbatch(graph)
    new_graphs = []

    edge_weights = edge_weights.reshape((num_graphs, edge_weights.shape[0] // num_graphs))
    for i in range(num_graphs):
        old_graph = unbatch[2 * i]
        logic = jnp.nonzero(edge_weights[i])[0]
        new_graph = jraph.GraphsTuple(
            nodes=old_graph.nodes, edges=old_graph.edges[logic], senders=old_graph.senders[logic], receivers=old_graph.receivers[logic], n_node=old_graph.n_node, n_edge=jnp.asarray([jnp.sum(edge_weights[i], dtype=int)]), globals=old_graph.globals
        )
        new_graphs.append(new_graph)
    return jraph.batch(new_graphs)
