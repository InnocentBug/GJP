from typing import Sequence, Any, NamedTuple
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn

from .model import MLP


def round_smoothing(x):
    x = jnp.abs(x)
    return x - jnp.sin(x * 2 * jnp.pi) / (4 * jnp.pi)


def indexify_graph(graph):
    graph = graph._replace(
        senders=jnp.round(graph.senders, 0).astype(int),
        receivers=jnp.round(graph.receivers, 0).astype(int),
    )
    return graph


class ReferenceGraph(NamedTuple):
    nodes: jnp.ndarray
    edges: jnp.ndarray
    senders: jnp.ndarray
    receivers: jnp.ndarray

def make_diff_graph(graph_a, graph_b, skip_graphs=1):
    result = ReferenceGraph(nodes = graph_a.nodes[:-skip_graphs] - graph_b.nodes[:-skip_graphs],
                            edges = graph_a.edges[:-skip_graphs] - graph_b.edges[:-skip_graphs],
                            senders = graph_a.senders[:-skip_graphs] - graph_b.senders[:-skip_graphs],
                            receivers = graph_a.receivers[:-skip_graphs] - graph_b.receivers[:-skip_graphs],
                            )
    return result

def make_abs_graph(graph):
    result = ReferenceGraph(nodes = jnp.abs(graph.nodes),
                            edges = jnp.abs(graph.edges),
                            senders = jnp.abs(graph.senders),
                            receivers = jnp.abs(graph.receivers),
                            )
    return result

def make_square_graph(graph):
    result = ReferenceGraph(nodes = graph.nodes**2,
                            edges = graph.edges**2,
                            senders = graph.senders**2,
                            receivers = graph.receivers**2,
                            )
    return result


def make_abs_diff_graph(graph_a, graph_b, skip_graphs=1):
    return make_abs_graph(make_diff_graph(graph_a, graph_b, skip_graphs))

def make_square_diff_graph(graph_a, graph_b, skip_graphs=1):
    return make_square_graph(make_diff_graph(graph_a, graph_b, skip_graphs))


def batch_graph_arrays(graph, max_edges:int, max_nodes:int):
    def mask_builder(n_edge, cumsum, node_cumsum):
        mask = jnp.arange(max_edges) < n_edge
        pad_widths = ((0, max(0, max_edges-graph.edges.shape[0])), (0, 0))
        pad_edges = jnp.pad(graph.edges, pad_widths)
        pad_senders = jnp.pad(graph.senders, pad_widths[0])
        pad_receivers = jnp.pad(graph.receivers, pad_widths[0])

        senders = jnp.roll(pad_senders, -cumsum)[:max_edges] - node_cumsum
        receivers = jnp.roll(pad_receivers, -cumsum)[:max_edges] - node_cumsum
        senders = mask * senders
        receivers = mask * receivers

        edges = jnp.roll(pad_edges, -cumsum, axis=0)[:max_edges]
        edges = mask[:,None] * edges

        return mask, senders, receivers, edges
    cumsum = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(graph.n_edge[:-1], dtype=int)))
    node_cumsum = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(graph.n_node[:-1], dtype=int) ))

    mask, senders, receivers, edges = jax.vmap(mask_builder)(graph.n_edge, cumsum, node_cumsum)
    fill_array = jnp.repeat(graph.n_node, max_edges).reshape((graph.n_edge.shape[0], max_edges))

    senders = senders + jnp.arange(senders.shape[0])[:, None] * int(max_nodes)
    receivers = receivers + jnp.arange(receivers.shape[0])[:, None] * int(max_nodes)
    fill_array = fill_array + jnp.arange(fill_array.shape[0])[:, None] * int(max_nodes)

    mask = mask.flatten()
    senders = senders.flatten()
    receivers = receivers.flatten()
    fill_array = fill_array.flatten()
    edges = edges.reshape((edges.shape[0]*edges.shape[1], edges.shape[2]))

    senders = senders * mask + (1-mask) * fill_array
    receivers = receivers * mask + (1-mask) * fill_array

    def node_builder(n_node, tmp_cumsum):
        mask = jnp.arange(max_nodes) < n_node
        pad_widths = ((0, max(0, max_nodes-graph.nodes.shape[0])), (0, 0))
        pad_nodes = jnp.pad(graph.nodes, pad_widths)
        nodes = jnp.roll(pad_nodes, -tmp_cumsum, axis=0)[:max_nodes]
        nodes = mask[:,None] * nodes
        return nodes
    node_cumsum = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(graph.n_node[:-1], dtype=int) ))
    nodes = jax.vmap(node_builder)(graph.n_node, node_cumsum)

    nodes = nodes.reshape((nodes.shape[0]*nodes.shape[1], nodes.shape[2]))

    ref_graph = ReferenceGraph(nodes=nodes, edges=edges, senders=senders, receivers=receivers)
    return ref_graph


class CheatDecoder(nn.Module):
    max_nodes: int
    max_edges: int
    arch_stack: Sequence[int]
    node_stack: Sequence[int]
    edge_stack: Sequence[int]
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self._mlp_kwargs = self.mlp_kwargs
        if self._mlp_kwargs is None:
            self._mlp_kwargs = {}
        self.arch_mlp = MLP(self.arch_stack + (2 * int(self.max_edges),), **self._mlp_kwargs)
        self.node_mlp = MLP(self.node_stack[:-1] + (int(self.max_nodes) * int(self.node_stack[-1]),), **self._mlp_kwargs)
        self.edge_mlp = MLP(self.edge_stack[:-1] + (int(self.max_edges) * int(self.edge_stack[-1]),), **self._mlp_kwargs)

        def _get_send_recv(x):
            n_edge = x[-1]
            n_node = x[-2]

            send_recv = self.arch_mlp(x).reshape((2, int(self.max_edges)))

            # Mask out the invalid edges
            logic = jnp.arange(self.max_edges) < n_edge
            logic = jnp.vstack([logic, logic])

            send_recv /= jnp.max(send_recv * logic) + 1e-3
            send_recv *= n_node

            send_recv = send_recv * logic + (1 - logic) * n_node

            send_recv = round_smoothing(send_recv)
            return send_recv[0, :], send_recv[1, :]

        self.get_send_recv = jax.vmap(_get_send_recv)

        def _get_nodes(x):
            n_node = x[-2]
            sub_x = x[:-2]

            nodes = self.node_mlp(sub_x).reshape((int(self.max_nodes), int(self.node_stack[-1])))
            logic = jnp.arange(self.max_nodes) < n_node
            nodes = nodes * logic[:, None]
            return nodes

        self.get_nodes = jax.vmap(_get_nodes)

        def _get_edges(x):
            n_edge = x[-1]
            sub_x = x[:-2]

            edges = self.edge_mlp(sub_x).reshape(int(self.max_edges), int(self.edge_stack[-1]))
            logic = jnp.arange(self.max_edges) < n_edge
            edges = edges * logic[:, None]

            return edges

        self.get_edges = jax.vmap(_get_edges)

        def _get_n_edge_node(x):
            n_edge = x[-1]
            n_node = x[-2]

            return jnp.asarray((n_edge, self.max_edges - n_edge)), jnp.asarray((n_node, self.max_nodes - n_node), dtype=int)

        self.get_n_edge_node = jax.vmap(_get_n_edge_node)

    def __call__(self, x):
        nodes = self.get_nodes(x)
        nodes = nodes.reshape((nodes.shape[0] * nodes.shape[1], nodes.shape[2]))
        edges = self.get_edges(x)
        edges = edges.reshape((edges.shape[0] * edges.shape[1], edges.shape[2]))
        n_edge, n_node = self.get_n_edge_node(x)
        n_edge = n_edge.flatten()
        n_node = n_node.flatten()

        senders, receivers = self.get_send_recv(x)

        senders = senders + jnp.arange(x.shape[0])[:, None] * int(self.max_nodes)
        receivers = receivers + jnp.arange(x.shape[0])[:, None] * int(self.max_nodes)

        new_globals = jnp.repeat(x, 2, axis=0)
        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            n_edge=n_edge.astype(int),
            n_node=n_node.astype(int),
            senders=senders.flatten(),
            receivers=receivers.flatten(),
            globals=new_globals,
        )
        return graph
