from typing import Sequence

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


class CheatDecoder(nn.Module):
    max_nodes: int
    max_edges: int
    arch_stack: Sequence[int]
    node_stack: Sequence[int]
    edge_stack: Sequence[int]

    def setup(self):
        self.arch_mlp = MLP(self.arch_stack + (2 * self.max_edges,))
        self.node_mlp = MLP(self.node_stack[:-1] + (self.max_nodes * self.node_stack[-1],))
        self.edge_mlp = MLP(self.edge_stack[:-1] + (self.max_edges * self.edge_stack[-1],))

        def _get_send_recv(x):
            n_edge = x[-1]
            n_node = x[-2]

            send_recv = self.arch_mlp(x).reshape((2, self.max_edges))

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

            nodes = self.node_mlp(sub_x).reshape((self.max_nodes, self.node_stack[-1]))
            logic = jnp.arange(self.max_nodes) < n_node
            nodes = nodes * logic[:, None]
            return nodes

        self.get_nodes = jax.vmap(_get_nodes)

        def _get_edges(x):
            n_edge = x[-1]
            sub_x = x[:-2]

            edges = self.edge_mlp(sub_x).reshape(self.max_edges, self.edge_stack[-1])
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
