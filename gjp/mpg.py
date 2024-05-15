from typing import Dict, Optional, Sequence

import jax.numpy as jnp
import jraph
from flax import linen as nn

from .model import MLP


class MessagePassingGraphLayer(nn.Module):
    node_stack: Optional[Sequence[int]] = None
    edge_stack: Optional[Sequence[int]] = None
    attention_stack: Optional[Sequence[int]] = None
    global_stack: Optional[Sequence[int]] = None
    mlp_kwargs: Optional[Dict] = None

    def setup(self):
        if self.mlp_kwargs is None:
            self._mlp_kwargs = {}
        else:
            self._mlp_kwargs = self.mlp_kwargs

        update_nodes = None
        if self.node_stack:
            self.node_mlp = MLP(self.node_stack, **self._mlp_kwargs)

            @jraph.concatenated_args
            def update_nodes(features):
                return self.node_mlp(features)

        update_edges = None
        if self.edge_stack:
            self.edge_mlp = MLP(self.edge_stack, **self._mlp_kwargs)

            @jraph.concatenated_args
            def update_edges(features):
                return self.edge_mlp(features)

        attention_edges = None
        attention_reduce = None
        attention_norm = None
        if self.attention_stack:
            self.attention_mlp = MLP(self.attention_stack + (1,), **self._mlp_kwargs)

            @jraph.concatenated_args
            def attention_edges(features):
                return self.attention_mlp(features)

            attention_norm = jraph.segment_softmax
            attention_reduce = jnp.multiply

        update_global = None
        node_to_global = None
        edge_to_global = None
        if self.global_stack:
            self.node_to_global_mlp = MLP(self.global_stack, **self._mlp_kwargs)

            def node_to_global(node_attributes, index, max_graph):
                tmp_node = self.node_to_global_mlp(node_attributes)
                return jraph.segment_mean(tmp_node, index, max_graph)

            self.edge_to_global_mlp = MLP(self.global_stack, **self._mlp_kwargs)

            def edge_to_global(edge_attributes, index, max_graph):
                tmp_edge = self.edge_to_global_mlp(edge_attributes)
                return jraph.segment_mean(tmp_edge, index, max_graph)

            self.global_mlp = MLP(self.global_stack, **self._mlp_kwargs)

            @jraph.concatenated_args
            def update_global(features):
                return self.global_mlp(features)

        self.graph_network = jraph.GraphNetwork(
            update_node_fn=update_nodes,
            aggregate_edges_for_nodes_fn=jraph.segment_mean,
            update_edge_fn=update_edges,
            attention_logit_fn=attention_edges,
            attention_normalize_fn=attention_norm,
            attention_reduce_fn=attention_reduce,
            update_global_fn=update_global,
            aggregate_nodes_for_globals_fn=node_to_global,
            aggregate_edges_for_globals_fn=edge_to_global,
        )

    def __call__(self, x):
        return self.graph_network(x)


class MessagePassingGraph(nn.Module):
    node_stack: Optional[Sequence[Optional[Sequence[int]]]] = None
    edge_stack: Optional[Sequence[Optional[Sequence[int]]]] = None
    attention_stack: Optional[Sequence[Optional[Sequence[int]]]] = None
    global_stack: Optional[Sequence[Optional[Sequence[int]]]] = None
    mlp_kwargs: Optional[Dict] = None

    def setup(self):
        if self.node_stack is not None and self.edge_stack is not None:
            if len(self.node_stack) != len(self.edge_stack):
                raise RuntimeError("The size of the edge, node, and global message passing stack must be identical.")
        if self.global_stack is not None and self.edge_stack is not None:
            if len(self.global_stack) != len(self.edge_stack):
                raise RuntimeError("The size of the edge, node, and global message passing stack must be identical.")
        if self.node_stack is not None and self.global_stack is not None:
            if len(self.global_stack) != len(self.node_stack):
                raise RuntimeError("The size of the edge, node, and global message passing stack must be identical.")
        if self.node_stack is not None and self.attention_stack is not None:
            if len(self.attention_stack) != len(self.node_stack):
                raise RuntimeError("The size of the edge, node, attention, and global message passing stack must be identical.")

        size = 0
        if self.node_stack is not None:
            size = len(self.node_stack)
        elif self.edge_stack is not None:
            size = len(self.edge_stack)
        elif self.global_stack is not None:
            size = len(self.global_stack)
        elif self.attention_stack is not None:
            size = len(self.attention_stack)

        self.msg_layers = [
            MessagePassingGraphLayer(
                self.node_stack[i] if self.node_stack is not None else None,
                self.edge_stack[i] if self.edge_stack is not None else None,
                self.attention_stack[i] if self.attention_stack is not None else None,
                self.global_stack[i] if self.global_stack is not None else None,
                self.mlp_kwargs,
            )
            for i in range(size)
        ]

    def __call__(self, in_graphs):
        tmp_graphs = in_graphs
        for layer in self.msg_layers:
            tmp_graphs = layer(tmp_graphs)
        return tmp_graphs
