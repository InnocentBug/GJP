from typing import Callable, List, Optional, Sequence

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn


# @jax.jit
def split_and_sum(array, indices):
    cumsum = jnp.cumsum(array, axis=0)
    end_cums = cumsum[jnp.cumsum(indices) - 1]

    diff_out_results = jnp.diff(end_cums, prepend=0, axis=0)
    return diff_out_results


class MLP(nn.Module):
    """A multi-layer perceptron."""

    feature_sizes: Sequence[int]
    dropout_rate: float = 0
    deterministic: bool = True
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for size in self.feature_sizes:
            x = nn.Dense(features=size)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(x)
        return x


class MessagePassingLayer(nn.Module):
    node_feature_sizes: Sequence[int]
    edge_feature_sizes: Sequence[int]
    edge_feature_sizes: Sequence[int]
    global_feature_sizes: Sequence[int]
    num_nodes: int = None
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    dropout_rate: float = 0
    deterministic: bool = True

    @nn.compact
    def __call__(self, graph):
        send_node_feature = graph.nodes[graph.senders]
        recv_node_feature = graph.nodes[graph.receivers]
        edge_features = graph.edges
        global_features = graph.globals

        concat_args = jnp.hstack([send_node_feature, recv_node_feature, edge_features])

        # Nodes
        node_mlp = MLP(self.node_feature_sizes, self.dropout_rate, self.deterministic, self.activation)

        def node_mlp_function(concat_args):
            return node_mlp(concat_args)

        vec_node_mlp = jax.vmap(node_mlp_function, in_axes=0)

        new_tmp_nodes = vec_node_mlp(concat_args)
        recv_nodes = jraph.segment_sum(new_tmp_nodes, graph.receivers, num_segments=self.num_nodes)
        if self.num_nodes is None:
            num_nodes = graph.nodes.shape[0]
            new_nodes = jnp.vstack((recv_nodes, jnp.zeros((num_nodes - recv_nodes.shape[0],) + recv_nodes.shape[1:])))
        else:
            new_nodes = recv_nodes

        # Edges
        edge_mlp = MLP(self.edge_feature_sizes, self.dropout_rate, self.deterministic, self.activation)

        def edge_mlp_function(concat_args):
            return edge_mlp(concat_args)

        vec_edge_mlp = jax.vmap(edge_mlp_function)

        new_edges = vec_edge_mlp(concat_args)

        # # Globals
        global_mlp = MLP(self.global_feature_sizes, self.dropout_rate, self.deterministic, self.activation)

        def global_mlp_function(concat_args):
            return global_mlp(concat_args)

        global_node_mlp = MLP(self.global_feature_sizes, self.dropout_rate, self.deterministic, self.activation)

        def global_node_mlp_function(concat_args):
            return global_node_mlp(concat_args)

        global_edge_mlp = MLP(self.global_feature_sizes, self.dropout_rate, self.deterministic, self.activation)

        def global_edge_mlp_function(concat_args):
            return global_edge_mlp(concat_args)

        global_node_mlp_vmap = jax.vmap(global_node_mlp_function, in_axes=0)
        global_edge_mlp_vmap = jax.vmap(global_edge_mlp_function, in_axes=0)
        global_mlp_vmap = jax.vmap(global_mlp_function, in_axes=0)

        # Split and sum node features by graph
        summed_node_features = split_and_sum(graph.nodes, graph.n_node)
        summed_edge_features = split_and_sum(graph.edges, graph.n_edge)

        tmp_node_global = global_node_mlp_vmap(summed_node_features)
        tmp_edge_global = global_edge_mlp_vmap(summed_edge_features)
        global_edge_mlp_vmap(summed_edge_features)

        tmp_global = global_mlp_vmap(global_features)

        final_global_mlp = MLP(self.global_feature_sizes, self.dropout_rate, self.deterministic, self.activation)
        final_args = jnp.hstack([tmp_global, tmp_node_global, tmp_edge_global])
        new_global = final_global_mlp(final_args)

        out_graph = graph._replace(nodes=new_nodes, edges=new_edges, globals=new_global)
        return out_graph


class MessagePassing(nn.Module):
    node_feature_sizes: Sequence[Sequence[int]]
    edge_feature_sizes: Sequence[Sequence[int]]
    global_feature_sizes: Sequence[Sequence[int]]
    num_nodes: int = None
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.leaky_relu
    dropout_rate: float = 0
    deterministic: bool = True

    def setup(self):
        if len(self.node_feature_sizes) != len(self.edge_feature_sizes):
            raise RuntimeError("The size of the edge, node, and global message passing stacks must be identical.")
        if len(self.global_feature_sizes) != len(self.edge_feature_sizes):
            raise RuntimeError("The size of the edge, node, and global message passing stacks must be identical.")

        size = len(self.node_feature_sizes)

        self.msg_layers = [
            MessagePassingLayer(
                self.node_feature_sizes[i],
                self.edge_feature_sizes[i],
                self.global_feature_sizes[i],
                activation=self.activation,
                dropout_rate=self.dropout_rate,
                deterministic=self.deterministic,
                num_nodes=self.num_nodes,
            )
            for i in range(size)
        ]

    def __call__(self, in_graphs):
        tmp_graphs = in_graphs
        for layer in self.msg_layers:
            tmp_graphs = layer(tmp_graphs)
        return tmp_graphs
