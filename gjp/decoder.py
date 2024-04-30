from typing import Sequence

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn

from .model import MLP


class InitialEdgeDecoder(nn.Module):
    mlp_stack: Sequence[int]
    max_num_edges: int
    max_num_nodes: int
    n_edge_features: int

    @nn.compact
    def __call__(self, x):
        edge_mlp = MLP(self.mlp_stack + (self.max_num_nodes * self.max_num_edges,))

        def apply_edge_mlp(sub_x):
            mlp_sub_x = edge_mlp(sub_x)
            return mlp_sub_x.reshape((self.max_num_edges, self.max_num_nodes))

        vmap_edge_mlp = jax.vmap(apply_edge_mlp, in_axes=0)
        edge_sparse = vmap_edge_mlp(x)
        edge_sparse_soft = jax.nn.softmax(edge_sparse, axis=2)

        edge_ids = jnp.argsort(-edge_sparse_soft, axis=2)
        edge_sort = jnp.sort(-edge_sparse_soft, axis=2)
        senders = edge_ids[:, :, 0]
        receivers = edge_ids[:, :, 1]

        senders = senders + jnp.arange(x.shape[0])[:, None] * self.max_num_nodes
        receivers = receivers + jnp.arange(x.shape[0])[:, None] * self.max_num_nodes

        edge_prob = -edge_sort[:, :, :2]

        feature_mlp = MLP(self.mlp_stack + (self.max_num_edges * (self.n_edge_features - 2),))

        def apply_feature_mlp(sub_x):
            mlp_sub_x = feature_mlp(sub_x)
            return mlp_sub_x.reshape((self.max_num_edges, (self.n_edge_features - 2)))

        vmap_feature_mlp = jax.vmap(apply_feature_mlp, in_axes=0)
        features = vmap_feature_mlp(x)

        features = jnp.concatenate((edge_prob, features), axis=2)

        return senders, receivers, features


class InitialNodeDecoder(nn.Module):
    mlp_stack: Sequence[int]
    n_node_features: int
    max_num_nodes: int

    @nn.compact
    def __call__(self, x):
        node_mlp = MLP(self.mlp_stack + ((self.n_node_features) * self.max_num_nodes,))

        def apply_mlp(sub_x):
            sub_mlp = node_mlp(sub_x)
            return sub_mlp.reshape((self.max_num_nodes, self.n_node_features))

        vmap_mlp = jax.vmap(apply_mlp, in_axes=0)
        node_features = vmap_mlp(x)
        return node_features


class InitGraphDecoder(nn.Module):
    init_edge_stack: Sequence[int]
    init_edge_features: int
    init_node_stack: Sequence[int]
    init_node_features: int
    max_num_nodes: int
    max_num_edges: int

    @nn.compact
    def __call__(self, x):
        jnp.rint(x[:, -2])
        jnp.rint(x[:, -1])

        init_edges = InitialEdgeDecoder(mlp_stack=self.init_edge_stack, max_num_nodes=self.max_num_nodes, max_num_edges=self.max_num_edges, n_edge_features=self.init_edge_features)
        init_senders, init_receivers, init_edge_features = init_edges(x)
        init_edge_features = init_edge_features.reshape((init_edge_features.shape[0] * init_edge_features.shape[1], init_edge_features.shape[2]))

        init_nodes = InitialNodeDecoder(mlp_stack=self.init_node_stack, n_node_features=self.init_node_features, max_num_nodes=self.max_num_nodes)

        init_node_features = init_nodes(x)
        init_node_features = init_node_features.reshape(init_node_features.shape[0] * init_node_features.shape[1], init_node_features.shape[2])

        initial_graph = jraph.GraphsTuple(
            nodes=init_node_features,
            edges=init_edge_features,
            senders=init_senders.flatten(),
            receivers=init_receivers.flatten(),
            n_node=jnp.ones(x.shape[0], dtype=int) * self.max_num_nodes,
            n_edge=jnp.ones(x.shape[0], dtype=int) * self.max_num_edges,
            globals=x,
        )
        return initial_graph


class GraphDecoder(nn.Module):
    prob_stack: Sequence[int]
    edge_feature_stack: Sequence[int]
