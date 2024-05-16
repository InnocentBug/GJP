from typing import Sequence

import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn

from .model import MLP, MessagePassing


class InitialEdgeDecoder(nn.Module):
    mlp_stack: Sequence[int]
    max_num_edges: int
    max_num_nodes: int
    max_edge_node: int
    n_edge_features: int

    def setup(self):
        self.edge_mlp = MLP(self.mlp_stack + (self.max_edge_node,))
        self.feature_dim: int = int(self.max_num_edges) * int((int(self.n_edge_features) - int(2)))
        self.feature_mlp = MLP(self.mlp_stack + (self.feature_dim,))

    def __call__(self, x):
        def apply_edge_mlp(sub_x):
            mlp_sub_x = self.edge_mlp(sub_x)
            return mlp_sub_x.reshape((self.max_num_edges, self.max_num_nodes))

        vmap_edge_mlp = jax.vmap(apply_edge_mlp, in_axes=0)

        edge_sparse = vmap_edge_mlp(x)
        edge_sparse_soft = jax.nn.softmax(edge_sparse, axis=2)

        edge_ids = jnp.argsort(-edge_sparse_soft, axis=2)
        edge_sort = jnp.sort(-edge_sparse_soft, axis=2)
        senders = edge_ids[:, :, 0]
        receivers = edge_ids[:, :, 1]

        senders = senders + jnp.arange(x.shape[0])[:, None] * int(self.max_num_nodes)
        receivers = receivers + jnp.arange(x.shape[0])[:, None] * int(self.max_num_nodes)

        edge_prob = -edge_sort[:, :, :2]

        def apply_feature_mlp(sub_x):
            mlp_sub_x = self.feature_mlp(sub_x)
            return mlp_sub_x.reshape((self.max_num_edges, (self.n_edge_features - 2)))

        vmap_feature_mlp = jax.vmap(apply_feature_mlp, in_axes=0)

        features = vmap_feature_mlp(x)

        features = jnp.concatenate((edge_prob, features), axis=2)

        return senders, receivers, features


class InitialNodeDecoder(nn.Module):
    mlp_stack: Sequence[int]
    n_node_features: int
    max_num_nodes: int

    def setup(self):
        self.node_mlp = MLP(self.mlp_stack + (int(self.n_node_features) * int(self.max_num_nodes),))

    def __call__(self, x):
        def apply_mlp(sub_x):
            sub_mlp = self.node_mlp(sub_x)
            return sub_mlp.reshape((self.max_num_nodes, self.n_node_features))

        vmap_mlp = jax.vmap(apply_mlp, in_axes=0)
        node_features = vmap_mlp(x)
        return node_features


class InitialGraphDecoder(nn.Module):
    init_edge_stack: Sequence[int]
    init_edge_features: int
    init_node_stack: Sequence[int]
    init_node_features: int
    max_num_nodes: int
    max_num_edges: int
    max_edge_node: int

    def setup(self):
        self.init_edges = InitialEdgeDecoder(mlp_stack=self.init_edge_stack, max_num_nodes=self.max_num_nodes, max_num_edges=self.max_num_edges, n_edge_features=self.init_edge_features, max_edge_node=self.max_edge_node)
        self.init_nodes = InitialNodeDecoder(mlp_stack=self.init_node_stack, n_node_features=self.init_node_features, max_num_nodes=self.max_num_nodes)

    def __call__(self, x):

        init_senders, init_receivers, init_edge_features = self.init_edges(x)
        init_edge_features = init_edge_features.reshape((init_edge_features.shape[0] * init_edge_features.shape[1], init_edge_features.shape[2]))

        init_node_features = self.init_nodes(x)
        init_node_features = init_node_features.reshape(init_node_features.shape[0] * init_node_features.shape[1], init_node_features.shape[2])

        initial_graph = jraph.GraphsTuple(
            nodes=init_node_features,
            edges=init_edge_features,
            senders=init_senders.flatten(),
            receivers=init_receivers.flatten(),
            n_node=jnp.ones(x.shape[0], dtype=int) * int(self.max_num_nodes),
            n_edge=jnp.ones(x.shape[0], dtype=int) * int(self.max_num_edges),
            globals=x,
        )
        return initial_graph


class GraphDecoder(nn.Module):
    init_edge_stack: Sequence[int]
    init_edge_features: int
    init_node_stack: Sequence[int]
    init_node_features: int
    max_num_nodes: int
    max_num_edges: int
    max_edge_node: int
    max_num_graphs: int
    prob_node_stack: Sequence[Sequence[int]]
    prob_edge_stack: Sequence[Sequence[int]]
    feature_edge_stack: Sequence[Sequence[int]]
    feature_node_stack: Sequence[Sequence[int]]
    mean_instead_of_sum: bool = True

    def setup(self):
        self.initial_graph_decoder = InitialGraphDecoder(
            init_edge_stack=self.init_edge_stack,
            init_edge_features=self.init_edge_features,
            init_node_stack=self.init_node_stack,
            init_node_features=self.init_node_features,
            max_num_nodes=self.max_num_nodes,
            max_num_edges=self.max_num_edges,
            max_edge_node=self.max_edge_node,
        )

        self.num_nodes: int = int(self.max_num_graphs) * int(self.max_num_nodes)
        # Use max_num_graphs instead of x
        self.prob_gnn = MessagePassing(node_feature_sizes=self.prob_node_stack + ((1,),), edge_feature_sizes=self.prob_node_stack + ((1,),), global_feature_sizes=None, mean_instead_of_sum=self.mean_instead_of_sum, num_nodes=self.num_nodes)

        self.final_gnn = MessagePassing(node_feature_sizes=self.feature_node_stack, edge_feature_sizes=self.feature_edge_stack, global_feature_sizes=None, mean_instead_of_sum=self.mean_instead_of_sum, num_nodes=self.num_nodes)

    def __call__(self, x):
        n_node = jnp.rint(x[:, -2])
        n_node_mod = n_node + jnp.arange(int(x.shape[0])) * int(self.max_num_nodes)
        n_edge = jnp.rint(x[:, -1])

        initial_graph = self.initial_graph_decoder(x)
        initial_nodes = initial_graph.nodes.reshape(x.shape[0], self.max_num_nodes, initial_graph.nodes.shape[1])
        # The initial graph has unsorted info, without interpretability yet.
        # We run 2 Message-Passing GNN on this, the first one predicts the probability of nodes and edges
        prob_graph = self.prob_gnn(initial_graph)

        prob_graph = prob_graph._replace(nodes=prob_graph.nodes[: x.shape[0] * int(self.max_num_nodes)])

        # Now we sort the the nodes and edges, to make a valid and an invalid graph out of every one
        node_probs = jax.nn.softmax(prob_graph.nodes.reshape((x.shape[0], self.max_num_nodes)), axis=1) + 1
        prob_sort_idx = jnp.argsort(-node_probs, axis=1)
        sorted_nodes = initial_nodes[jnp.arange(initial_nodes.shape[0])[:, None], prob_sort_idx]

        # Set batch node properties to 0
        node_logic = jnp.arange(sorted_nodes.shape[0] * sorted_nodes.shape[1]) < jnp.repeat(n_node_mod, jnp.ones(x.shape[0], dtype=int) * self.max_num_nodes, total_repeat_length=sorted_nodes.shape[0] * sorted_nodes.shape[1])
        node_logic = node_logic.reshape((sorted_nodes.shape[0], sorted_nodes.shape[1]))
        node_logic = jnp.repeat(node_logic, sorted_nodes.shape[-1], axis=1).reshape(sorted_nodes.shape)
        sorted_nodes = sorted_nodes * node_logic

        # Now we do the edges.
        # First sort them as we sorted the nodes.
        inverse_edge_permutation = jnp.argsort(prob_sort_idx, axis=1) + jnp.arange(x.shape[0])[:, None] * self.max_num_nodes

        new_senders = inverse_edge_permutation.flatten()[initial_graph.senders.astype(int)]
        new_senders = new_senders.reshape((x.shape[0], self.max_num_edges))
        new_receivers = inverse_edge_permutation.flatten()[initial_graph.receivers.astype(int)]
        new_receivers = new_receivers.reshape((x.shape[0], self.max_num_edges))

        # Then evaluate the probabilities of the edges existing
        edge_probs = jax.nn.softmax(prob_graph.edges.reshape((x.shape[0], int(self.max_num_edges))), axis=1) + 1
        edge_probs = jnp.arange(x.shape[0] * int(self.max_num_edges)).reshape((x.shape[0], int(self.max_num_edges))) + 1

        # Edges that connect to non-existing nodes are set to prob = 0
        edge_logic = jnp.logical_and(new_senders < n_node_mod[:, None], new_receivers < n_node_mod[:, None])
        edge_probs = edge_probs * edge_logic

        edge_sort_idx = jnp.argsort(-edge_probs, axis=1)
        initial_edges = initial_graph.edges.reshape(x.shape[0], self.max_num_edges, initial_graph.edges.shape[1])
        sorted_edges = initial_edges[jnp.arange(initial_edges.shape[0])[:, None], edge_sort_idx]
        new_senders = new_senders[jnp.arange(new_senders.shape[0])[:, None], edge_sort_idx]
        new_receivers = new_receivers[jnp.arange(new_receivers.shape[0])[:, None], edge_sort_idx]

        n_edge = jnp.min(jnp.vstack([n_edge, jnp.sum(edge_logic, axis=1, dtype=int)], dtype=int), axis=0)
        n_edge_mod = n_edge + jnp.arange(x.shape[0], dtype=int) * int(self.max_num_edges)

        # Eliminate non-existent edges by send/recv
        n_edge_repeat = jnp.repeat(n_edge_mod, int(self.max_num_edges), axis=0)
        exclusion_logic = jnp.arange(n_edge_repeat.size, dtype=int) >= n_edge_repeat
        n_node_repeat = jnp.repeat(n_node_mod, int(self.max_num_edges), axis=0)
        new_senders = new_senders.flatten() * ~exclusion_logic + exclusion_logic * n_node_repeat
        new_receivers = new_receivers.flatten() * ~exclusion_logic + exclusion_logic * n_node_repeat

        # Eliminate all probs of non-existent edge properties
        repeat_logic = jnp.repeat(exclusion_logic, sorted_edges.shape[2], axis=0)
        sorted_edges = sorted_edges.flatten() * ~repeat_logic.flatten()
        sorted_edges = sorted_edges.reshape(initial_edges.shape)

        new_n_node = jnp.vstack((n_node, jnp.repeat(self.max_num_nodes, x.shape[0]) - n_node)).flatten(order="F")
        new_n_edge = jnp.vstack((n_edge, jnp.repeat(self.max_num_edges, x.shape[0]) - n_edge)).flatten(order="F")
        new_global = jnp.hstack((x, x * 0)).reshape((2 * x.shape[0],) + x.shape[1:])

        new_arch_graph = jraph.GraphsTuple(
            nodes=sorted_nodes.reshape(sorted_nodes.shape[0] * sorted_nodes.shape[1], sorted_nodes.shape[-1]),
            edges=sorted_edges.reshape(sorted_edges.shape[0] * sorted_edges.shape[1], sorted_edges.shape[2]),
            senders=new_senders.astype(int),
            receivers=new_receivers.astype(int),
            n_edge=new_n_edge.astype(int),
            n_node=new_n_node.astype(int),
            globals=new_global,
        )

        # And finally we run message passing again new graph with correct architecture, to finalize node and edge attributes

        final_graph = self.final_gnn(new_arch_graph)
        final_graph = final_graph._replace(nodes=final_graph.nodes[: x.shape[0] * int(self.max_num_nodes)])

        return final_graph
