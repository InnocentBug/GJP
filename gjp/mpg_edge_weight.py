import functools
from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import jraph
from flax import linen as nn

from .model import MLP, split_and_mean, split_and_sum


class MessagePassingLayerEW(nn.Module):
    node_feature_sizes: Sequence[int]
    edge_feature_sizes: Sequence[int]
    global_feature_sizes: Sequence[int]
    mean_instead_of_sum: bool = False
    mlp_kwargs: Optional[Dict] = None

    def setup(self):
        self._mlp_kwargs = self.mlp_kwargs
        if self._mlp_kwargs is None:
            self._mlp_kwargs = {}

        if self.node_feature_sizes:
            self._node_mlp = MLP(self.node_feature_sizes, **self._mlp_kwargs)

            def node_mlp_function(concat_args):
                return self._node_mlp(concat_args)

            self.node_mlp = jax.vmap(node_mlp_function, in_axes=0)

        if self.edge_feature_sizes:
            self._edge_mlp = MLP(self.edge_feature_sizes, **self._mlp_kwargs)

            def edge_mlp_function(concat_args):
                return self._edge_mlp(concat_args)

            self.edge_mlp = jax.vmap(edge_mlp_function)

        if self.global_feature_sizes:
            self._global_mlp = MLP(self.global_feature_sizes, **self._mlp_kwargs)

            def global_mlp_function(concat_args):
                return self._global_mlp(concat_args)

            self._global_node_mlp = MLP(self.global_feature_sizes, **self._mlp_kwargs)

            def global_node_mlp_function(concat_args):
                return self._global_node_mlp(concat_args)

            self._global_edge_mlp = MLP(self.global_feature_sizes, **self._mlp_kwargs)

            def global_edge_mlp_function(concat_args):
                return self._global_edge_mlp(concat_args)

            self.global_node_mlp = jax.vmap(global_node_mlp_function, in_axes=0)
            self.global_edge_mlp = jax.vmap(global_edge_mlp_function, in_axes=0)
            self.global_mlp = jax.vmap(global_mlp_function, in_axes=0)

            self.final_global_mlp = MLP(self.global_feature_sizes, **self._mlp_kwargs)

    def __call__(self, graph, edge_weights=None):
        num_nodes = tree.tree_leaves(graph.nodes)[0].shape[0]
        send_node_feature = graph.nodes[graph.senders]
        recv_node_feature = graph.nodes[graph.receivers]
        edge_features = graph.edges

        global_features = graph.globals

        edge_repeat_global = jnp.repeat(global_features, graph.n_edge, axis=0, total_repeat_length=graph.receivers.shape[0])

        concat_args = jnp.hstack([send_node_feature, recv_node_feature, edge_features, edge_repeat_global])
        if edge_weights is not None:
            edge_weights = edge_weights.flatten()
            edge_features = edge_weights[:, None] * edge_features
            concat_args = edge_weights[:, None] * concat_args

        # Nodes
        if self.node_feature_sizes is not None:
            new_tmp_nodes = self.node_mlp(concat_args)
            if self.mean_instead_of_sum:
                recv_nodes = jraph.segment_mean(new_tmp_nodes, graph.receivers, num_segments=num_nodes)
            else:
                recv_nodes = jraph.segment_sum(new_tmp_nodes, graph.receivers, num_segments=num_nodes)
            new_nodes = recv_nodes
        else:
            new_nodes = graph.nodes

        # Edges
        if self.edge_feature_sizes is not None:
            new_edges = self.edge_mlp(concat_args)
        else:
            new_edges = graph.edges

        # Globals
        if self.global_feature_sizes is not None:

            # Split and sum node features by graph
            if self.mean_instead_of_sum:
                summed_node_features = split_and_mean(graph.nodes, graph.n_node)
                summed_edge_features = split_and_mean(edge_features, graph.n_edge)
            else:
                summed_node_features = split_and_sum(graph.nodes, graph.n_node)
                summed_edge_features = split_and_sum(edge_features, graph.n_edge)

            tmp_node_global = self.global_node_mlp(summed_node_features)
            tmp_edge_global = self.global_edge_mlp(summed_edge_features)
            tmp_global = self.global_mlp(global_features)

            final_args = jnp.hstack([tmp_global, tmp_node_global, tmp_edge_global])
            new_global = self.final_global_mlp(final_args)
        else:
            new_global = graph.globals

        out_graph = graph._replace(nodes=new_nodes, edges=new_edges, globals=new_global)
        return out_graph


class MessagePassingEW(nn.Module):
    node_feature_sizes: Sequence[Sequence[int]]
    edge_feature_sizes: Sequence[Sequence[int]]
    global_feature_sizes: Sequence[Sequence[int]]
    mean_instead_of_sum: bool = False
    mlp_kwargs: Optional[Dict] = None

    def setup(self):
        if self.node_feature_sizes is not None and self.edge_feature_sizes is not None:
            if len(self.node_feature_sizes) != len(self.edge_feature_sizes):
                raise RuntimeError("The size of the edge, node, and global message passing stacks must be identical.")
        if self.global_feature_sizes is not None and self.edge_feature_sizes is not None:
            if len(self.global_feature_sizes) != len(self.edge_feature_sizes):
                raise RuntimeError("The size of the edge, node, and global message passing stacks must be identical.")
        if self.node_feature_sizes is not None and self.global_feature_sizes is not None:
            if len(self.global_feature_sizes) != len(self.node_feature_sizes):
                raise RuntimeError("The size of the edge, node, and global message passing stacks must be identical.")

        size = 0
        if self.node_feature_sizes is not None:
            size = len(self.node_feature_sizes)
        elif self.edge_feature_sizes is not None:
            size = len(self.edge_feature_sizes)
        elif self.global_feature_sizes is not None:
            size = len(self.global_feature_sizes)

        self.msg_layers = [
            MessagePassingLayerEW(
                self.node_feature_sizes[i] if self.node_feature_sizes is not None else None,
                self.edge_feature_sizes[i] if self.edge_feature_sizes is not None else None,
                self.global_feature_sizes[i] if self.global_feature_sizes is not None else None,
                mean_instead_of_sum=self.mean_instead_of_sum,
                mlp_kwargs=self.mlp_kwargs,
            )
            for i in range(size)
        ]

    def __call__(self, in_graphs, edge_weights=None):
        tmp_graphs = in_graphs
        for layer in self.msg_layers:
            tmp_graphs = layer(tmp_graphs, edge_weights)
        return tmp_graphs


def _tanh_filter(x, factor=1):
    return (1 + jnp.tanh((x - 0.5) * 2 * jnp.pi * factor)) / 2.0


def gumbel_softmax_sample(logits, temperature, rng, mask=None, axis=-1):
    """
    @article{jang2016categorical,
    title={Categorical reparameterization with gumbel-softmax},
    author={Jang, Eric and Gu, Shixiang and Poole, Ben},
    journal={arXiv preprint arXiv:1611.01144},
    year={2016}
    }
    """
    gumbel_noise = jax.random.gumbel(rng, logits.shape)
    random_sample = (logits + gumbel_noise) / temperature
    if mask is not None:
        tanh_mask = _tanh_filter(mask)
        random_sample *= 1 - tanh_mask

    softmax = jax.nn.softmax(random_sample, axis=axis)
    return softmax


def gumbel_topk(x, k, max_iter, rng, temperature, pre_mask=None, pre_filter=None):

    if pre_filter is not None:
        x = pre_filter(x)
    if pre_mask is None:
        pre_mask = jnp.zeros(x.shape)
    else:
        k += jnp.sum(pre_mask, axis=1)
    init_val = (rng, pre_mask)

    mask_filter = functools.partial(_tanh_filter, factor=2)

    def _inner_gumbel_topk(i, val):
        val_rng, mask = val
        return_rng, gumbel_rng = jax.random.split(val_rng)
        sample = gumbel_softmax_sample(x, temperature, gumbel_rng, mask=mask, axis=1)

        new_mask = mask_filter(mask + sample)

        old_diff = jnp.abs(jnp.sum(mask, axis=1) - k)
        new_diff = jnp.abs(jnp.sum(new_mask, axis=1) - k)

        update_logic = new_diff < old_diff
        update_logic = (jnp.tanh((old_diff - new_diff) * 2 * jnp.pi) + 1) / 2

        mask = mask_filter(mask + update_logic[:, None] * sample)

        return return_rng, mask

    return_rng, mask = jax.lax.fori_loop(0, max_iter, _inner_gumbel_topk, init_val)

    return mask


def edge_weights_sharpness_loss(edge_weights, factor=2):
    data_clean = jnp.clip(edge_weights, 0, 1)

    data_extra = edge_weights - data_clean
    extra_val = jnp.sum(jnp.abs(data_extra))

    data = -(data_clean**2) + data_clean
    data = factor**2 * data

    # Just better then mean.
    # Not meant to be differentiable
    n_elements = 1 + jnp.sum(data > 1e-5)

    loss = jnp.sum(data) / n_elements + (jnp.sqrt(extra_val + 1e-3) + extra_val**2) / 2

    return loss


def edge_weights_n_edge_loss(edge_weights, n_edge):
    a = jnp.sum(edge_weights, axis=1)
    la = n_edge.flatten()
    b = a - la

    loss = jnp.mean(b**2) + jnp.mean(jnp.abs(b))
    return loss
