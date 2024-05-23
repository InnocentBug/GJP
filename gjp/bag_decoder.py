from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .model import MLP


@jax.custom_jvp
def diff_round(x):
    return jnp.rint(x)


_APPROX_FACTOR = 10.0


def _approx_round(x):
    return jnp.cos(x * 2 * jnp.pi - jnp.pi / 2) / _APPROX_FACTOR + x


def _approx_round_diff(x):
    return 2 * jnp.pi / _APPROX_FACTOR * jnp.cos(2 * x * jnp.pi) + 1


@diff_round.defjvp
def diff_round_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = diff_round(x)
    tangent_out = _approx_round_diff(x) * x_dot
    return primal_out, tangent_out


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
    max_edges: int
    max_nodes: int
    mlp_size: Sequence[int]
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self._mlp_kwargs = self.mlp_kwargs
        if self._mlp_kwargs is None:
            self._mlp_kwargs = {}

        self.init_edge_mlp = MLP(self.mlp_size + (int(self.max_edges) * 2,), **self._mlp_kwargs)
        self.init_edge_feature_mlp = MLP(self.mlp_size[:-1] + (int(self.max_edges) * int(self.mlp_size[-1]),), **self._mlp_kwargs)

        def _init_edges(sub_x):
            n_node = sub_x[-2]
            sub_x = sub_x[:-2]
            # controlled output between 0, 1
            edge_send_recv = (jnp.tanh(self.init_edge_mlp(sub_x)) + 1) / 2
            edge_send_recv = diff_round(edge_send_recv * (n_node - 1)).reshape((int(self.max_edges), 2))

            senders = edge_send_recv[:, 0]
            receivers = edge_send_recv[:, 1]
            features = self.init_edge_feature_mlp(sub_x).reshape(self.max_edges, self.mlp_size[-1])
            return senders, receivers, features

        self.init_edges = jax.vmap(_init_edges)

    def __call__(self, x):
        return self.init_edges(x)


# class GraphBagDecoder(nn.Module):
# n_node = x[:, -2]
# n_edge = x[:, -1]
