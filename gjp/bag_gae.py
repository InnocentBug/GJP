from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .bag_decoder import GraphBagDecoder
from .mpg import MessagePassingGraph


class BagGAE(nn.Module):
    max_num_nodes: int
    encoder_stack: Sequence[Sequence[int]]
    init_stack: Sequence[int]
    decoder_stack: Sequence[Sequence[int]]
    node_features: int
    edge_features: int
    multi_edge_repeat: int = 1
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self.encoder = MessagePassingGraph(node_stack=self.encoder_stack, edge_stack=self.encoder_stack, attention_stack=self.encoder_stack, global_stack=self.encoder_stack, mean_aggregate=False, mlp_kwargs=self.mlp_kwargs)
        self.sigma_encoder = MessagePassingGraph(node_stack=self.encoder_stack, edge_stack=self.encoder_stack, attention_stack=self.encoder_stack, global_stack=self.encoder_stack, mean_aggregate=False, mlp_kwargs=self.mlp_kwargs)

        self.decoder = GraphBagDecoder(
            max_nodes=self.max_num_nodes,
            init_node_stack=self.init_stack,
            init_edge_stack=self.init_stack,
            init_mpg_stack=self.decoder_stack,
            final_mpg_edge_stack=self.decoder_stack + ((self.edge_features,),),
            final_mpg_node_stack=self.decoder_stack + ((self.node_features,),),
            multi_edge_repeat=self.multi_edge_repeat,
            mlp_kwargs=self.mlp_kwargs,
        )

    def __call__(self, x):
        mu_wo_node = self.encoder(x).globals
        log_sigma_wo_node = self.sigma_encoder(x).globals
        sigma_wo_node = jnp.exp(log_sigma_wo_node)
        rng = self.make_rng("reparametrize")
        eps = jax.random.normal(rng, mu_wo_node.shape)
        z_wo_node = mu_wo_node + sigma_wo_node * eps  # Reparameterization trick

        # Add the node and edge info after adding noise
        z = jnp.vstack((z_wo_node.transpose(), x.n_node, x.n_edge)).transpose()
        reconstructed = self.decoder(z)
        return reconstructed

    def decode(self, z):
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def encode_decode(self, x):
        mu_wo_node = self.encoder(x).globals
        # Add the node and edge info after adding noise
        z = jnp.vstack((mu_wo_node.transpose(), x.n_node, x.n_edge)).transpose()
        reconstructed = self.decoder(z)
        return reconstructed


def find_multi_edge_repeat(batched_graph, ignore_node=None):
    edge_count_map = {}
    for a, b in zip(batched_graph.senders, batched_graph.receivers):
        a = int(a)
        b = int(b)
        if ignore_node is not None:
            if a >= ignore_node or b >= ignore_node:
                continue
        try:
            edge_count_map[(a, b)] += 1
        except KeyError:
            edge_count_map[(a, b)] = 1
    return max(edge_count_map.values())


def loss_function(params, train_state, in_graphs, rngs, metric_state, norm):
    in_metric = metric_state.apply_fn(metric_state.params, in_graphs).globals[:-1]
    tmp_out_graphs = train_state.apply_fn(params, in_graphs, rngs=rngs)
    embedded_space = tmp_out_graphs.globals[::2]
    embedded_space = embedded_space[:, :-2]

    out_globals = jnp.repeat(in_graphs.globals, 2, axis=0)
    out_graphs = tmp_out_graphs._replace(globals=out_globals)

    out_metric = metric_state.apply_fn(metric_state.params, out_graphs).globals[::2]
    out_metric = out_metric[:-1]

    loss = jnp.mean(jnp.sqrt((in_metric - out_metric) ** 2))

    if norm:
        loss *= 1 + jnp.mean(embedded_space**2)

    return loss


def train_step(batch_train, batch_test, train_state, rng, norm, metric_state):
    loss_fn = partial(loss_function, metric_state=metric_state, norm=norm)
    loss_fn_a = partial(loss_function, metric_state=metric_state, norm=True)
    loss_fn_b = partial(loss_function, metric_state=metric_state, norm=False)
    loss_grad_fn = jax.value_and_grad(loss_fn)

    rng, rng_a, rng_b = jax.random.split(rng, 3)
    rngs = {"reparametrize": rng_a, "dropout": rng_b}
    train_loss, grads = loss_grad_fn(train_state.params, train_state, batch_train, rngs)
    rng, rng_a, rng_b = jax.random.split(rng, 3)
    rngs = {"reparametrize": rng_a, "dropout": rng_b}
    train_loss_a = loss_fn_a(train_state.params, train_state, batch_train, rngs)
    train_loss_b = loss_fn_b(train_state.params, train_state, batch_train, rngs)
    test_loss_a = loss_fn_a(train_state.params, train_state, batch_test, rngs)
    test_loss_b = loss_fn_b(train_state.params, train_state, batch_test, rngs)

    train_state = train_state.apply_gradients(grads=grads)

    return train_state, train_loss, train_loss_a, train_loss_b, test_loss_a, test_loss_b
