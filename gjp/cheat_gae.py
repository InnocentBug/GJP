from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .cheat_decoder import CheatDecoder, batch_graph_arrays, make_abs_diff_graph
from .mpg import MessagePassingGraph


class CheatGAE(nn.Module):
    max_nodes: int
    max_edges: int
    arch_stack: Sequence[int]
    node_stack: Sequence[int]
    edge_stack: Sequence[int]
    encoder_stack: Sequence[Sequence[int]]
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self.encoder = MessagePassingGraph(
            node_stack=self.encoder_stack,
            edge_stack=self.encoder_stack,
            attention_stack=self.encoder_stack,
            global_stack=self.encoder_stack,
            mean_aggregate=False,
            mlp_kwargs=self.mlp_kwargs,
        )
        self.sigma_encoder = MessagePassingGraph(
            node_stack=self.encoder_stack,
            edge_stack=self.encoder_stack,
            attention_stack=self.encoder_stack,
            global_stack=self.encoder_stack,
            mean_aggregate=False,
            mlp_kwargs=self.mlp_kwargs,
        )
        self.decoder = CheatDecoder(
            max_nodes=self.max_nodes,
            max_edges=self.max_edges,
            arch_stack=self.arch_stack,
            node_stack=self.node_stack,
            edge_stack=self.edge_stack,
            mlp_kwargs=self.mlp_kwargs,
        )

    def __call__(self, x_graphs):
        # assert x_graphs.nodes.shape[1] == self.node_stack[-1]
        # assert x_graphs.edges.shape[1] == self.edge_stack[-1]

        mu_wo_noise = self.encoder(x_graphs).globals
        log_sigma_wo_noise = self.sigma_encoder(x_graphs).globals
        sigma_wo_noise = jnp.exp(log_sigma_wo_noise)

        rng = self.make_rng("reparametrize")
        eps = jax.random.normal(rng, mu_wo_noise.shape)
        mu = mu_wo_noise + sigma_wo_noise * eps

        z = jnp.vstack((mu.transpose(), x_graphs.n_node, x_graphs.n_edge)).transpose()
        y_graphs = self.decoder(z)
        return y_graphs, mu_wo_noise, log_sigma_wo_noise

    def decode(self, z):
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def encode_decode(self, x):
        mu_wo_node = self.encoder(x).globals
        # No noise or reparametrization
        z = jnp.vstack((mu_wo_node.transpose(), x.n_node, x.n_edge)).transpose()
        reconstructed = self.decoder(z)
        return reconstructed


def train_loss_function(params, state, in_graphs, ref_graph, rngs):
    out_graphs, mu, log_sigma = state.apply_fn(params, in_graphs, rngs=rngs)
    diff_graph = make_abs_diff_graph(out_graphs, ref_graph)

    loss = jnp.mean(diff_graph.nodes) + jnp.max(diff_graph.nodes)
    loss += jnp.mean(diff_graph.edges) + jnp.max(diff_graph.edges)
    loss += jnp.mean(diff_graph.senders) + jnp.max(diff_graph.senders)
    loss += jnp.mean(diff_graph.receivers) + jnp.max(diff_graph.receivers)

    kl_divergence = -0.5 * jnp.sum(1 + log_sigma[:-1] - jnp.square(mu[:-1]) - jnp.exp(log_sigma[:-1]))
    loss += kl_divergence

    return loss


def test_loss_function(params, model, in_graphs, ref_graph, rngs):
    out_graphs = model.apply(params, in_graphs, rngs=rngs, method=model.encode_decode)
    diff_graph = make_abs_diff_graph(out_graphs, ref_graph)

    loss = jnp.mean(diff_graph.nodes) + jnp.max(diff_graph.nodes)
    loss += jnp.mean(diff_graph.edges) + jnp.max(diff_graph.edges)
    loss += jnp.mean(diff_graph.senders) + jnp.max(diff_graph.senders)
    loss += jnp.mean(diff_graph.receivers) + jnp.max(diff_graph.receivers)

    return loss


def train_step(state, model, train_graph, test_graph, rng, ref_train_graph=None, ref_test_graph=None):

    if ref_train_graph is None:
        ref_train_graph = batch_graph_arrays(train_graph, model.max_edges, model.max_nodes)
    if ref_test_graph is None:
        ref_test_graph = batch_graph_arrays(test_graph, model.max_edges, model.max_nodes)

    reduced_loss_grad = partial(train_loss_function, state=state, in_graphs=train_graph, ref_graph=ref_train_graph)
    loss_grad_fn = jax.value_and_grad(reduced_loss_grad)

    reduced_test_loss = partial(test_loss_function, model=model, in_graphs=test_graph, ref_graph=ref_test_graph)

    rng, rng_a, rng_b = jax.random.split(rng, 3)
    rngs = {"reparametrize": rng_a, "dropout": rng_b}
    train_loss, grads = loss_grad_fn(state.params, rngs=rngs)
    rng, rng_a, rng_b = jax.random.split(rng, 3)
    rngs = {"reparametrize": rng_a, "dropout": rng_b}
    test_loss = reduced_test_loss(state.params, rngs=rngs)

    state = state.apply_gradients(grads=grads)
    return state, train_loss, test_loss
