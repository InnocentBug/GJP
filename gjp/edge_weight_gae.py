from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from .edge_weight_decoder import EdgeWeightDecoder
from .mpg_edge_weight import (
    MessagePassingEW,
    edge_weights_n_edge_loss,
    edge_weights_sharpness_loss,
)


class EdgeWeightGAE(nn.Module):
    max_num_nodes: int
    encoder_stack: Sequence[Sequence[int]]
    node_stack: Sequence[int]
    edge_stack: Sequence[int]
    decoder_mpg_stack: Sequence[Sequence[int]]
    multi_edge_repeat: int = 1
    mlp_kwargs: dict[str, Any] | None = None

    def setup(self):
        self._mlp_kwargs = self.mlp_kwargs
        if self._mlp_kwargs is None:
            self._mlp_kwargs = {}

        self.encoder = MessagePassingEW(node_feature_sizes=self.encoder_stack, edge_feature_sizes=self.encoder_stack, global_feature_sizes=self.encoder_stack, mean_instead_of_sum=False, mlp_kwargs=self._mlp_kwargs)

        self.sigma_encoder = MessagePassingEW(node_feature_sizes=self.encoder_stack, edge_feature_sizes=self.encoder_stack, global_feature_sizes=self.encoder_stack, mean_instead_of_sum=False, mlp_kwargs=self._mlp_kwargs)

        self.decoder = EdgeWeightDecoder(max_nodes=self.max_num_nodes, init_node_stack=self.node_stack, init_edge_stack=self.edge_stack, prob_mpg_stack=self.decoder_mpg_stack, multi_edge_repeat=self.multi_edge_repeat, mlp_kwargs=self._mlp_kwargs)

    def __call__(self, x):
        mu_wo_noise = self.encoder(x).globals
        log_sigma_wo_noise = self.sigma_encoder(x).globals
        sigma_wo_noise = jnp.exp(log_sigma_wo_noise)
        rng = self.make_rng("reparametrize")
        eps = jax.random.normal(rng, mu_wo_noise.shape)
        z_wo_node = mu_wo_noise + sigma_wo_noise * eps  # Reparameterization trick

        # Add the node and edge info after adding noise
        z = jnp.vstack((z_wo_node.transpose(), x.n_node, x.n_edge)).transpose()
        reconstructed, edge_weights = self.decoder(z)
        return reconstructed, edge_weights, mu_wo_noise, log_sigma_wo_noise

    def decode(self, z):
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def encode_decode(self, x):
        mu_wo_node = self.encoder(x).globals
        # Add the node and edge info after adding noise
        z = jnp.vstack((mu_wo_node.transpose(), x.n_node, x.n_edge)).transpose()
        reconstructed, edge_weights = self.decoder(z)
        return reconstructed, edge_weights


def pre_loss_function(params, train_state, in_graphs, rngs, metric_state):
    in_metric = metric_state.apply_fn(metric_state.params, in_graphs).globals[:-1]

    tmp_out_graphs, edge_weights, mu, log_sigma = train_state.apply_fn(params, in_graphs, rngs=rngs)
    embedded_space = tmp_out_graphs.globals[::2]
    embedded_space = embedded_space[:, :-2]

    out_globals = jnp.hstack([in_graphs.globals, in_graphs.globals * 0]).reshape(tmp_out_graphs.globals.shape[0], in_graphs.globals.shape[1])
    out_graphs = tmp_out_graphs._replace(globals=out_globals)

    out_metric = metric_state.apply_fn(metric_state.params, out_graphs, edge_weights).globals[::2]
    out_metric = out_metric[:-1]

    recon_loss = jnp.sqrt(jnp.mean((in_metric - out_metric) ** 2))
    kl_divergence = -0.5 * jnp.sum(1 + log_sigma[:-1] - jnp.square(mu[:-1]) - jnp.exp(log_sigma[:-1]))
    sharp_loss = edge_weights_sharpness_loss(edge_weights[:-1])
    n_edge_compare = in_graphs.n_edge
    n_edge_loss = edge_weights_n_edge_loss(edge_weights[:-1], n_edge_compare[:-1])

    return recon_loss, kl_divergence, sharp_loss, n_edge_loss


def loss_function(params, train_state, in_graphs, rngs, metric_state):
    recon_loss, kl_divergence, sharp_loss, n_edge_loss = pre_loss_function(params, train_state, in_graphs, rngs, metric_state)
    return jnp.sqrt(recon_loss) + kl_divergence + sharp_loss + n_edge_loss


def train_step(batch_train, batch_test, train_state, rng, metric_state):
    loss_fn = partial(loss_function, metric_state=metric_state)
    pre_loss_fn = partial(pre_loss_function, metric_state=metric_state)

    loss_grad_fn = jax.value_and_grad(loss_fn)

    rng, rng_reparametrize_train, rng_reparametrize_test, rng_drop_train, rng_drop_test = jax.random.split(rng, 5)

    rngs_train = {"reparametrize": rng_reparametrize_train, "dropout": rng_drop_train}
    rngs_test = {"reparametrize": rng_reparametrize_test, "dropout": rng_drop_test}

    train_recon, train_kl, train_sharp, train_n_edge = pre_loss_fn(train_state.params, train_state, batch_train, rngs_train)
    test_recon, test_kl, test_sharp, test_n_edge = pre_loss_fn(train_state.params, train_state, batch_test, rngs_test)

    train_loss, grads = loss_grad_fn(train_state.params, train_state, batch_train, rngs_train)
    train_state = train_state.apply_gradients(grads=grads)

    return train_state, train_loss, train_recon, train_kl, train_sharp, train_n_edge, test_recon, test_kl, test_sharp, test_n_edge
